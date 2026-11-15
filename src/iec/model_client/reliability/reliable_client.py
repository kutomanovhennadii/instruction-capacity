"""The reliability layer: a common shell over any provider adapter.

``ReliableModelClient`` is the only owner of retries, of the overall time limit of a
call, of cancellation, of a simple circuit breaker and of the observability of active
network calls. It decides not by the type of a Python exception but by the policy of the
normalized failure given to it by the lower layers
(:mod:`iec.model_client.reliability.failures`), so it works the same way with any provider
implementing :class:`ProviderAdapter`.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from iec.contracts import GenParams, Prompt, RawOutput
from iec.model_client.reliability.failures import (
    ClassifiedFailure,
    ModelCallFailure,
    ModelFailureCategory,
)

#: The overall limit per point is five minutes, the same as the read timeout of one call
#: (``DEFAULT_READ_TIMEOUT_S``). A limit of 180 s would coincide with the ceiling of
#: gpt-4o-mini (an answer of 16 384 tokens takes 173-179 s); gpt-5.4-mini with a ceiling of
#: 128 000 manages 39-64 thousand tokens in 300 s.
#:
#: The limit is a measure of the patience of the instrument: a node that stays silent this
#: long is measured not by waiting but by its outcome. A call that does not fit into it is
#: retried once with the limit doubled (``TIMEOUT_RETRY_FACTOR``); a second silence is the
#: final failure ``PROVIDER_TIMEOUT``, and the point gets no answer (decision 1 of the passport
#: standard).
DEFAULT_OVERALL_DEADLINE_S = 300.0
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_COOLDOWN_S = 30.0
#: A timeout is retried once. A read timeout means the model did not fit into the limit on this
#: node - a property of the node, so a retry under the same limit would give the same result in
#: the same time; the retry gets the limit doubled. Network failures and provider codes (429,
#: 5xx) are retried by their own policy.
TIMEOUT_RETRIES = 1
#: The factor of the overall limit of the attempt that follows a timeout: 600 s instead of 300.
TIMEOUT_RETRY_FACTOR = 2.0


class ProviderAdapter(Protocol):
    """One provider behind the reliable layer: exactly one attempt of a call, without retries."""

    @property
    def provider_id(self) -> str:
        """Return the identifier of the provider, the prefix of the call identifiers."""
        ...

    @property
    def model_id(self) -> str:
        """Return the opaque identifier of the model."""
        ...

    def generate_once(self, prompt: Prompt, params: GenParams, context: CallContext) -> RawOutput:
        """Make exactly one attempt of the call.

        Raises:
            ClassifiedFailure: the attempt failed; its policy tells the reliable layer what to do.
        """
        ...


@dataclass(frozen=True, slots=True)
class CallContext:
    """The context of one model call: its identifier and its overall deadline.

    ``clock`` is the monotonic clock of the reliable layer; tests replace it.
    """

    call_id: str
    deadline_monotonic: float
    clock: Callable[[], float] = time.monotonic

    @property
    def remaining_s(self) -> float:
        """Return the seconds left until the overall deadline, never below zero."""
        return max(0.0, self.deadline_monotonic - self.clock())


@dataclass(frozen=True, slots=True)
class ActiveCall:
    """A network call in flight, as :meth:`ReliableModelClient.in_flight_snapshot` shows it."""

    call_id: str
    model_id: str
    attempt: int
    elapsed_s: float
    remaining_s: float


@dataclass(slots=True)
class ReliableModelClient:
    """The single owner of retries, the overall deadline, cancellation and the circuit breaker.

    Every decision about a failed attempt is read from the policy of the failure: whether a
    retry may cure it, the pause, whether it counts against the circuit and the category of the
    final failure. ``sleep`` and ``clock`` are the test doors.
    """

    adapter: ProviderAdapter
    max_retries: int = 3
    backoff_base_s: float = 1.0
    overall_deadline_s: float = DEFAULT_OVERALL_DEADLINE_S
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    cooldown_s: float = DEFAULT_COOLDOWN_S
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    _cancelled: threading.Event = field(default_factory=threading.Event, init=False)
    _cancel_reason: str = field(default="", init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _failures: int = field(default=0, init=False)
    _open_until: float = field(default=0.0, init=False)
    _probe_in_flight: bool = field(default=False, init=False)
    _in_flight: dict[str, tuple[int, float, CallContext]] = field(default_factory=dict, init=False)

    @property
    def model_id(self) -> str:
        """Return the identifier of the model of the adapter."""
        return self.adapter.model_id

    def cancel(self, reason: str) -> None:
        """Cancel the current and future retries of this client.

        An already started synchronous HTTP call is not destroyed; it ends by its own
        timeout.
        """
        if not self._cancelled.is_set():
            self._cancel_reason = reason
            self._cancelled.set()

    def in_flight_snapshot(self) -> tuple[ActiveCall, ...]:
        """Return the network calls in flight, the longest first."""
        now = self.clock()
        with self._lock:
            items = list(self._in_flight.items())
        calls = [
            ActiveCall(
                call_id=call_id,
                model_id=self.adapter.model_id,
                attempt=attempt,
                elapsed_s=now - started,
                remaining_s=context.remaining_s,
            )
            for call_id, (attempt, started, context) in items
        ]
        return tuple(sorted(calls, key=lambda item: item.elapsed_s, reverse=True))

    def generate(self, prompt: Prompt, params: GenParams) -> RawOutput:
        """Call the model through the adapter with retries, the overall deadline and the breaker.

        Args:
            prompt: The ready prompt of the task.
            params: Technical parameters of the call.

        Returns:
            The raw answer of the first successful attempt.

        Raises:
            ModelCallFailure: the call was cancelled, the circuit is open, the overall deadline
                ran out, or the last failure is not cured by a retry; the category tells which.

        Side effects:
            Calls the provider, pauses between attempts, changes the state of the circuit
            breaker and the list of calls in flight.
        """
        call_id = f"{self.adapter.provider_id}-{uuid.uuid4().hex[:12]}"
        context = CallContext(
            call_id=call_id,
            deadline_monotonic=self.clock() + self.overall_deadline_s,
            clock=self.clock,
        )
        attempts = 1 + max(self.max_retries, 0)
        timeouts_seen = 0

        for attempt in range(1, attempts + 1):
            self._raise_if_cancelled(attempt)
            if context.remaining_s <= 0.0:
                raise self._terminal_timeout(attempt)

            is_probe = self._before_call()
            started = self.clock()
            with self._lock:
                self._in_flight[call_id] = (attempt, started, context)
            try:
                output = self.adapter.generate_once(prompt, params, context)
            except ClassifiedFailure as failure:
                self._record_failure(failure)
                if failure.policy.is_timeout:
                    timeouts_seen += 1
                if not self._should_retry(failure, attempt, attempts, timeouts_seen):
                    raise self._terminal(failure, attempt) from failure
                if failure.policy.is_timeout:
                    # The retry after a timeout gets its own, doubled overall limit.
                    context = CallContext(
                        call_id=call_id,
                        deadline_monotonic=self.clock()
                        + TIMEOUT_RETRY_FACTOR * self.overall_deadline_s,
                        clock=self.clock,
                    )
                self._pause(self._delay(failure, attempt), context, attempt)
                continue
            finally:
                # The probe flag is cleared on ANY outcome of the attempt, not only on a
                # failure counted against the circuit breaker. Otherwise a probe call that
                # failed with a quota, an authentication or a broken response body would
                # leave the flag raised forever, and the client would reject all further
                # calls with "probe is already in flight" even with a fully healthy
                # provider.
                if is_probe:
                    self._release_probe()
                with self._lock:
                    self._in_flight.pop(call_id, None)

            self._record_success()
            return output

        raise ModelCallFailure(
            "model call produced no response",
            category=ModelFailureCategory.PROVIDER_UNAVAILABLE,
            attempts=attempts,
            provider=self.adapter.provider_id,
            model_id=self.adapter.model_id,
        )

    def _raise_if_cancelled(self, attempt: int) -> None:
        """Raise the cancellation failure when the client was cancelled.

        Raises:
            ModelCallFailure: ``CANCELLED`` - :meth:`cancel` was called; ``attempts`` counts the
                attempts made before this one.
        """
        if self._cancelled.is_set():
            raise ModelCallFailure(
                f"model call cancelled: {self._cancel_reason}",
                category=ModelFailureCategory.CANCELLED,
                attempts=max(attempt - 1, 0),
                provider=self.adapter.provider_id,
                model_id=self.adapter.model_id,
            )

    def _should_retry(
        self,
        failure: ClassifiedFailure,
        attempt: int,
        attempts: int,
        timeouts_seen: int,
    ) -> bool:
        """Tell whether another attempt follows the failed one.

        No attempt follows the last one. A timeout is retried at most ``TIMEOUT_RETRIES`` times;
        any other failure when its policy and, for a provider failure, its hint allow it.
        """
        if attempt >= attempts:
            return False
        if failure.policy.is_timeout:
            return timeouts_seen <= TIMEOUT_RETRIES
        return failure.retryable

    def _delay(self, failure: ClassifiedFailure, attempt: int) -> float:
        """Return the pause before the next attempt, seconds.

        A timeout is retried IMMEDIATELY, without a pause: this is what the earlier code did
        (``continue`` right after logging), and changing it would change the observable
        behaviour. The purpose of this retry differs from 429/5xx: not "let the provider catch
        its breath" but "check whether the first request was hung" - there is no reason to wait
        before such a check. Any other failure waits ``backoff_base_s * 2 ** (attempt - 1)`` or
        the longer pause the provider asked for.
        """
        if failure.policy.is_timeout:
            return 0.0
        own = self.backoff_base_s * (2 ** (attempt - 1))
        if failure.retry_after_s is not None:
            return max(own, failure.retry_after_s)
        return own

    def _pause(self, delay_s: float, context: CallContext, attempt: int) -> None:
        """Wait out the pause with one ``sleep`` call, cut by the rest of the overall limit.

        The pause is NOT split into one-second steps. Splitting would be bought too dearly:
        in the synchronous model cancellation does not interrupt a network request already
        in progress anyway, so the only gain of splitting would be an earlier wake-up inside
        the pause - and the price would be a change of observable behaviour. The earlier
        code slept once for the whole pause, and the existing boundary tests pin exactly
        this: ``Retry-After: 7`` must give one sleep of 7 s, not seven sleeps of a second.
        A split record would also quietly break respect for ``Retry-After``.

        Cancellation is checked before the pause and after it: between attempts control
        belongs to the bench, and this is enough for the scenario "the operator stopped the
        series".
        """
        if delay_s <= 0.0:
            self._raise_if_cancelled(attempt + 1)
            return
        remaining = context.remaining_s
        if remaining <= 0.0:
            raise self._terminal_timeout(attempt)
        self._raise_if_cancelled(attempt + 1)
        self.sleep(min(delay_s, remaining))
        self._raise_if_cancelled(attempt + 1)

    def _before_call(self) -> bool:
        """Let the attempt through or refuse it without touching the network.

        Returns:
            ``True`` when this is an allowed probe call after the cooldown. The caller must
            clear the flag through :meth:`_release_probe` in ``finally``.
        """
        now = self.clock()
        with self._lock:
            if self._open_until > now:
                raise ModelCallFailure(
                    "provider circuit is open",
                    category=ModelFailureCategory.PROVIDER_UNAVAILABLE,
                    attempts=0,
                    provider=self.adapter.provider_id,
                    model_id=self.adapter.model_id,
                )
            if self._open_until > 0.0:
                # The cooldown is over. Exactly one probe call is allowed.
                if self._probe_in_flight:
                    raise ModelCallFailure(
                        "provider probe is already in flight",
                        category=ModelFailureCategory.PROVIDER_UNAVAILABLE,
                        attempts=0,
                        provider=self.adapter.provider_id,
                        model_id=self.adapter.model_id,
                    )
                self._probe_in_flight = True
                return True
        return False

    def _release_probe(self) -> None:
        """Clear the sign of a probe call in flight.

        It is called in ``finally`` of the attempt, so it runs on any outcome. If the probe
        call did not reach success, the circuit breaker opens again: a failure of the probe
        of any nature means the provider is not healthy yet - narrowing this to
        "infrastructure" failures is not allowed, otherwise a probe that failed with a
        quota or a broken response body would leave the breaker neither open nor closed.

        Side effects:
            Changes the state of the circuit breaker.
        """
        now = self.clock()
        with self._lock:
            if not self._probe_in_flight:
                # Success has already closed the breaker through ``_record_success``.
                return
            self._probe_in_flight = False
            self._open_until = now + self.cooldown_s
            self._failures = self.failure_threshold

    def _record_success(self) -> None:
        """Close the circuit breaker after a successful attempt."""
        with self._lock:
            self._failures = 0
            self._open_until = 0.0
            self._probe_in_flight = False

    def _record_failure(self, failure: ClassifiedFailure) -> None:
        """Count a failure against the circuit breaker when its policy says it counts.

        The fate of a probe call is decided by ``_release_probe`` in ``finally``: it sees any
        outcome, while this method sees only failures counted against the breaker.
        """
        if not failure.policy.counts_against_circuit:
            return
        now = self.clock()
        with self._lock:
            # The check is under the lock, not before it: the flag is shared by the batch
            # threads.
            if self._probe_in_flight:
                return
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open_until = now + self.cooldown_s

    def _terminal_timeout(self, attempt: int) -> ModelCallFailure:
        """Return the final failure of a call whose overall deadline ran out."""
        return ModelCallFailure(
            "model call exceeded overall deadline",
            category=ModelFailureCategory.TIMEOUT,
            attempts=attempt,
            provider=self.adapter.provider_id,
            model_id=self.adapter.model_id,
        )

    def _terminal(self, failure: ClassifiedFailure, attempt: int) -> ModelCallFailure:
        """Return the final failure of the call in the category of the policy of ``failure``."""
        return ModelCallFailure(
            str(failure),
            category=failure.policy.category,
            attempts=attempt,
            provider=self.adapter.provider_id,
            model_id=self.adapter.model_id,
        )
