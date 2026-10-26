"""The reliability layer: a common shell over any provider adapter.

``ReliableModelClient`` is the only owner of retries, of the overall time limit of a
call, of cancellation, of a simple circuit breaker and of the observability of active
network calls. It decides not by the type of a Python exception but by the normalized
failure name given to it by the lower layers (:mod:`iec.model_client.reliability.failures`),
so it works the same way with any provider implementing :class:`ProviderAdapter`.
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
    ModelCallFailure,
    ModelFailureCategory,
    ProviderFailure,
    ProviderFailureKind,
    TransportFailure,
    TransportFailureKind,
)

#: The overall limit per point is five minutes, the same as the read timeout of one call
#: (``DEFAULT_READ_TIMEOUT_S``). A limit of 180 s would coincide with the ceiling of
#: gpt-4o-mini (an answer of 16 384 tokens takes 173-179 s); gpt-5.4-mini with a ceiling of
#: 128 000 manages 39-64 thousand tokens in 300 s.
#:
#: The limit is a measure of the patience of the instrument: a node that stays silent this
#: long is measured not by waiting but by its outcome. A run that does not fit into it is not
#: counted as a failure: it is recorded as a collapse of the 3rd kind from the first word -
#: for the statistics silence and collapse are indistinguishable, there is no text in
#: either case.
DEFAULT_OVERALL_DEADLINE_S = 300.0
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_COOLDOWN_S = 30.0
#: There are no retries on timeout. A read timeout means the model did not fit into the
#: limit (``DEFAULT_READ_TIMEOUT_S``) on this node - a property of the node, not a random
#: network failure, and a second attempt gives the same result in the same time. Network
#: failures and provider codes (429, 5xx), where a retry makes sense, are still retried.
TIMEOUT_RETRIES = 0


class ProviderAdapter(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def generate_once(
        self, prompt: Prompt, params: GenParams, context: "CallContext"
    ) -> RawOutput: ...


@dataclass(frozen=True, slots=True)
class CallContext:
    call_id: str
    deadline_monotonic: float
    clock: Callable[[], float] = time.monotonic

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.deadline_monotonic - self.clock())


@dataclass(frozen=True, slots=True)
class ActiveCall:
    call_id: str
    model_id: str
    attempt: int
    elapsed_s: float
    remaining_s: float


@dataclass(slots=True)
class ReliableModelClient:
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
            except (TransportFailure, ProviderFailure) as failure:
                self._record_failure(failure)
                if self._is_timeout(failure):
                    timeouts_seen += 1
                if not self._should_retry(failure, attempt, attempts, timeouts_seen):
                    raise self._terminal(failure, attempt) from failure
                delay_s = self._delay(failure, attempt)
                self._pause(delay_s, context, attempt)
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
        failure: Exception,
        attempt: int,
        attempts: int,
        timeouts_seen: int,
    ) -> bool:
        if attempt >= attempts:
            return False
        if self._is_timeout(failure):
            return timeouts_seen <= TIMEOUT_RETRIES
        if isinstance(failure, TransportFailure):
            return True
        if isinstance(failure, ProviderFailure):
            if failure.kind is ProviderFailureKind.INVALID_RESPONSE:
                return False
            return failure.retry_hint.retryable
        return False

    def _delay(self, failure: Exception, attempt: int) -> float:
        # A timeout is retried IMMEDIATELY, without a pause: this is what the earlier code
        # did (``continue`` right after logging), and changing it would change the
        # observable behaviour. The purpose of this retry differs from 429/5xx: not "let
        # the provider catch its breath" but "check whether the first request was hung" -
        # there is no reason to wait before such a check.
        if self._is_timeout(failure):
            return 0.0
        own = self.backoff_base_s * (2 ** (attempt - 1))
        if isinstance(failure, ProviderFailure) and failure.retry_hint.retry_after_s is not None:
            return max(own, failure.retry_hint.retry_after_s)
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
        with self._lock:
            self._failures = 0
            self._open_until = 0.0
            self._probe_in_flight = False

    def _record_failure(self, failure: Exception) -> None:
        # The fate of a probe call is decided by ``_release_probe`` in ``finally``: it sees
        # any outcome, while this method sees only failures counted against the breaker.
        if not self._counts_against_circuit(failure):
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

    @classmethod
    def _counts_against_circuit(cls, failure: Exception) -> bool:
        """Whether a failure counts against the breaker - as a sign of an unhealthy provider.

        **A read timeout does not count.** It means the model did not fit into the limit on
        this node, that is a property of the node, not of the provider. A batch at the edge of
        the measured area can hold several heavy nodes in a row; opened by their silence, the
        breaker would hand out failures to the neighbouring points of the same batch.

        A connection timeout still counts: the handshake does not depend on the weight of
        the task, and its failure speaks precisely about the channel.
        """
        if cls._is_timeout(failure):
            return (
                isinstance(failure, TransportFailure)
                and failure.kind is TransportFailureKind.CONNECT_TIMEOUT
            )
        if isinstance(failure, TransportFailure):
            return True
        return (
            isinstance(failure, ProviderFailure)
            and failure.kind is ProviderFailureKind.PROVIDER_OVERLOADED
        )

    @staticmethod
    def _is_timeout(failure: Exception) -> bool:
        return isinstance(failure, TransportFailure) and failure.kind in {
            TransportFailureKind.CONNECT_TIMEOUT,
            TransportFailureKind.READ_TIMEOUT,
            TransportFailureKind.WRITE_TIMEOUT,
        }

    def _terminal_timeout(self, attempt: int) -> ModelCallFailure:
        return ModelCallFailure(
            "model call exceeded overall deadline",
            category=ModelFailureCategory.TIMEOUT,
            attempts=attempt,
            provider=self.adapter.provider_id,
            model_id=self.adapter.model_id,
        )

    def _terminal(self, failure: Exception, attempt: int) -> ModelCallFailure:
        if self._is_timeout(failure):
            category = ModelFailureCategory.TIMEOUT
        elif isinstance(failure, TransportFailure):
            category = ModelFailureCategory.NETWORK_UNAVAILABLE
        elif isinstance(failure, ProviderFailure):
            category = {
                ProviderFailureKind.RATE_LIMITED: ModelFailureCategory.RATE_LIMIT,
                ProviderFailureKind.QUOTA_EXHAUSTED: ModelFailureCategory.QUOTA_EXHAUSTED,
                ProviderFailureKind.AUTHENTICATION_FAILED: ModelFailureCategory.AUTH_FAILURE,
                ProviderFailureKind.PERMISSION_DENIED: ModelFailureCategory.AUTH_FAILURE,
                ProviderFailureKind.MODEL_NOT_FOUND: ModelFailureCategory.INVALID_REQUEST,
                ProviderFailureKind.PROVIDER_OVERLOADED: (
                    ModelFailureCategory.PROVIDER_UNAVAILABLE
                ),
                ProviderFailureKind.INVALID_REQUEST: ModelFailureCategory.INVALID_REQUEST,
                ProviderFailureKind.INVALID_RESPONSE: ModelFailureCategory.INVALID_RESPONSE,
                ProviderFailureKind.UNKNOWN_PROVIDER_FAILURE: (
                    ModelFailureCategory.PROVIDER_UNAVAILABLE
                ),
            }[failure.kind]
        else:
            category = ModelFailureCategory.PROVIDER_UNAVAILABLE
        return ModelCallFailure(
            str(failure),
            category=category,
            attempts=attempt,
            provider=self.adapter.provider_id,
            model_id=self.adapter.model_id,
        )
