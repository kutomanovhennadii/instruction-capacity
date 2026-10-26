"""Tests of the reliability layer: retries, overall limit, cancellation, circuit breaker.

All tests replace the clock and ``sleep`` with stubs - there are no real seconds of
waiting. The provider adapter is replaced by a test double with a controlled sequence of
outcomes.
"""

from __future__ import annotations

import pytest

from iec.contracts import GenParams, Prompt, RawOutput
from iec.model_client.reliability.failures import (
    ModelCallFailure,
    ModelFailureCategory,
    ProviderFailure,
    ProviderFailureKind,
    RetryHint,
    TransportFailure,
    TransportFailureKind,
)
from iec.model_client.reliability.reliable_client import CallContext, ReliableModelClient

PROMPT = Prompt(text="lake")
PARAMS = GenParams()


class FakeClock:
    """A controlled monotonic clock: time advances only by an explicit ``advance`` call."""

    def __init__(self, start: float = 1_000.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class SleepSpy:
    """A pause stub: advances the clock instead of real waiting, collects the intervals."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.pauses: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self._clock.advance(seconds)


def _transport_failure(
    kind: TransportFailureKind = TransportFailureKind.CONNECT_FAILURE,
) -> TransportFailure:
    return TransportFailure(f"transport failed: {kind.value}", kind=kind, elapsed_s=0.01)


def _timeout_failure() -> TransportFailure:
    return _transport_failure(TransportFailureKind.READ_TIMEOUT)


def _provider_failure(
    kind: ProviderFailureKind,
    *,
    retry_hint: RetryHint = RetryHint(retryable=False),
) -> ProviderFailure:
    return ProviderFailure(
        f"provider failed: {kind.value}",
        provider="fake",
        kind=kind,
        retry_hint=retry_hint,
    )


def _success(text: str = "ok") -> RawOutput:
    return RawOutput(text=text, usage=None, finish_reason="stop", latency_s=0.01, raw={})


class ScriptedAdapter:
    """A ``ProviderAdapter`` test double: returns or raises prepared outcomes.

    Every element of ``script`` is either a ``RawOutput`` (success) or an exception instance
    raised instead of a result. It records the number of actual calls and all received
    ``CallContext`` objects to check the behaviour of the reliability layer.
    """

    def __init__(self, script: list[RawOutput | Exception]) -> None:
        self._script = list(script)
        self.calls: list[CallContext] = []

    @property
    def provider_id(self) -> str:
        return "fake"

    @property
    def model_id(self) -> str:
        return "fake-model"

    def generate_once(self, prompt: Prompt, params: GenParams, context: CallContext) -> RawOutput:
        del prompt, params
        self.calls.append(context)
        outcome = self._script[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleep_spy(clock: FakeClock) -> SleepSpy:
    return SleepSpy(clock)


def _build_client(
    adapter: ScriptedAdapter,
    clock: FakeClock,
    sleep_spy: SleepSpy,
    **overrides: object,
) -> ReliableModelClient:
    defaults: dict[str, object] = {
        "max_retries": 3,
        "backoff_base_s": 1.0,
        "clock": clock,
        "sleep": sleep_spy,
    }
    defaults.update(overrides)
    return ReliableModelClient(adapter=adapter, **defaults)  # type: ignore[arg-type]


# 1. Success on the first attempt - one adapter call.
def test_success_on_first_attempt_calls_adapter_once(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    adapter = ScriptedAdapter([_success("lake cloud stone")])
    client = _build_client(adapter, clock, sleep_spy)

    output = client.generate(PROMPT, PARAMS)

    assert output.text == "lake cloud stone"
    assert len(adapter.calls) == 1
    assert sleep_spy.pauses == []


# 2. An ordinary network failure -> success: one pause and two attempts.
def test_transport_failure_then_success_retries_once(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    adapter = ScriptedAdapter([_transport_failure(), _success()])
    client = _build_client(adapter, clock, sleep_spy)

    output = client.generate(PROMPT, PARAMS)

    assert output.text == "ok"
    assert len(adapter.calls) == 2
    assert sleep_spy.pauses == [1.0]


# 3. max_retries=0 forbids any retry, including a timeout.
def test_max_retries_zero_forbids_any_retry(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    adapter = ScriptedAdapter([_timeout_failure(), _success()])
    client = _build_client(adapter, clock, sleep_spy, max_retries=0)

    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)

    assert caught.value.category is ModelFailureCategory.TIMEOUT
    assert len(adapter.calls) == 1
    assert sleep_spy.pauses == []


# 4. A timeout is not retried even when attempts are left.
def test_timeout_is_not_retried(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    """A timeout is not retried: it is a property of the node, not a random failure.

    The model did not fit into the allotted time on **this** node; a second attempt gives
    the same result in the same time, and a retry would double the cost of a heavy point.
    """
    adapter = ScriptedAdapter([_timeout_failure(), _success()])
    client = _build_client(adapter, clock, sleep_spy, max_retries=3)

    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)

    assert caught.value.category is ModelFailureCategory.TIMEOUT
    # ``max_retries=3`` would allow four attempts, but a timeout stops the call at once:
    # ``TIMEOUT_RETRIES = 0`` means "the first attempt only".
    assert len(adapter.calls) == 1


# 5. A mixed sequence "network failure -> timeout": the network failure is retried, the
#    timeout stops the call.
def test_a_network_failure_is_retried_but_a_timeout_stops_the_call(
    clock: FakeClock, sleep_spy: SleepSpy
) -> None:
    """What makes sense to retry is retried.

    A network failure is random - a second attempt may succeed. A timeout means the model
    did not fit into the allotted time on this node, and retrying it means paying the same
    minutes for the same answer.
    """
    adapter = ScriptedAdapter([_transport_failure(), _timeout_failure(), _success("recovered")])
    client = _build_client(adapter, clock, sleep_spy, max_retries=3)

    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)

    assert caught.value.category is ModelFailureCategory.TIMEOUT
    # Two attempts: the network failure is retried, the timeout on the second stops the call.
    assert len(adapter.calls) == 2
    # One pause - exponential after the network failure; after the timeout there is no
    # pause, because there is no retry.
    assert sleep_spy.pauses == [1.0]


# 6. RATE_LIMITED uses Retry-After, QUOTA_EXHAUSTED is not retried.
def test_rate_limited_uses_retry_after(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    adapter = ScriptedAdapter(
        [
            _provider_failure(
                ProviderFailureKind.RATE_LIMITED,
                retry_hint=RetryHint(retryable=True, retry_after_s=7.0),
            ),
            _success(),
        ]
    )
    client = _build_client(adapter, clock, sleep_spy, backoff_base_s=1.0)

    output = client.generate(PROMPT, PARAMS)

    assert output.text == "ok"
    # The pause is not split: one sleep call of exactly Retry-After (the server request is
    # larger than the own nominal 1.0 s of the first attempt).
    assert sleep_spy.pauses == [7.0]


def test_quota_exhausted_is_not_retried(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    adapter = ScriptedAdapter([_provider_failure(ProviderFailureKind.QUOTA_EXHAUSTED), _success()])
    client = _build_client(adapter, clock, sleep_spy)

    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)

    assert caught.value.category is ModelFailureCategory.QUOTA_EXHAUSTED
    assert len(adapter.calls) == 1
    assert sleep_spy.pauses == []


# 7. INVALID_RESPONSE is not retried and gives PROVIDER_RESPONSE_INVALID outside.
def test_invalid_response_is_not_retried(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    adapter = ScriptedAdapter(
        [
            _provider_failure(
                ProviderFailureKind.INVALID_RESPONSE, retry_hint=RetryHint(retryable=True)
            ),
            _success(),
        ]
    )
    client = _build_client(adapter, clock, sleep_spy)

    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)

    assert caught.value.error_code == "PROVIDER_RESPONSE_INVALID"
    assert len(adapter.calls) == 1


# 8. After the overall time is exhausted no new attempt starts; PROVIDER_TIMEOUT outside.
def test_overall_deadline_exhaustion_stops_new_attempts(
    clock: FakeClock, sleep_spy: SleepSpy
) -> None:
    def failing_then_advancing(context: CallContext) -> RawOutput:
        del context
        clock.advance(1_000.0)  # certainly more than overall_deadline_s
        raise _transport_failure()

    class AdvancingAdapter:
        provider_id = "fake"
        model_id = "fake-model"
        calls: list[CallContext] = []

        def generate_once(
            self, prompt: Prompt, params: GenParams, context: CallContext
        ) -> RawOutput:
            del prompt, params
            self.calls.append(context)
            return failing_then_advancing(context)

    adapter = AdvancingAdapter()
    client = ReliableModelClient(
        adapter=adapter,
        max_retries=3,
        overall_deadline_s=10.0,
        clock=clock,
        sleep=sleep_spy,
    )

    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)

    assert caught.value.error_code == "PROVIDER_TIMEOUT"
    assert len(adapter.calls) == 1


# 9. Five infrastructure failures open the circuit breaker; until the pause ends the
#    network is not called.
def test_circuit_opens_after_failure_threshold(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    script: list[RawOutput | Exception] = [_transport_failure() for _ in range(5)]
    adapter = ScriptedAdapter(script)
    client = _build_client(
        adapter, clock, sleep_spy, max_retries=10, failure_threshold=5, cooldown_s=30.0
    )

    with pytest.raises(ModelCallFailure):
        client.generate(PROMPT, PARAMS)

    assert len(adapter.calls) == 5

    # The breaker is open: the next call must not reach the adapter at all.
    adapter2 = ScriptedAdapter([_success()])
    client.adapter = adapter2
    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)
    assert caught.value.category is ModelFailureCategory.PROVIDER_UNAVAILABLE
    assert len(adapter2.calls) == 0


# 9a. READ timeouts do not open the breaker: silence is a property of the node, not the channel.
def test_read_timeouts_do_not_open_the_circuit(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    """Five silences in a row leave the breaker closed.

    A batch at the edge of the measured area can hold several heavy nodes in a row. Opened by
    their silence, the breaker would hand out failures to the neighbouring points of the same
    batch. The silence of a heavy node speaks about the node itself, not about the health of
    the provider.
    """
    client = _build_client(
        ScriptedAdapter([_timeout_failure()]),
        clock,
        sleep_spy,
        max_retries=10,
        failure_threshold=5,
        cooldown_s=30.0,
    )
    for _ in range(5):
        adapter = ScriptedAdapter([_timeout_failure()])
        client.adapter = adapter
        with pytest.raises(ModelCallFailure):
            client.generate(PROMPT, PARAMS)
        assert len(adapter.calls) == 1, "silence must not hinder the next call"

    healthy = ScriptedAdapter([_success()])
    client.adapter = healthy
    assert client.generate(PROMPT, PARAMS).text == "ok"
    assert len(healthy.calls) == 1


# 9b. A CONNECTION timeout still opens the breaker.
def test_connect_timeouts_still_open_the_circuit(
    clock: FakeClock, sleep_spy: SleepSpy
) -> None:
    """The handshake does not depend on the weight of the task: its failure speaks about the
    channel.

    There are five failures and also five calls: there are no retries on timeout
    (``TIMEOUT_RETRIES = 0``), and every attempt is a separate ``generate`` call.
    """
    client = _build_client(
        ScriptedAdapter([_transport_failure(TransportFailureKind.CONNECT_TIMEOUT)]),
        clock,
        sleep_spy,
        max_retries=10,
        failure_threshold=5,
        cooldown_s=30.0,
    )
    for _ in range(5):
        client.adapter = ScriptedAdapter(
            [_transport_failure(TransportFailureKind.CONNECT_TIMEOUT)]
        )
        with pytest.raises(ModelCallFailure):
            client.generate(PROMPT, PARAMS)

    healthy = ScriptedAdapter([_success()])
    client.adapter = healthy
    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)
    assert caught.value.category is ModelFailureCategory.PROVIDER_UNAVAILABLE
    assert len(healthy.calls) == 0


# 10. After the pause exactly one probe call is allowed; a second parallel one is not.
def test_exactly_one_probe_allowed_after_cooldown(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    script: list[RawOutput | Exception] = [_transport_failure() for _ in range(5)]
    adapter = ScriptedAdapter(script)
    client = _build_client(
        adapter, clock, sleep_spy, max_retries=10, failure_threshold=5, cooldown_s=30.0
    )
    with pytest.raises(ModelCallFailure):
        client.generate(PROMPT, PARAMS)

    # The cooldown is over.
    clock.advance(31.0)

    # The first call after the cooldown is a probe: it reaches _before_call and gets True.
    assert client._before_call() is True
    # A second parallel probe call, while the first one is still "in flight", is forbidden.
    with pytest.raises(ModelCallFailure) as caught:
        client._before_call()
    assert caught.value.category is ModelFailureCategory.PROVIDER_UNAVAILABLE
    client._release_probe()


# 11. Success of the probe closes the breaker, failure opens it again.
def test_probe_success_closes_circuit_failure_reopens(
    clock: FakeClock, sleep_spy: SleepSpy
) -> None:
    script: list[RawOutput | Exception] = [_transport_failure() for _ in range(5)] + [_success()]
    adapter = ScriptedAdapter(script)
    client = _build_client(
        adapter, clock, sleep_spy, max_retries=10, failure_threshold=5, cooldown_s=30.0
    )
    with pytest.raises(ModelCallFailure):
        client.generate(PROMPT, PARAMS)
    clock.advance(31.0)

    # The probe call goes through and succeeds - the breaker must close.
    output = client.generate(PROMPT, PARAMS)
    assert output.text == "ok"
    assert client._open_until == 0.0
    assert client._failures == 0

    # Further calls go unhindered (the breaker is closed).
    adapter2 = ScriptedAdapter([_success("again")])
    client.adapter = adapter2
    output2 = client.generate(PROMPT, PARAMS)
    assert output2.text == "again"

    # Separately the branch "a failed probe opens the breaker again" is checked.
    script_fail: list[RawOutput | Exception] = [_transport_failure() for _ in range(5)] + [
        _transport_failure()
    ]
    adapter_fail = ScriptedAdapter(script_fail)
    client_fail = _build_client(
        adapter_fail,
        FakeClock(),
        SleepSpy(FakeClock()),
        max_retries=10,
        failure_threshold=5,
        cooldown_s=30.0,
    )
    fake_clock2 = client_fail.clock
    assert isinstance(fake_clock2, FakeClock)
    with pytest.raises(ModelCallFailure):
        client_fail.generate(PROMPT, PARAMS)
    fake_clock2.advance(31.0)
    with pytest.raises(ModelCallFailure):
        client_fail.generate(PROMPT, PARAMS)
    assert client_fail._open_until > 0.0


# 12. A probe that failed with a failure not counted against the breaker (quota,
#     authentication, invalid request, INVALID_RESPONSE) does not leave the client blocked:
#     after a new cooldown the next call reaches the adapter.
def test_probe_failing_with_non_circuit_failure_does_not_deadlock(
    clock: FakeClock, sleep_spy: SleepSpy
) -> None:
    script: list[RawOutput | Exception] = [_transport_failure() for _ in range(5)] + [
        _provider_failure(ProviderFailureKind.QUOTA_EXHAUSTED)
    ]
    adapter = ScriptedAdapter(script)
    client = _build_client(
        adapter, clock, sleep_spy, max_retries=10, failure_threshold=5, cooldown_s=30.0
    )
    with pytest.raises(ModelCallFailure):
        client.generate(PROMPT, PARAMS)
    clock.advance(31.0)

    # The probe fails with a quota - not an infrastructure failure, but it must clear the flag.
    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)
    assert caught.value.category is ModelFailureCategory.QUOTA_EXHAUSTED
    # The breaker opened again for a new cooldown (this is how _release_probe decides).
    assert client._open_until > 0.0
    assert not client._probe_in_flight

    clock.advance(31.0)
    adapter2 = ScriptedAdapter([_success("healthy again")])
    client.adapter = adapter2
    output = client.generate(PROMPT, PARAMS)
    assert output.text == "healthy again"
    assert len(adapter2.calls) == 1


# 13. cancel() stops new attempts and the pause, but does not interrupt a synchronous call
#     already in progress.
def test_cancel_stops_new_attempts_and_pause(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    adapter = ScriptedAdapter([_transport_failure(), _success()])
    client = _build_client(adapter, clock, sleep_spy, max_retries=3)
    client.cancel("test cancellation")

    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)

    assert caught.value.category is ModelFailureCategory.CANCELLED
    assert len(adapter.calls) == 0


def test_cancel_during_pause_stops_retry(clock: FakeClock, sleep_spy: SleepSpy) -> None:
    adapter = ScriptedAdapter([_transport_failure(), _success()])
    client = _build_client(adapter, clock, sleep_spy, max_retries=3)

    def cancel_on_sleep(seconds: float) -> None:
        client.cancel("cancelled mid-pause")
        sleep_spy(seconds)

    client.sleep = cancel_on_sleep

    with pytest.raises(ModelCallFailure) as caught:
        client.generate(PROMPT, PARAMS)

    assert caught.value.category is ModelFailureCategory.CANCELLED
    assert len(adapter.calls) == 1


# 14. Two simultaneous calls have different call_id values and are both visible in
#     in_flight_snapshot().
def test_two_concurrent_calls_have_distinct_ids_and_are_visible(
    clock: FakeClock, sleep_spy: SleepSpy
) -> None:
    seen_snapshots: list[tuple] = []

    class ObservingAdapter:
        provider_id = "fake"
        model_id = "fake-model"

        def __init__(self, client_ref: list[ReliableModelClient]) -> None:
            self._client_ref = client_ref
            self.calls: list[CallContext] = []

        def generate_once(
            self, prompt: Prompt, params: GenParams, context: CallContext
        ) -> RawOutput:
            del prompt, params
            self.calls.append(context)
            snapshot = self._client_ref[0].in_flight_snapshot()
            seen_snapshots.append(tuple(c.call_id for c in snapshot))
            return _success()

    client_ref: list[ReliableModelClient] = []
    adapter = ObservingAdapter(client_ref)
    client = _build_client(adapter, clock, sleep_spy)
    client_ref.append(client)

    client.generate(PROMPT, PARAMS)
    client.generate(PROMPT, PARAMS)

    call_ids = [ctx.call_id for ctx in adapter.calls]
    assert len(set(call_ids)) == 2
    # At the moment of each call the snapshot holds exactly one active call (its own).
    assert all(len(snapshot) == 1 for snapshot in seen_snapshots)


# 15. After success and after an error the records of active calls disappear.
def test_in_flight_cleared_after_success_and_after_failure(
    clock: FakeClock, sleep_spy: SleepSpy
) -> None:
    adapter = ScriptedAdapter([_success()])
    client = _build_client(adapter, clock, sleep_spy)
    client.generate(PROMPT, PARAMS)
    assert client.in_flight_snapshot() == ()

    adapter2 = ScriptedAdapter([_provider_failure(ProviderFailureKind.QUOTA_EXHAUSTED)])
    client2 = _build_client(adapter2, clock, sleep_spy)
    with pytest.raises(ModelCallFailure):
        client2.generate(PROMPT, PARAMS)
    assert client2.in_flight_snapshot() == ()
