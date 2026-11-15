"""Mirror tests of the configuration of the model under measurement."""

from iec.contracts import GenParams, ModelConfig


def test_model_config_round_trip() -> None:
    config = ModelConfig(
        model_id="gpt-4o-mini-2024-07-18",
        provider="openai",
        gen_params=GenParams(temperature=0.0, max_tokens=16384, top_p=1.0, seed=7),
        tokenizer_name="gpt-4o-mini",
        context_window_tokens=128000,
        response_ceiling_tokens=16384,
    )
    assert ModelConfig.from_dict(config.as_dict()) == config
