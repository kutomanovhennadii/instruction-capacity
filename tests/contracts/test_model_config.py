from iec.contracts import GenParams, MeasurementDescription, ModelConfig, Tariff


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


def test_measurement_description_round_trip() -> None:
    description = MeasurementDescription(
        model=ModelConfig(model_id="m", provider="mock", gen_params=GenParams()),
        protocol_version="0.1",
        vocab_id="v1_intersection",
        vocab_hash="sha256:abc",
        tariff=Tariff(input_usd_per_million=0.15, output_usd_per_million=0.6),
        files={"points.jsonl.gz": "00ff"},
    )
    assert MeasurementDescription.from_dict(description.as_dict()) == description
