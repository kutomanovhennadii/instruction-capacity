"""A minimal end-to-end run through the universal model contracts."""

from iec.contracts import ModelConfig, Prompt, RawOutput
from iec.model_client import ModelBuilder, ModelClient, Tokenizer


def test_build_generate_returns_project_contract(
    model_builder: ModelBuilder,
    mock_model_config: ModelConfig,
) -> None:
    """The bench needs only the builder and the Protocol contracts, not the adapter."""
    client, tokenizer = model_builder.build(mock_model_config)
    prompt = Prompt(text="lake cloud stone")

    output = _generate_through_contract(
        client,
        prompt,
        mock_model_config,
    )

    assert isinstance(client, ModelClient)
    assert isinstance(tokenizer, Tokenizer)
    assert isinstance(output, RawOutput)
    assert output.text == prompt.text
    assert tokenizer.is_single_token("lake") is True


def _generate_through_contract(
    client: ModelClient,
    prompt: Prompt,
    config: ModelConfig,
) -> RawOutput:
    """Call the model using only the public contracts of the bench."""
    return client.generate(prompt, config.gen_params)
