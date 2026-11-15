"""Model price per 1M tokens - the contract between the model registry and budget accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Tariff:
    """Model price per million tokens, fixed for one measurement.

    The price is a property of the model, not of a run, so its source is a registry next to
    the model adapters: a new model brings its tariff together with its adapter, and the
    configs of measurements do not have to repeat the same price.

    For a finished measurement, however, the price is a historical fact: it was paid at the
    tariff in force at the time of the run. So the tariff of a measurement is fixed by its
    snapshot and read from there afterwards, not from the registry: an edit of the registry
    changes the calculation only for new measurements and does not rewrite the cost of old
    ones retroactively.

    The unit is dollars per 1M tokens, as in provider price lists: converting to the price
    of one token would give numbers with six zeros after the point and lose precision for
    no reason.
    """

    input_usd_per_million: float
    output_usd_per_million: float

    def cost_usd(self, prompt_tokens: int, response_tokens: int) -> float:
        """Return the cost of a call from the tokens actually spent.

        Args:
            prompt_tokens: Input tokens as counted by the provider.
            response_tokens: Answer tokens as counted by the provider.

        Returns:
            Cost in dollars. The only place where tokens are converted to money - there
            must be no second such formula in the code base, otherwise budget accounting
            and the report would disagree on the price of the same measurement.
        """
        return (
            prompt_tokens * self.input_usd_per_million
            + response_tokens * self.output_usd_per_million
        ) / 1_000_000

    def as_dict(self) -> dict[str, float]:
        """Return a serializable snapshot of the tariff."""
        return {
            "input_usd_per_million": self.input_usd_per_million,
            "output_usd_per_million": self.output_usd_per_million,
        }
