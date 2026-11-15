"""Loading of the YAML files of the bench: configurations into Pydantic models, plain mappings."""

from iec.infrastructure.configuration.yaml_loader import load_yaml_mapping, load_yaml_model

__all__ = ["load_yaml_mapping", "load_yaml_model"]
