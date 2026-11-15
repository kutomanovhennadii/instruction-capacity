# configuration — loading of the YAML files

Every external configuration of the bench is a YAML file validated by a Pydantic model that forbids unknown keys; a data
file of a free form, such as the version reference of a vocabulary, is YAML whose top level is a mapping. This folder is
the one place that reads such files.

## What is here

| Module | Role |
|---|---|
| `yaml_loader.py` | `load_yaml_model`: the file, the YAML and the schema; `load_yaml_mapping`: the file, the YAML and a mapping at the top level; one rule for their errors |

## Key signatures

```python
load_yaml_model(path: Path, model: type[ModelT], *, error_code: str, description: str) -> ModelT
load_yaml_mapping(path: Path, *, error_code: str, description: str) -> dict[str, Any]
```

## Boundaries

A missing file, invalid YAML, a violated schema and a top level that is not a mapping become one `ConfigError` with the
error code of the file and the original cause in `__cause__`, where there is one. The models of the configurations live
with their layers; this folder knows none of them.
