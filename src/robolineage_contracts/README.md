# robolineage_contracts

Python contracts shared across RoboLineage stages. Business modules should exchange stable dataclasses and Pydantic models through this package instead of importing each other's private types.

Run:

```bash
pytest tests/robolineage_contracts -v
pytest tests/robolineage_schemas -v
```