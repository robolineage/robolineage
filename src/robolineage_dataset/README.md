# robolineage_dataset

Dataset lifecycle utilities. `DatasetUpdater` merges accepted manifest entries, computes stable hashes through `robolineage_contracts.pipeline`, and writes immutable `dataset.lock` files for training provenance.

```bash
python -m robolineage_dataset update --train-manifest manifest.jsonl --out datasets/
python -m robolineage_dataset diff datasets/v1/dataset.lock datasets/v2/dataset.lock
```