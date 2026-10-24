RAW_URL := https://physionet.org/files/sleep-accel/1.0.0/

.PHONY: all data features gate train eval test

all: eval

data/raw/labels:
	mkdir -p data/raw
	cd data/raw && wget -r -N -c -np -nH --cut-dirs=3 -R "index.html*" $(RAW_URL)

data/interim/epochs.parquet: data/raw/labels src/sleepwear/data.py src/sleepwear/config.py
	uv run python -m sleepwear.data

data/processed/features.parquet: data/interim/epochs.parquet src/sleepwear/features.py
	uv run python -m sleepwear.features

data: data/interim/epochs.parquet

features: data/processed/features.parquet

results/tables/gate.csv: data/processed/features.parquet src/sleepwear/models.py
	uv run python -W ignore -m sleepwear.evaluate --gate

gate: results/tables/gate.csv

models/oof_3class.parquet: results/tables/gate.csv src/sleepwear/models.py
	uv run python -W ignore -m sleepwear.models

models/oof_transformer_3class.parquet: results/tables/gate.csv src/sleepwear/transformer.py
	uv run python -W ignore -m sleepwear.transformer

train: models/oof_3class.parquet models/oof_transformer_3class.parquet

results/tables/main.csv: models/oof_3class.parquet models/oof_transformer_3class.parquet src/sleepwear/evaluate.py
	uv run python -W ignore -m sleepwear.evaluate

eval: results/tables/main.csv

test:
	uv run pytest -q
