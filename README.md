# Freight Rate Prediction

LightGBM (Huber loss) model on rate per mile, multiplied by distance and a forecast market-drift factor.
Validated time-forward only (train Jan-Aug, test Sep-Oct). See `DIAGNOSIS.md` and `freight_report.pdf`.

## Setup
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p reports/models
```

## Run
Run from the repository root (the code imports `src.drift_model`):
```bash
python pipeline.py
```
This trains the model on `data/train_test.csv` and writes:
- `validation_predictions.csv` (columns `load_id,predicted_rate`, 12,000 rows)
- `data/december_chart_inputs.csv` with the December predictions filled in

Validate the outputs and create the December chart with the provided scorer:
```bash
python score.py --predictions validation_predictions.csv \
  --december-predictions data/december_chart_inputs.csv --output-dir scorer_results
```
The chart is saved to `scorer_results/candidate_december.png` (a copy is at the repo root).

## Layout
- `pipeline.py`: main entry point
- `src/drift_model.py`: model (Huber GBM, month noise weights, drift forecast)
- `freight_rate_prediction_solution.ipynb`: EDA, data quality, validation walkthrough
- `data/`: input CSVs and the prediction template
- `DIAGNOSIS.md`: data issues, fixes and results
