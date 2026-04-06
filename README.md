# Fertilizer Recommendation System

This project predicts a recommended fertilizer from soil, crop, and environmental inputs. It includes:

- a reusable training and inference pipeline
- a Streamlit web application
- a training script that regenerates all required model artifacts
- a small test suite for smoke-checking the pipeline

## Project structure

```text
IAI_project/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- fertilizer_recommendation.csv
|-- fertilizer.ipynb
|-- fertilizer_recommender/
|   |-- __init__.py
|   |-- config.py
|   `-- pipeline.py
|-- model/
|   `-- fertilizer_model.pkl
`-- tests/
    `-- test_pipeline.py
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Train the model artifacts

The app needs all four artifacts below:

- `model/fertilizer_model.pkl`
- `model/scaler.pkl`
- `model/label_encoder.pkl`
- `model/cat_encoders.pkl`

Generate them with:

```bash
python train_model.py
```

## Run the app

```bash
streamlit run app.py
```

## Run tests

```bash
pytest
```

## Notes

- The notebook is still included for exploration, but the project no longer depends on manually running it.
- If the app says artifacts are missing, run `python train_model.py` first.
