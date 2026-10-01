# MURA Musculoskeletal X-Ray Abnormality Classifier

This project resizes each radiograph to 128×128 grayscale pixels and trains a balanced scikit-learn logistic-regression classifier. The same deterministic preprocessing is used in the notebook and Streamlit app.

## Dataset
Place `MURA-v1.1/` directly inside this project folder, or under `data/MURA-v1.1/`. It must contain `train/` and `valid/`.

## Install
If you cannot create a virtual environment, use:

```bash
python3 -m pip install --user --break-system-packages --no-cache-dir -r requirements.txt
```

## Run notebook

```bash
python3 -m jupyter notebook notebooks/MURA_Grayscale_Sklearn_Analysis.ipynb
```

Run cells from top to bottom. Keep `QUICK_RUN=True` for the first test. The notebook performs dataset auditing, visualizations, patient-disjoint splitting, training, evaluation, error examples, and model export.

## Run Streamlit

After the notebook completes:

```bash
python3 -m streamlit run app.py
```

The model is exported to `models/mura_grayscale_logistic.joblib`. This remains a coursework prototype, not a medical device or diagnostic tool.
