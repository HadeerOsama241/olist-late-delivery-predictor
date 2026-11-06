# 📦 Late-Delivery Predictor for an E-commerce Marketplace

Predicts at **purchase time** whether an order will arrive after the promised date, so the marketplace can warn customers and prioritise risky orders.

**🔗 Live app:** _paste your Streamlit link here_
**📸 Screenshot / GIF:** _add here_

## Results (test set)
| Model | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|
| Baseline (majority class) | 0.500 | 0.081 | 0.000 | 0.000 |
| Best tuned model (Gradient Boosting, threshold 0.50) | 0.787 | 0.287 | 0.681 | 0.195 |

The model catches about 68% of late orders. Among the orders it flags, about 19.5% are truly late, compared with an 8.1% base late rate in the data (roughly 2.4x).

## What's inside
- `Olist_Late_Delivery_Project.ipynb` – full pipeline: joins of 7 tables, target creation, EDA, leakage-safe sklearn Pipeline, 3 models + baseline, tuning, threshold selection, error analysis.
- `app.py` – Streamlit app (order risk form + seller late-rate page).
- `model/` – saved pipeline (`pipeline.joblib`), `meta.json`, `seller_stats.csv` (created by the notebook).

## How to run
```bash
pip install -r requirements.txt jupyter
# 1. Download the dataset and unzip the CSVs into ./data
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# 2. Run the notebook (Restart & Run All) -> creates ./model
# 3. Start the app
streamlit run app.py
```

## Deploy
Push the repo (with the `model/` folder, **without** `data/`) to GitHub → https://share.streamlit.io → New app → select `app.py`.

## Limitations
Brazilian data (2016–2018); distance uses state centroids; weather/strikes/carrier capacity not modelled. About 4 out of 5 flagged orders still arrive on time, so the recommended action should be cheap (e.g., a notification).

**Data credit:** Olist, Brazilian E-Commerce Public Dataset – https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
