"""Streamlit app: Late-Delivery Risk for an e-commerce marketplace (Olist)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).parent / "model"
st.set_page_config(page_title="Late-Delivery Predictor", page_icon="📦", layout="centered")


@st.cache_resource
def load_artifacts():
    pipe = joblib.load(MODEL_DIR / "pipeline.joblib")
    meta = json.load(open(MODEL_DIR / "meta.json"))
    sellers = pd.read_csv(MODEL_DIR / "seller_stats.csv")
    return pipe, meta, sellers


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


try:
    pipe, meta, sellers = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run the notebook first so it creates the `model/` folder.")
    st.stop()

thr = meta["threshold"]
st.title("📦 Late-Delivery Predictor")
st.caption("Predicts at purchase time whether an order will arrive after the promised date. "
           "Trained on the Olist Brazilian e-commerce dataset.")

tab1, tab2, tab3 = st.tabs(["Order risk", "Seller page", "About the model"])

# ---------------------------------------------------------------- Order risk
with tab1:
    c1, c2 = st.columns(2)
    category = c1.selectbox("Product category", meta["categories"])
    price = c2.number_input("Order price", min_value=1.0, max_value=20000.0, value=120.0, step=10.0)
    freight = c1.number_input("Freight cost", min_value=0.0, max_value=2000.0, value=20.0, step=1.0)
    n_items = c2.number_input("Number of items", min_value=1, max_value=20, value=1)
    cust_state = c1.selectbox("Customer state", meta["customer_states"])
    sell_state = c2.selectbox("Seller state", meta["seller_states"])
    promised = c1.slider("Promised delivery (days)", 3, 60, 20)
    n_sellers = c2.number_input("Number of different sellers", min_value=1, max_value=5, value=1)
    d = st.date_input("Purchase date")
    hour = st.slider("Purchase hour", 0, 23, 14)

    with st.expander("Optional: product size & weight"):
        weight = st.number_input("Weight (g)", min_value=1.0, value=meta["defaults"]["weight_g"])
        volume = st.number_input("Volume (cm³)", min_value=1.0, value=meta["defaults"]["volume_cm3"])

    if st.button("Predict risk", type="primary"):
        a, b = meta["centroids"].get(cust_state), meta["centroids"].get(sell_state)
        dist = haversine(a[0], a[1], b[0], b[1]) if a and b else np.nan
        row = pd.DataFrame([{
            "price": price, "freight_value": freight, "freight_ratio": freight / price,
            "n_items": n_items, "n_sellers": n_sellers, "weight_g": weight, "volume_cm3": volume,
            "photos_qty": meta["defaults"]["photos_qty"], "distance_km": dist,
            "same_state": int(cust_state == sell_state), "promised_days": promised,
            "purchase_month": d.month, "purchase_dow": d.weekday(), "purchase_hour": hour,
            "category": category, "customer_state": cust_state, "seller_state": sell_state}])
        p = float(pipe.predict_proba(row)[0, 1])

        if p >= thr:
            level, icon, action = "HIGH", "🔴", "Warn the customer, prioritise this order with the carrier, or upgrade shipping."
        elif p >= thr / 2:
            level, icon, action = "MEDIUM", "🟠", "Monitor this order; consider a small buffer on the promised date."
        else:
            level, icon, action = "LOW", "🟢", "No action needed."
        st.subheader(f"{icon} {level} late-delivery risk")
        st.metric("Model score", f"{p:.0%}", help="Score from a class-balanced model - use it to rank orders, not as an exact probability.")
        st.write(f"**Suggested action:** {action}")
        st.caption(f"Seller–customer distance used: {dist:,.0f} km · alert threshold: {thr:.0%}")

# ---------------------------------------------------------------- Seller page
with tab2:
    st.write("Late-delivery rate per seller (sellers with at least 10 orders).")
    states = ["All"] + sorted(sellers.seller_state.unique())
    s_state = st.selectbox("Filter by seller state", states)
    min_orders = st.slider("Minimum orders", 10, int(max(sellers.orders.max(), 11)), 10)
    view = sellers[sellers.orders >= min_orders]
    if s_state != "All":
        view = view[view.seller_state == s_state]
    view = view.sort_values("late_rate", ascending=False)
    st.dataframe(view.assign(late_rate=(view.late_rate * 100).round(1)).rename(columns={"late_rate": "late_rate_%"}),
                 width="stretch", hide_index=True)
    if len(view):
        st.bar_chart(view.head(15).set_index("seller_id")["late_rate"])

# ---------------------------------------------------------------- About
with tab3:
    m = meta["metrics"]
    st.write(f"**Model:** {meta['model_name']} (tuned with cross-validation)")
    st.table(pd.DataFrame({"Test-set metric": m}).T)
    st.write(f"Overall late rate in the data: **{meta['base_late_rate']:.1%}**.")
    st.warning("Limitations: Brazilian data from 2016–2018; distance is between state centres, not exact addresses; "
               "weather, strikes and carrier capacity are not modelled.")
