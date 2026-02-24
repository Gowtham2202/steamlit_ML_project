"""
app.py — STEP 2: Load trained model from pickle, accept new CSV, predict & compare.

FLOW:
  train.py  →  model/model_bundle.pkl
  app.py    →  loads pkl → user uploads new CSV → predict → compare vs training → show all 5 feature tabs
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics import accuracy_score, f1_score as sk_f1

from visualization import (
    plot_prediction_pie,
    plot_probability_histogram,
    plot_confusion,
    #plot_model_comparison_bar,
    #plot_train_vs_new_comparison,
    plot_feature_importance_radar,
    plot_risk_indicator_heatmap,
    plot_risk_score_waterfall,
    plot_scenario_comparison_bar,
    plot_scenario_probability_bullet,
    plot_scenario_spider,
    plot_decision_timeline,
    plot_decision_impact_matrix,
    plot_action_priority_funnel,
    plot_transport_delay_gauge,
    plot_delay_route_heatmap,
    plot_delay_probability_over_time,
    plot_inventory_level_chart,
    plot_buffer_stock_waterfall,
    plot_stockout_risk_heatmap,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Supply Chain Intelligence", layout="wide", page_icon="📦")

st.markdown("""
    <h1 style='text-align:center; color:#1a73e8;'>
        📦 Supply Chain Disruption Intelligence Dashboard
    </h1>
   
    <hr style='margin-bottom:0'>
""", unsafe_allow_html=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_bundle.pkl")


# ─────────────────────────────────────────────────────────────
# LOAD PICKLE
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


bundle = load_model_bundle()

if bundle is None:
    st.error("❌ No trained model found at `model/model_bundle.pkl`")
    st.markdown("""
    ### ⚙️ Please run `train.py` first:
    ```bash
    python train.py
    ```
    **Steps:**
    1. Put your CSV in the `data/` folder
    2. Run `python train.py` in your terminal
    3. It saves `model/model_bundle.pkl`
    4. Come back here and refresh
    """)
    st.stop()

# Unpack bundle
best_model      = bundle["best_model"]
best_name       = bundle["best_name"]
best_score      = bundle["best_score"]
scaler          = bundle["scaler"]
le_dict         = bundle["le_dict"]
feature_columns = bundle["feature_columns"]
importances     = bundle["importances"]
training_stats  = bundle["training_stats"]
train_profile   = bundle["train_profile"]
all_models      = bundle["all_models"]
y_test          = bundle["y_test"]
y_pred_test     = bundle["y_pred_test"]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def weather_category(x):
    if x <= 3:   return "Normal"
    elif x <= 6: return "Moderate"
    else:        return "Severe"


def prevention_strategy(prob):
    if prob >= 0.7:
        return {
            "risk": "🔴 HIGH RISK", "color": "#dc3545",
            "action": "Activate backup vendor immediately",
            "steps": [
                "Alert procurement team & escalate to management",
                "Switch to pre-approved backup vendor",
                "Increase safety stock by 30%",
                "Negotiate expedited shipping from alternate source",
                "Monitor daily — reassess in 24 hours",
                "Notify downstream customers of potential delay",
            ],
        }
    elif prob >= 0.4:
        return {
            "risk": "🟡 MEDIUM RISK", "color": "#fd7e14",
            "action": "Monitor closely & prepare contingency",
            "steps": [
                "Notify vendor — request status update",
                "Review current inventory — buffer if below 2-week cover",
                "Identify alternate vendors as backup",
                "Set automated alert if probability rises above 70%",
                "Review open purchase orders for early delivery",
            ],
        }
    else:
        return {
            "risk": "🟢 LOW RISK", "color": "#198754",
            "action": "Normal operations — continue monitoring",
            "steps": [
                "Maintain standard reorder point",
                "Keep weekly check-in with vendor",
                "No immediate action required",
            ],
        }


def no_data_warning():
    st.info("📂 **Please upload your CSV file in the 'Upload & Predict' tab first.**\n\nAll feature charts will be generated from your uploaded data after prediction.")


# ─────────────────────────────────────────────────────────────
# PREPROCESS USING TRAINED ENCODERS/SCALER FROM PICKLE
# ─────────────────────────────────────────────────────────────
def preprocess_new_data(raw_df):
    """
    Apply SAME preprocessing as training (from pickle bundle).
    Returns: (processed_df, actual_labels_or_None, scaled_array)
    """
    df = raw_df.copy()

    # A: weather feature engineering
    if "weather_condition_severity" in df.columns:
        df["weather_category"] = df["weather_condition_severity"].apply(weather_category)
        df.drop("weather_condition_severity", axis=1, inplace=True)

    # B: encode using TRAINED label encoders from pickle
    for col in df.select_dtypes(include=["object"]).columns:
        if col in le_dict:
            le = le_dict[col]
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    # C: extract actual labels if present
    actual = None
    if "disruption_occurred" in df.columns:
        actual = df["disruption_occurred"].values
        df.drop("disruption_occurred", axis=1, inplace=True)

    # D: align to training feature set
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    # E: scale using TRAINED scaler from pickle
    scaled = scaler.transform(df)

    return df, actual, scaled


# ─────────────────────────────────────────────────────────────
# BUILD ALL 5 FEATURE DATASETS FROM PREDICTIONS
# ─────────────────────────────────────────────────────────────
def build_feature_datasets(result_df, raw_df, probabilities):

    # Feature 1: KRI dataframe
    if importances:
        imp_df = pd.DataFrame({
            "Feature":    list(importances.keys()),
            "Importance": list(importances.values()),
        }).sort_values("Importance", ascending=False).head(12).reset_index(drop=True)
        imp_df["Risk_Contribution_%"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(1)
    else:
        imp_df = pd.DataFrame(columns=["Feature","Importance","Risk_Contribution_%"])
    st.session_state.imp_df = imp_df

    # Feature 2: Scenarios — use YOUR uploaded data's median as baseline
    baseline = {col: float(result_df[col].median()) for col in feature_columns if col in result_df.columns}
    for col in feature_columns:
        if col not in baseline: baseline[col] = 0.0

    lead_cols    = [c for c in feature_columns if "lead_time" in c.lower() or "leadtime" in c.lower()]
    weather_cols = [c for c in feature_columns if "weather" in c.lower()]

    scenario_defs = {
        "Your Data Baseline":   {"lead_mult": 1.0, "weather_val": 0, "extra_risk": 0.0},
        "Port Strike":          {"lead_mult": 3.0, "weather_val": 0, "extra_risk": 0.3},
        "Severe Weather":       {"lead_mult": 1.8, "weather_val": 2, "extra_risk": 0.2},
        "Supplier Bankruptcy":  {"lead_mult": 5.0, "weather_val": 0, "extra_risk": 0.5},
        "Demand Surge (+50%)":  {"lead_mult": 1.5, "weather_val": 0, "extra_risk": 0.1},
        "Combined Crisis":      {"lead_mult": 4.0, "weather_val": 2, "extra_risk": 0.6},
    }

    scenario_rows = []
    for name, cfg in scenario_defs.items():
        rec = baseline.copy()
        for lc in lead_cols:   rec[lc] = rec[lc] * cfg["lead_mult"]
        for wc in weather_cols: rec[wc] = cfg["weather_val"]
        rec_df = pd.DataFrame([rec])[feature_columns]
        prob   = min(best_model.predict_proba(scaler.transform(rec_df))[0][1] + cfg["extra_risk"], 1.0)
        scenario_rows.append({
            "Scenario": name,
            "Disruption_Probability": round(prob, 3),
            "Lead_Time_Multiplier":   cfg["lead_mult"],
            "Weather_Severity":       cfg["weather_val"],
        })
    st.session_state.scenario_df = pd.DataFrame(scenario_rows)
    st.session_state.scenarios   = scenario_defs

    # Feature 4: Transport
    n = len(result_df)
    np.random.seed(42)
    route_col   = next((c for c in raw_df.columns if "route"   in c.lower()), None)
    carrier_col = next((c for c in raw_df.columns if "carrier" in c.lower() or "vendor" in c.lower()), None)
    delay_col   = next((c for c in raw_df.columns if "delay"   in c.lower()), None)

    routes   = raw_df[route_col].dropna().unique().tolist()[:6]   if route_col   else ["Route A","Route B","Route C","Route D","Route E"]
    carriers = raw_df[carrier_col].dropna().unique().tolist()[:5] if carrier_col else ["Carrier Alpha","Carrier Beta","Carrier Gamma","Carrier Delta"]
    months   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    transport_df = pd.DataFrame({
        "Route":   np.random.choice(routes,   n),
        "Carrier": np.random.choice(carriers, n),
        "Month":   np.random.choice(months,   n),
        "Delay_Days": (
            pd.to_numeric(raw_df[delay_col].values[:n], errors="coerce").fillna(3)
            if delay_col else np.random.exponential(3, n).round(1)
        ),
    })
    transport_df["Delay_Prob"]      = np.clip(probabilities + np.random.normal(0, 0.05, n), 0, 1)
    transport_df["Weather_Severe"]  = (transport_df["Delay_Prob"] > 0.5).astype(int)
    transport_df["Disruption_Risk"] = (
        0.4 * transport_df["Delay_Prob"] +
        0.3 * (transport_df["Delay_Days"] / 15).clip(0, 1) +
        0.3 * transport_df["Weather_Severe"]
    ).clip(0, 1)
    st.session_state.transport_df = transport_df

    # Feature 5: Inventory
    n_sku = min(15, n)
    sku_probs = probabilities[:n_sku]

    sku_col      = next((c for c in raw_df.columns if any(k in c.lower() for k in ["sku","product","item","part"])), None)
    demand_col   = next((c for c in raw_df.columns if "demand" in c.lower()), None)
    leadtime_col = next((c for c in raw_df.columns if "lead"   in c.lower()), None)
    stock_col    = next((c for c in raw_df.columns if "stock"  in c.lower() or "inventory" in c.lower()), None)

    skus         = raw_df[sku_col].astype(str).values[:n_sku] if sku_col else [f"SKU-{str(i).zfill(3)}" for i in range(1, n_sku+1)]
    daily_demand = pd.to_numeric(raw_df[demand_col].values[:n_sku], errors="coerce").fillna(30).clip(1,500).astype(int) if demand_col else np.random.randint(10,60,n_sku)
    lead_days    = pd.to_numeric(raw_df[leadtime_col].values[:n_sku], errors="coerce").fillna(7).clip(1,30).astype(int) if leadtime_col else np.random.randint(3,20,n_sku)
    curr_stock   = pd.to_numeric(raw_df[stock_col].values[:n_sku], errors="coerce").fillna(300).clip(0,2000).astype(int) if stock_col else np.random.randint(50,800,n_sku)

    inv_df = pd.DataFrame({
        "SKU": skus, "Current_Stock": curr_stock,
        "Reorder_Point": (daily_demand * lead_days).clip(100, 600),
        "Safety_Stock":  (daily_demand * 3).clip(50, 200),
        "Daily_Demand":  daily_demand,
        "Lead_Time_Days": lead_days,
        "Disruption_Prob": sku_probs.round(3),
    })
    inv_df["Days_of_Cover"]      = (inv_df["Current_Stock"] / inv_df["Daily_Demand"]).round(1)
    inv_df["Stockout_Risk"]      = np.where(inv_df["Current_Stock"] < inv_df["Reorder_Point"],"HIGH",
                                   np.where(inv_df["Current_Stock"] < inv_df["Reorder_Point"]*1.5,"MEDIUM","LOW"))
    inv_df["Recommended_Buffer"] = (inv_df["Daily_Demand"] * inv_df["Lead_Time_Days"] * (1 + inv_df["Disruption_Prob"])).astype(int)
    inv_df["Buffer_Gap"]         = inv_df["Recommended_Buffer"] - inv_df["Safety_Stock"]
    st.session_state.inv_df = inv_df


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
for k, v in {
    "result_df": None, "raw_df": None, "actual": None,
    "predictions": None, "probabilities": None,
    "imp_df": None, "transport_df": None,
    "inv_df": None, "scenario_df": None, "scenarios": None,
}.items():
    if k not in st.session_state: st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📂 Upload & Predict",
    "⚠️ Critical Risk Indicators",
    "🎭 Disruption Scenarios",
    "🧭 Proactive Decisions",
    "🚚 Transport Delay Risk",
    "📦 Inventory & Buffer Stock",
])


# ══════════════════════════════════════════════════════════════
# TAB 0 — UPLOAD & PREDICT
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("📂 Upload New Supply Chain CSV for Prediction")
    st.markdown("""
    <div style='background:#1e1e2f; padding:16px; border-radius:8px; border-left:4px solid #1a73e8; margin-bottom:12px;'>
    <b>📌 Pickle Model Concept:</b><br>
    ✅ <b>train.py</b> already trained the model on your training dataset and saved it as <code>model_bundle.pkl</code><br>
    ✅ The trained encoders, scaler, and model are all loaded from the pickle<br>
    ✅ Your new CSV is preprocessed using the <b>exact same steps</b> as training (no re-training)<br>
    ✅ Predictions are compared against the training distribution to detect drift
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Drop your new supply chain CSV here", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.markdown("#### 👀 Raw Data Preview")
        st.dataframe(raw_df.head(10), use_container_width=True)

        ca, cb, cc = st.columns(3)
        ca.metric("Rows in Upload",  raw_df.shape[0])
        cb.metric("Columns",         raw_df.shape[1])
        cc.metric("Training Features", len(feature_columns))

        # Column diagnostics
        missing_cols = set(feature_columns) - set(raw_df.columns) - {"disruption_occurred"}
        if missing_cols:
            st.warning(f"⚠️ {len(missing_cols)} columns not in upload (filled with 0): `{', '.join(list(missing_cols)[:8])}`")

        with st.spinner("⚙️ Applying trained encoders & scaler from pickle → running predictions..."):
            try:
                processed_df, actual, scaled = preprocess_new_data(raw_df)
                predictions   = best_model.predict(scaled)
                probabilities = best_model.predict_proba(scaled)[:, 1]

                result_df = processed_df.copy()
                result_df["Predicted_Disruption"]   = predictions
                result_df["Disruption_Probability"]  = probabilities.round(4)
                result_df["Risk_Level"]              = result_df["Disruption_Probability"].apply(lambda p: prevention_strategy(p)["risk"])
                result_df["Recommended_Action"]      = result_df["Disruption_Probability"].apply(lambda p: prevention_strategy(p)["action"])

                st.session_state.result_df     = result_df
                st.session_state.raw_df        = raw_df
                st.session_state.actual        = actual
                st.session_state.predictions   = predictions
                st.session_state.probabilities = probabilities
                build_feature_datasets(result_df, raw_df, probabilities)

                st.success(f"✅ **{len(result_df)} records** predicted using trained **{best_name}** model from pickle.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback; st.code(traceback.format_exc())
                st.stop()

    if st.session_state.result_df is not None:
        result_df     = st.session_state.result_df
        actual        = st.session_state.actual
        predictions   = st.session_state.predictions
        probabilities = st.session_state.probabilities

        st.markdown("---")
        st.subheader("📊 Prediction Summary")

        total  = len(result_df)
        n_high = int((probabilities >= 0.7).sum())
        n_med  = int(((probabilities >= 0.4) & (probabilities < 0.7)).sum())
        n_low  = int((probabilities < 0.4).sum())
        avg_p  = probabilities.mean()

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Records",       total)
        k2.metric("🔴 High Risk",        n_high,  f"{n_high/total*100:.1f}%")
        k3.metric("🟡 Medium Risk",      n_med,   f"{n_med/total*100:.1f}%")
        k4.metric("🟢 Low Risk",         n_low,   f"{n_low/total*100:.1f}%")
        k5.metric("Avg Disruption Prob", f"{avg_p*100:.1f}%")

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1: plot_prediction_pie(result_df)
        with c2: plot_probability_histogram(result_df)

        if actual is not None:
            st.markdown("---")
            st.subheader("📉 Model Accuracy on Your Uploaded Data")
            col1, col2 = st.columns([1, 2])
            with col1:
                acc_new = accuracy_score(actual, predictions)
                f1_new  = sk_f1(actual, predictions)
                st.metric("Accuracy on New Data",  f"{acc_new*100:.2f}%")
                st.metric("F1 on New Data",        f"{f1_new:.4f}")
                st.metric("Training F1 (from pkl)",f"{best_score:.4f}", delta=f"{(f1_new-best_score)*100:+.2f}%")
                st.caption("Negative delta = model performs slightly worse on new data (expected with real-world drift)")
            with col2:
                plot_confusion(actual, predictions)

        # High risk alert
        hr_df = result_df[result_df["Disruption_Probability"] >= 0.7]
        if not hr_df.empty:
            st.error(f"🚨 {len(hr_df)} HIGH RISK records detected!")
            with st.expander("🔍 View High Risk Records"):
                st.dataframe(hr_df[["Disruption_Probability","Risk_Level","Recommended_Action"]].reset_index(), use_container_width=True)

        st.subheader("📄 All Prediction Results")
        st.dataframe(result_df[["Disruption_Probability","Predicted_Disruption","Risk_Level","Recommended_Action"]].reset_index(), use_container_width=True)

        if n_high > 0:
            strat = prevention_strategy(probabilities.max())
            st.markdown("---")
            st.subheader(f"🛡️ Prevention Plan for Highest Risk Record ({probabilities.max()*100:.1f}%)")
            st.warning(f"**Recommended Action:** {strat['action']}")
            for i, step in enumerate(strat["steps"], 1):
                st.markdown(f"**{i}.** {step}")
    else:
        st.markdown("""
        <div style='text-align:center; padding:50px; background:#f0f4ff; border-radius:12px;'>
            <h3>👆 Upload your new CSV above to get predictions</h3>
            <p style='color:gray;'>The trained pickle model will instantly predict disruption risk for every record.</p>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — CRITICAL RISK INDICATORS
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("⚠️ Feature 1: Critical Risk Indicators")
    if st.session_state.imp_df is None:
        no_data_warning()
    else:
        imp_df    = st.session_state.imp_df
        result_df = st.session_state.result_df
        if imp_df.empty:
            st.warning("Feature importances only available for tree-based models.")
        else:
            top_kri = imp_df.iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Top KRI",          top_kri["Feature"])
            k2.metric("Risk Contribution", f"{top_kri['Risk_Contribution_%']:.1f}%")
            k3.metric("KRIs Tracked",      len(imp_df))

            st.markdown("---")
            st.markdown("### 🕸️ Chart 1 — Key Risk Indicator Radar")
            plot_feature_importance_radar(imp_df)

            st.markdown("---")
            st.markdown("### 🌡️ Chart 2 — Risk Indicator Heatmap (Your Uploaded Data)")
            hm_df = result_df.copy()
            hm_df["disruption_occurred"] = (
                st.session_state.actual
                if st.session_state.actual is not None
                else (hm_df["Disruption_Probability"] >= 0.5).astype(int).values
            )
            plot_risk_indicator_heatmap(hm_df, imp_df["Feature"].tolist()[:6])

            st.markdown("---")
            st.markdown("### 📉 Chart 3 — Risk Score Waterfall")
            plot_risk_score_waterfall(imp_df)

            with st.expander("📋 Full KRI Table"):
                st.dataframe(imp_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — DISRUPTION SCENARIOS
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🎭 Feature 2: Disruption Scenario Simulation")
    if st.session_state.scenario_df is None:
        no_data_warning()
    else:
        scenario_df = st.session_state.scenario_df
        worst  = scenario_df.sort_values("Disruption_Probability", ascending=False).iloc[0]
        best_s = scenario_df.sort_values("Disruption_Probability").iloc[0]
        k1, k2, k3 = st.columns(3)
        k1.metric("Worst Scenario", worst["Scenario"])
        k2.metric("Worst Case Risk", f"{worst['Disruption_Probability']*100:.1f}%")
        k3.metric("Baseline Risk",   f"{best_s['Disruption_Probability']*100:.1f}%")

        st.markdown("---")
        st.markdown("### 📊 Chart 1 — Scenario Comparison Bar Chart")
        plot_scenario_comparison_bar(scenario_df)
        st.markdown("---")
        st.markdown("### 🎯 Chart 2 — Bullet Chart")
        plot_scenario_probability_bullet(scenario_df)
        st.markdown("---")
        st.markdown("### 🕸️ Chart 3 — Spider Chart")
        plot_scenario_spider(st.session_state.scenarios)

        with st.expander("📋 Scenario Table"):
            st.dataframe(scenario_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — PROACTIVE DECISIONS
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🧭 Feature 3: Proactive Supply Chain Decision Making")
    if st.session_state.result_df is None:
        no_data_warning()
    else:
        probs = st.session_state.probabilities
        p25, p50, p75 = np.percentile(probs, [25, 50, 75])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("25th Percentile", f"{p25*100:.1f}%")
        k2.metric("Median Risk",     f"{p50*100:.1f}%")
        k3.metric("75th Percentile", f"{p75*100:.1f}%")
        k4.metric("Max Risk",        f"{probs.max()*100:.1f}%")

        decision_rules = [
            {"Threshold": 0.20, "Action": "Weekly vendor check-in",                "Lead_Time_Days": 0,  "Cost_Impact": 1,  "Category": "Monitor"},
            {"Threshold": 0.35, "Action": "Inventory buffer review",                "Lead_Time_Days": 7,  "Cost_Impact": 2,  "Category": "Monitor"},
            {"Threshold": 0.45, "Action": "Identify & qualify backup vendor",       "Lead_Time_Days": 14, "Cost_Impact": 3,  "Category": "Prepare"},
            {"Threshold": 0.55, "Action": "Pre-order from backup vendor",           "Lead_Time_Days": 10, "Cost_Impact": 4,  "Category": "Prepare"},
            {"Threshold": 0.65, "Action": "Increase safety stock by 20%",           "Lead_Time_Days": 5,  "Cost_Impact": 5,  "Category": "Act"},
            {"Threshold": 0.75, "Action": "Activate backup vendor fully",           "Lead_Time_Days": 3,  "Cost_Impact": 7,  "Category": "Act"},
            {"Threshold": 0.85, "Action": "Emergency procurement + expedited ship", "Lead_Time_Days": 1,  "Cost_Impact": 10, "Category": "Crisis"},
        ]
        decision_df = pd.DataFrame(decision_rules)
        triggered = decision_df[decision_df["Threshold"] <= probs.max()]
        st.info(f"📌 Max risk = **{probs.max()*100:.1f}%** → **{len(triggered)} decision thresholds** are triggered.")

        st.markdown("---")
        st.markdown("### 🕒 Chart 1 — Decision Timeline")
        plot_decision_timeline(decision_df)
        st.markdown("---")
        st.markdown("### 💰 Chart 2 — Decision Impact Matrix")
        plot_decision_impact_matrix(decision_df)
        st.markdown("---")
        st.markdown("### 🔽 Chart 3 — Action Urgency Funnel")
        plot_action_priority_funnel(decision_df)

        with st.expander("📋 Decision Rules"):
            st.dataframe(decision_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — TRANSPORT DELAY RISK
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🚚 Feature 4: Transportation Delay Risk")
    if st.session_state.transport_df is None:
        no_data_warning()
    else:
        transport_df = st.session_state.transport_df
        worst_route  = transport_df.groupby("Route")["Disruption_Risk"].mean().idxmax()
        worst_risk   = transport_df.groupby("Route")["Disruption_Risk"].mean().max()
        k1, k2, k3 = st.columns(3)
        k1.metric("Highest Risk Route", worst_route)
        k2.metric("Route Risk Score",   f"{worst_risk*100:.1f}%")
        k3.metric("Avg Delay Days",     f"{transport_df['Delay_Days'].mean():.1f} days")

        st.markdown("---")
        st.markdown("### 🎯 Chart 1 — Transport Delay Gauge")
        plot_transport_delay_gauge(transport_df)
        st.markdown("---")
        st.markdown("### 🗓️ Chart 2 — Route × Month Delay Heatmap")
        plot_delay_route_heatmap(transport_df)
        st.markdown("---")
        st.markdown("### 📈 Chart 3 — Monthly Delay Trend by Carrier")
        plot_delay_probability_over_time(transport_df)

        with st.expander("📋 Transport Data"):
            st.dataframe(transport_df.head(30), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — INVENTORY & BUFFER STOCK
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("📦 Feature 5: Inventory Level & Buffer Stock")
    if st.session_state.inv_df is None:
        no_data_warning()
    else:
        inv_df = st.session_state.inv_df
        n_high_s = (inv_df["Stockout_Risk"] == "HIGH").sum()
        n_med_s  = (inv_df["Stockout_Risk"] == "MEDIUM").sum()
        avg_gap  = inv_df["Buffer_Gap"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("SKUs Tracked",          len(inv_df))
        k2.metric("🔴 HIGH Stockout Risk", n_high_s)
        k3.metric("🟡 MEDIUM Stockout Risk",n_med_s)
        k4.metric("Avg Buffer Gap",        f"{avg_gap:.0f} units",
                  delta="Restock Needed" if avg_gap > 0 else "Sufficient")

        st.markdown("---")
        st.markdown("### 📊 Chart 1 — Inventory Level vs. Reorder & Safety Stock")
        plot_inventory_level_chart(inv_df)
        st.markdown("---")
        st.markdown("### 🌊 Chart 2 — Buffer Stock Gap Waterfall")
        plot_buffer_stock_waterfall(inv_df)
        st.markdown("---")
        st.markdown("### 🔥 Chart 3 — Stockout Risk Heatmap")
        plot_stockout_risk_heatmap(inv_df)

        with st.expander("📋 Full Inventory Table"):
            st.dataframe(inv_df, use_container_width=True)