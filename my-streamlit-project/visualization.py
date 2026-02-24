"""
visualization.py — All chart functions for Supply Chain Intelligence Dashboard.
Includes 2 new charts:
  - plot_model_comparison_bar: compares all 3 trained models
  - plot_train_vs_new_comparison: training data vs uploaded new data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ─────────────────────────────────────────────────────────────
# CORE CHARTS
# ─────────────────────────────────────────────────────────────

def plot_prediction_pie(df):
    counts = df["Predicted_Disruption"].value_counts().reset_index()
    counts.columns = ["Disruption", "Count"]
    counts["Disruption"] = counts["Disruption"].map({1: "Disruption ⚠️", 0: "No Disruption ✅"})
    fig = px.pie(counts, values="Count", names="Disruption",
                 color_discrete_sequence=["#EF553B","#00CC96"],
                 title="Predicted Disruption Split (Your New Data)")
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def plot_probability_histogram(df):
    fig = px.histogram(df, x="Disruption_Probability", nbins=25,
                       color_discrete_sequence=["#636EFA"],
                       title="Disruption Probability Distribution (Your New Data)")
    fig.add_vline(x=0.4, line_dash="dash", line_color="orange",
                  annotation_text="Medium Risk (0.4)", annotation_position="top right")
    fig.add_vline(x=0.7, line_dash="dash", line_color="red",
                  annotation_text="High Risk (0.7)", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion(actual, predictions):
    cm  = confusion_matrix(actual, predictions)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disruption","Disruption"],
                yticklabels=["No Disruption","Disruption"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_realtime_gauge(prob):
    color = "#00CC96" if prob < 0.4 else ("#FFA500" if prob < 0.7 else "#EF553B")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob*100, 1),
        title={"text": "Disruption Risk Score", "font": {"size": 20}},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0,100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 40],  "color": "#d4edda"},
                {"range": [40, 70], "color": "#fff3cd"},
                {"range": [70,100], "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color":"black","width":4},"thickness":0.75,"value":prob*100}
        },
        number={"suffix":"%","font":{"size":28}}
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_trend(hist_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["Time"], y=hist_df["Probability"],
        mode="lines+markers", name="Risk Probability",
        line=dict(color="#636EFA", width=2), marker=dict(size=8)
    ))
    fig.add_hrect(y0=0,   y1=0.4,  fillcolor="green",  opacity=0.07, line_width=0)
    fig.add_hrect(y0=0.4, y1=0.7,  fillcolor="orange", opacity=0.07, line_width=0)
    fig.add_hrect(y0=0.7, y1=1.0,  fillcolor="red",    opacity=0.07, line_width=0)
    fig.update_layout(title="📈 Risk Trend", xaxis_title="Time",
                      yaxis_title="Probability", yaxis=dict(range=[0,1]), height=350)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# NEW: UPLOADED DATA VISUALIZATION ONLY
# ─────────────────────────────────────────────────────────────

def plot_uploaded_data_summary(predictions, probabilities):
    """
    Visualizes ONLY the uploaded dataset.
    Shows:
    - Disruption split
    - Risk distribution
    """

    total = len(predictions)
    disruption_rate = predictions.mean() * 100
    avg_probability = probabilities.mean() * 100

    fig = go.Figure()

    # Bar 1 — Disruption Rate
    fig.add_trace(go.Bar(
        name="Disruption Rate %",
        x=["Uploaded Data"],
        y=[disruption_rate],
        marker_color="#EF553B",
        text=[f"{disruption_rate:.1f}%"],
        textposition="outside"
    ))

    # Bar 2 — Average Risk Probability
    fig.add_trace(go.Bar(
        name="Average Risk Probability %",
        x=["Uploaded Data"],
        y=[avg_probability],
        marker_color="#636EFA",
        text=[f"{avg_probability:.1f}%"],
        textposition="outside"
    ))

    fig.update_layout(
        title="📊 Uploaded Dataset — Risk Overview",
        yaxis=dict(title="Percentage (%)", range=[0, 100]),
        height=400,
        barmode="group",
        legend=dict(orientation="h", y=1.1)
    )

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FEATURE 1 — CRITICAL RISK INDICATORS
# ─────────────────────────────────────────────────────────────

def plot_feature_importance_radar(imp_df):
    top = imp_df.head(8)
    features = top["Feature"].tolist()
    values   = top["Risk_Contribution_%"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], theta=features + [features[0]],
        fill="toself", fillcolor="rgba(99,110,250,0.25)",
        line=dict(color="#636EFA", width=2), name="Risk Contribution %"
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(values)*1.2])),
        title="🕸️ Key Risk Indicator Radar — Top 8 Factors", height=480
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_indicator_heatmap(df, top_features):
    if "disruption_occurred" not in df.columns:
        st.info("'disruption_occurred' needed for this heatmap.")
        return
    records = []
    for feat in top_features:
        if feat not in df.columns: continue
        try:
            binned = pd.cut(df[feat], bins=3, labels=["Low","Med","High"])
            rates  = df.groupby(binned, observed=True)["disruption_occurred"].mean()
            for level, rate in rates.items():
                records.append({"Feature": feat, "Level": str(level), "Disruption_Rate": round(rate,3)})
        except Exception:
            continue
    if not records:
        st.warning("Could not generate heatmap.")
        return
    pivot = pd.DataFrame(records).pivot(index="Feature", columns="Level", values="Disruption_Rate")
    pivot = pivot[[c for c in ["Low","Med","High"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(6, max(3, len(top_features)*0.6)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title("🌡️ Disruption Rate by Feature Level (Your Uploaded Data)")
    plt.tight_layout()
    st.pyplot(fig)


def plot_risk_score_waterfall(imp_df):
    top      = imp_df.head(10).copy()
    values   = top["Risk_Contribution_%"].tolist()
    features = top["Feature"].tolist()
    fig = go.Figure(go.Waterfall(
        name="Risk Score", orientation="v",
        measure=["relative"]*len(features)+["total"],
        x=features+["Total Risk Score"], y=values+[None],
        text=[f"+{v:.1f}%" for v in values]+[""],
        connector={"line":{"color":"rgb(63,63,63)"}},
        increasing={"marker":{"color":"#EF553B"}},
        totals={"marker":{"color":"#636EFA"}},
    ))
    fig.update_layout(title="📉 Cumulative Risk Score Waterfall",
                      xaxis_title="Risk Indicator", yaxis_title="Risk Contribution %",
                      height=430, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FEATURE 2 — DISRUPTION SCENARIOS
# ─────────────────────────────────────────────────────────────

def plot_scenario_comparison_bar(scenario_df):
    df = scenario_df.sort_values("Disruption_Probability", ascending=True)
    colors = ["#EF553B" if p>=0.7 else "#FFA500" if p>=0.4 else "#00CC96" for p in df["Disruption_Probability"]]
    fig = go.Figure(go.Bar(
        x=df["Disruption_Probability"], y=df["Scenario"], orientation="h",
        marker_color=colors, text=[f"{p*100:.1f}%" for p in df["Disruption_Probability"]],
        textposition="outside"
    ))
    fig.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
    fig.add_vline(x=0.7, line_dash="dash", line_color="red",    annotation_text="High Risk")
    fig.update_layout(title="📊 Disruption Probability by Scenario (Your Data Baseline)",
                      xaxis=dict(range=[0,1.15]), height=420)
    st.plotly_chart(fig, use_container_width=True)


def plot_scenario_probability_bullet(scenario_df):
    scenarios = scenario_df["Scenario"].tolist()
    probs     = scenario_df["Disruption_Probability"].tolist()
    fig = go.Figure()
    for idx, y_name in enumerate(scenarios):
        fig.add_shape(type="rect", x0=0,   x1=0.4, y0=idx-0.4, y1=idx+0.4, fillcolor="#d4edda", opacity=0.3, line_width=0)
        fig.add_shape(type="rect", x0=0.4, x1=0.7, y0=idx-0.4, y1=idx+0.4, fillcolor="#fff3cd", opacity=0.3, line_width=0)
        fig.add_shape(type="rect", x0=0.7, x1=1.0, y0=idx-0.4, y1=idx+0.4, fillcolor="#f8d7da", opacity=0.3, line_width=0)
    fig.add_trace(go.Bar(
        x=probs, y=scenarios, orientation="h",
        marker_color=["#EF553B" if p>=0.7 else "#FFA500" if p>=0.4 else "#00CC96" for p in probs],
        width=0.5, text=[f"{p*100:.1f}%" for p in probs], textposition="outside"
    ))
    fig.update_layout(title="🎯 Bullet Chart — Scenario vs. Risk Bands",
                      xaxis=dict(range=[0,1.15]), height=450, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


def plot_scenario_spider(scenarios):
    dimensions = ["Lead Time Stress","Weather Severity","Delay Stress"]
    all_lt = [v["lead_mult"]   for v in scenarios.values()]
    all_ws = [v["weather_val"] for v in scenarios.values()]
    all_ex = [v["extra_risk"]  for v in scenarios.values()]
    fig    = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, (name, cfg) in enumerate(scenarios.items()):
        norm = [
            cfg["lead_mult"]  / max(all_lt),
            cfg["weather_val"]/ max(all_ws) if max(all_ws)>0 else 0,
            cfg["extra_risk"] / max(all_ex) if max(all_ex)>0 else 0,
        ]
        fig.add_trace(go.Scatterpolar(
            r=norm+[norm[0]], theta=dimensions+[dimensions[0]],
            fill="toself", opacity=0.35,
            line=dict(color=colors[i % len(colors)], width=2),
            name=name
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                      title="🕸️ Multi-Factor Stress Spider Chart", height=500)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FEATURE 3 — PROACTIVE DECISIONS
# ─────────────────────────────────────────────────────────────

def plot_decision_timeline(decision_df):
    cat_colors = {"Monitor":"#00CC96","Prepare":"#FFA500","Act":"#EF553B","Crisis":"#7B1E1E"}
    fig = go.Figure()
    for _, row in decision_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Threshold"]], y=[row["Category"]],
            mode="markers+text",
            marker=dict(size=18, color=cat_colors.get(row["Category"],"#636EFA"), symbol="diamond"),
            text=[f"  {row['Action']}"], textposition="middle right", showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=decision_df["Threshold"].tolist(), y=decision_df["Category"].tolist(),
        mode="lines", line=dict(color="gray",width=1,dash="dot"), showlegend=False
    ))
    fig.update_layout(title="🕒 Proactive Decision Timeline",
                      xaxis=dict(title="Disruption Probability Threshold",range=[0,1.05]),
                      yaxis_title="Decision Category", height=400, plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


def plot_decision_impact_matrix(decision_df):
    cat_colors = {"Monitor":"#00CC96","Prepare":"#FFA500","Act":"#EF553B","Crisis":"#7B1E1E"}
    fig = go.Figure()
    for _, row in decision_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Threshold"]], y=[row["Lead_Time_Days"]],
            mode="markers+text",
            marker=dict(size=row["Cost_Impact"]*8, color=cat_colors.get(row["Category"],"#636EFA"),
                        opacity=0.7, line=dict(width=1,color="white")),
            text=row["Action"], textposition="top center",
            name=row["Category"], showlegend=False,
        ))
    fig.update_layout(title="💰 Decision Impact Matrix — Act Early = Lower Cost",
                      xaxis=dict(title="Probability Threshold",range=[0.1,0.95]),
                      yaxis=dict(title="Lead Time Required (days)"),
                      height=450, plot_bgcolor="#f9f9f9")
    st.plotly_chart(fig, use_container_width=True)


def plot_action_priority_funnel(decision_df):
    fig = go.Figure(go.Funnel(
        y=["Monitor","Prepare","Act","Crisis"],
        x=[100,75,50,25],
        textinfo="label+text",
        text=["Low probability — routine checks","Rising risk — build contingency",
              "High risk — switch vendors / add stock","Crisis — emergency procurement"],
        marker=dict(color=["#00CC96","#FFA500","#EF553B","#7B1E1E"]),
        connector={"line":{"color":"royalblue","dash":"dot","width":3}},
    ))
    fig.update_layout(title="🔽 Action Urgency Funnel", height=400)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FEATURE 4 — TRANSPORT DELAY RISK
# ─────────────────────────────────────────────────────────────

def plot_transport_delay_gauge(transport_df):
    route_risk = transport_df.groupby("Route")["Disruption_Risk"].mean().reset_index()
    worst      = route_risk.sort_values("Disruption_Risk",ascending=False).iloc[0]
    prob       = worst["Disruption_Risk"]
    color      = "#00CC96" if prob<0.4 else ("#FFA500" if prob<0.7 else "#EF553B")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        title={"text":f"Worst Route: {worst['Route']}","font":{"size":14}},
        gauge={"axis":{"range":[0,100]},"bar":{"color":color},
               "steps":[{"range":[0,40],"color":"#d4edda"},
                        {"range":[40,70],"color":"#fff3cd"},
                        {"range":[70,100],"color":"#f8d7da"}]},
        number={"suffix":"%"}
    ))
    fig2 = px.bar(route_risk.sort_values("Disruption_Risk"),
                  x="Disruption_Risk", y="Route", orientation="h",
                  color="Disruption_Risk", color_continuous_scale=["#00CC96","#FFA500","#EF553B"],
                  title="All Routes — Avg Delay Risk")
    fig2.add_vline(x=0.4,line_dash="dash",line_color="orange")
    fig2.add_vline(x=0.7,line_dash="dash",line_color="red")
    fig2.update_layout(height=300)
    col1,col2 = st.columns([1,1.5])
    with col1: fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)
    with col2: st.plotly_chart(fig2, use_container_width=True)


def plot_delay_route_heatmap(transport_df):
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = transport_df.groupby(["Route","Month"])["Delay_Prob"].mean().reset_index()
    pt    = pivot.pivot(index="Route",columns="Month",values="Delay_Prob")
    pt    = pt.reindex(columns=[m for m in month_order if m in pt.columns])
    fig, ax = plt.subplots(figsize=(12, max(3, len(pt)*0.7)))
    sns.heatmap(pt, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=0.7,
                linewidths=0.5, ax=ax, cbar_kws={"label":"Delay Probability"})
    ax.set_title("🗓️ Route × Month Delay Probability (Your Data)")
    plt.tight_layout()
    st.pyplot(fig)


def plot_delay_probability_over_time(transport_df):
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    trend  = transport_df.groupby(["Carrier","Month"])["Delay_Prob"].mean().reset_index()
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, carrier in enumerate(transport_df["Carrier"].unique()):
        df_c = trend[trend["Carrier"]==carrier].copy()
        df_c["Month_Num"] = df_c["Month"].apply(lambda m: month_order.index(m) if m in month_order else 0)
        df_c = df_c.sort_values("Month_Num")
        fig.add_trace(go.Scatter(
            x=df_c["Month"], y=df_c["Delay_Prob"], mode="lines+markers",
            name=carrier, line=dict(color=colors[i%len(colors)],width=2), marker=dict(size=7)
        ))
    fig.add_hline(y=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
    fig.add_hline(y=0.7, line_dash="dash", line_color="red",    annotation_text="High Risk")
    fig.update_layout(title="📈 Monthly Delay Trend by Carrier",
                      xaxis_title="Month", yaxis_title="Avg Delay Probability",
                      yaxis=dict(range=[0,1]), height=400)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FEATURE 5 — INVENTORY / BUFFER STOCK
# ─────────────────────────────────────────────────────────────

def plot_inventory_level_chart(inv_df):
    below  = inv_df["Current_Stock"] < inv_df["Reorder_Point"]
    colors = ["#EF553B" if b else "#636EFA" for b in below]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current Stock",  x=inv_df["SKU"], y=inv_df["Current_Stock"],  marker_color=colors, opacity=0.9))
    fig.add_trace(go.Bar(name="Safety Stock",   x=inv_df["SKU"], y=inv_df["Safety_Stock"],   marker_color="#FFA500", opacity=0.8))
    fig.add_trace(go.Bar(name="Reorder Point",  x=inv_df["SKU"], y=inv_df["Reorder_Point"],  marker_color="#00CC96", opacity=0.8))
    fig.update_layout(barmode="group",
                      title="📊 Inventory Level vs. Safety Stock & Reorder Point (Red = Below Reorder)",
                      xaxis_title="SKU", yaxis_title="Units",
                      height=430, legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig, use_container_width=True)


def plot_buffer_stock_waterfall(inv_df):
    df     = inv_df.sort_values("Buffer_Gap",ascending=False).copy()
    colors = ["#EF553B" if g>0 else "#00CC96" for g in df["Buffer_Gap"]]
    fig = go.Figure(go.Bar(
        x=df["SKU"], y=df["Buffer_Gap"], marker_color=colors,
        text=[f"+{g}" if g>0 else str(g) for g in df["Buffer_Gap"]], textposition="outside"
    ))
    fig.add_hline(y=0, line_color="black", line_width=1.5)
    fig.update_layout(title="🌊 Buffer Stock Gap — Red = Restock Needed",
                      xaxis_title="SKU", yaxis_title="Buffer Gap (Units)", height=400)
    st.plotly_chart(fig, use_container_width=True)


def plot_stockout_risk_heatmap(inv_df):
    hm = inv_df[["SKU","Days_of_Cover","Disruption_Prob","Lead_Time_Days"]].copy().set_index("SKU")
    hm_norm = hm.copy()
    hm_norm["Days_of_Cover"]  = 1 - (hm["Days_of_Cover"]  / hm["Days_of_Cover"].max())
    hm_norm["Disruption_Prob"]=      hm["Disruption_Prob"]
    hm_norm["Lead_Time_Days"] =      hm["Lead_Time_Days"]  / hm["Lead_Time_Days"].max()
    hm_norm.columns = ["Low Days of Cover (Risk↑)","Disruption Probability","Long Lead Time (Risk↑)"]
    fig, ax = plt.subplots(figsize=(8, max(4, len(inv_df)*0.45)))
    sns.heatmap(hm_norm, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"label":"Normalized Risk (0=Low, 1=High)"})
    ax.set_title("🔥 Stockout Risk Heatmap — SKU × Risk Dimensions")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    st.pyplot(fig)