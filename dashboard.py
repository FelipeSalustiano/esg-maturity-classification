"""
ESG Maturity Monitor — ML Performance Dashboard
Streamlit application for executive visualization and model evaluation.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ESG Monitor · ML Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────
C_BG       = "#0F1724"
C_CARD     = "#1A2332"
C_BORDER   = "#243040"
C_PRIMARY  = "#1B4F72"
C_BLUE     = "#2E86C1"
C_TEAL     = "#17A589"
C_GREEN    = "#1ABC9C"
C_RED      = "#E74C3C"
C_ORANGE   = "#E67E22"
C_PURPLE   = "#7D3C98"
C_TEXT     = "#ECF0F1"
C_MUTED    = "#7F8C8D"

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
/* ── Base ── */
.stApp {{ background-color: {C_BG}; }}
[data-testid="stAppViewContainer"] > .main {{ padding-top: 0rem; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {C_CARD};
    border-right: 1px solid {C_BORDER};
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── KPI card ── */
.kpi-card {{
    background: linear-gradient(145deg, {C_CARD} 0%, #1C2B3A 100%);
    border-radius: 14px;
    padding: 20px 22px 16px 22px;
    border-left: 4px solid {C_BLUE};
    box-shadow: 0 6px 20px rgba(0,0,0,.35);
    margin-bottom: 4px;
    height: 115px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}
.kpi-card.green  {{ border-left-color: {C_GREEN}; }}
.kpi-card.red    {{ border-left-color: {C_RED}; }}
.kpi-card.orange {{ border-left-color: {C_ORANGE}; }}
.kpi-card.purple {{ border-left-color: {C_PURPLE}; }}

.kpi-label {{
    color: {C_MUTED};
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}}
.kpi-value {{
    color: {C_TEXT};
    font-size: 34px;
    font-weight: 700;
    line-height: 1.1;
}}
.kpi-sub {{
    color: {C_MUTED};
    font-size: 11px;
    margin-top: 2px;
}}

/* ── Section title ── */
.section-title {{
    font-size: 16px;
    font-weight: 700;
    color: {C_TEXT};
    letter-spacing: 0.5px;
    padding-bottom: 8px;
    border-bottom: 2px solid {C_BLUE};
    margin-bottom: 16px;
    margin-top: 8px;
}}

/* ── Metric badge (model page) ── */
.metric-card {{
    background: linear-gradient(145deg, {C_CARD} 0%, #1C2B3A 100%);
    border-radius: 14px;
    padding: 22px 20px;
    text-align: center;
    border: 1px solid {C_BORDER};
    box-shadow: 0 6px 20px rgba(0,0,0,.3);
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}}
.metric-value {{ font-size: 36px; font-weight: 800; line-height: 1.0; }}
.metric-label {{
    font-size: 11px;
    font-weight: 600;
    color: {C_MUTED};
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-top: 6px;
}}
.metric-desc {{ font-size: 10.5px; color: {C_MUTED}; margin-top: 4px; }}

/* ── Small summary cards ── */
.mini-card {{
    background: {C_CARD};
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    border: 1px solid {C_BORDER};
}}
.mini-value {{ font-size: 24px; font-weight: 700; color: {C_TEXT}; }}
.mini-label {{ font-size: 11px; color: {C_MUTED}; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    background: {C_CARD} !important;
    border-radius: 10px;
}}

/* ── Divider ── */
hr {{ border-color: {C_BORDER}; margin: 12px 0; }}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data(show_spinner=False)
def load_gold_data() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, "data", "gold", "esg_reporting_gold.parquet")
    return pd.read_parquet(path)


@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(BASE_DIR, "model", "xgboost_model.pkl")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def build_results(_df_len: int) -> dict:
    """
    Load data + model, split, preprocess, predict.
    Returns a dict with metrics, predictions df, confusion matrix, etc.
    """
    df    = load_gold_data()
    model = load_model()

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )

    # ── Preprocessing (fit on train, apply to test) ──
    X_tr = df_train.drop(columns=["target"]).copy()
    X_te = df_test.drop(columns=["target"]).copy()
    y_te = df_test["target"].reset_index(drop=True)

    exc_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cur_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr[["exchange"]] = exc_enc.fit_transform(X_tr[["exchange"]])
    X_tr[["currency"]] = cur_enc.fit_transform(X_tr[["currency"]])

    industry_freq = X_tr["industry"].value_counts().to_dict()
    X_tr["industry"] = X_tr["industry"].map(industry_freq)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_tr)

    X_te[["exchange"]] = exc_enc.transform(X_te[["exchange"]])
    X_te[["currency"]] = cur_enc.transform(X_te[["currency"]])
    X_te["industry"]   = X_te["industry"].map(industry_freq).fillna(0)
    X_te_imp = pd.DataFrame(imputer.transform(X_te), columns=X_te.columns)

    # ── Predict ──
    y_pred = model.predict(X_te_imp)
    y_prob = (
        model.predict_proba(X_te_imp)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    # ── Metrics ──
    metrics = {
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall":    recall_score(y_te, y_pred, zero_division=0),
        "f1":        f1_score(y_te, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_te, y_pred)

    # ── Predictions dataframe (for table) ──
    result_df = df_test[["exchange", "currency", "industry"]].copy().reset_index(drop=True)
    result_df["Real"]     = y_te.map({0: "Não Alto", 1: "Alto"})
    result_df["Previsto"] = pd.Series(y_pred).map({0: "Não Alto", 1: "Alto"})
    result_df["acerto"]   = (y_te.values == y_pred).astype(int)

    feat_imp = (
        dict(zip(["exchange", "currency", "industry"], model.feature_importances_))
        if hasattr(model, "feature_importances_")
        else None
    )

    return {
        "metrics":   metrics,
        "cm":        cm,
        "result_df": result_df,
        "y_true":    y_te.values,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "feat_imp":  feat_imp,
    }


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def metric_color(val: float) -> str:
    if val >= 0.80:
        return C_GREEN
    if val >= 0.65:
        return C_ORANGE
    return C_RED


def chart_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT, size=12),
        margin=dict(t=30, b=30, l=10, r=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT)),
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, zerolinecolor=C_BORDER),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

with st.spinner("Carregando dados e modelo…"):
    try:
        df    = load_gold_data()
        _     = load_model()
        data  = build_results(len(df))
    except Exception as exc:
        st.error(f"❌ Erro ao inicializar o dashboard: {exc}")
        st.stop()

metrics   = data["metrics"]
cm        = data["cm"]
result_df = data["result_df"]
feat_imp  = data["feat_imp"]


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding:28px 0 24px;">
            <div style="font-size:42px;">🌿</div>
            <div style="font-size:19px; font-weight:800; color:{C_TEXT}; margin-top:10px; letter-spacing:0.5px;">
                ESG Monitor
            </div>
            <div style="font-size:12px; color:{C_MUTED}; margin-top:4px;">
                ML Performance Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    page = st.radio(
        "nav",
        ["Visão Geral do Modelo", "Avaliação Detalhada do Modelo"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Quick metrics in sidebar
    acc_color = metric_color(metrics["accuracy"])
    st.markdown(
        f"""
        <div style="padding: 8px 4px;">
            <div style="font-size:11px; color:{C_MUTED}; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">
                Resumo do Modelo
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:{C_MUTED}; font-size:12px;">Algoritmo</span>
                <span style="color:{C_TEXT}; font-size:12px; font-weight:600;">XGBoost</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:{C_MUTED}; font-size:12px;">Tarefa</span>
                <span style="color:{C_TEXT}; font-size:12px; font-weight:600;">Classificação</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:{C_MUTED}; font-size:12px;">Classes</span>
                <span style="color:{C_TEXT}; font-size:12px; font-weight:600;">Alto / Não Alto</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="color:{C_MUTED}; font-size:12px;">Acurácia</span>
                <span style="color:{acc_color}; font-size:13px; font-weight:700;">
                    {metrics['accuracy']:.1%}
                </span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:{C_MUTED}; font-size:12px;">F1-Score</span>
                <span style="color:{metric_color(metrics['f1'])}; font-size:13px; font-weight:700;">
                    {metrics['f1']:.1%}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    total_full = len(df)
    test_size  = len(result_df)
    train_size = total_full - test_size
    st.markdown(
        f"""
        <div style="font-size:11px; color:{C_MUTED}; padding: 4px;">
            <div style="text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">Dataset</div>
            <div style="margin-bottom:4px;">Total: <b style="color:{C_TEXT};">{total_full:,}</b></div>
            <div style="margin-bottom:4px;">Treino: <b style="color:{C_TEXT};">{train_size:,}</b></div>
            <div>Teste: <b style="color:{C_TEXT};">{test_size:,}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════
# PAGE 1 ─ EXECUTIVE DASHBOARD
# ═════════════════════════════════════════════════════════════

if page == "Visão Geral do Modelo":

    # ── Header ──
    st.markdown(
        f"""
        <div style="padding: 24px 0 20px;">
            <h1 style="color:{C_TEXT}; font-size:26px; font-weight:800; margin:0; letter-spacing:-0.3px;">
                Visão Geral
            </h1>
            <p style="color:{C_MUTED}; font-size:13px; margin:6px 0 0;">
                Visão geral das classificações de maturidade ESG e desempenho do modelo no conjunto de teste
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI Cards ──
    total   = len(result_df)
    acertos = int(result_df["acerto"].sum())
    erros   = total - acertos
    t_ac    = acertos / total * 100
    t_er    = erros   / total * 100

    c1, c2, c3, c4, c5 = st.columns(5)

    kpis = [
        (c1, "default", "Total Analisado",  f"{total:,}",      "Registros no conjunto de teste"),
        (c2, "green",   "Acertos",          f"{acertos:,}",    "Previsões corretas"),
        (c3, "red",     "Erros",            f"{erros:,}",      "Previsões incorretas"),
        (c4, "green",   "Taxa de Acerto",   f"{t_ac:.1f}%",    "Accuracy no teste"),
        (c5, "red",     "Taxa de Erro",     f"{t_er:.1f}%",    "Proporção de erros"),
    ]

    value_colors = {
        "default": C_TEXT,
        "green":   C_GREEN,
        "red":     C_RED,
        "orange":  C_ORANGE,
        "purple":  C_PURPLE,
    }

    for col, style, label, value, sub in kpis:
        vc = value_colors.get(style, C_TEXT)
        with col:
            st.markdown(
                f"""
                <div class="kpi-card {style}">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{vc};">{value}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Donut + Acertos vs Erros por classe ──
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown('<div class="section-title">Distribuição das Classes Previstas</div>', unsafe_allow_html=True)
        pred_counts = result_df["Previsto"].value_counts()
        fig_donut = go.Figure(
            go.Pie(
                labels=pred_counts.index,
                values=pred_counts.values,
                hole=0.60,
                marker=dict(
                    colors=[C_TEAL, C_BLUE],
                    line=dict(color=C_BG, width=3),
                ),
                textinfo="percent+label",
                textfont=dict(color=C_TEXT, size=12),
                hovertemplate="<b>%{label}</b><br>Qtd: %{value}<br>%{percent}<extra></extra>",
            )
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=270,
            annotations=[
                dict(
                    text=f"<b>{total}</b><br><span style='font-size:11px'>registros</span>",
                    x=0.5, y=0.5,
                    font=dict(size=18, color=C_TEXT),
                    showarrow=False,
                )
            ],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Acertos e Erros por Classe Prevista</div>', unsafe_allow_html=True)
        grp = (
            result_df.groupby(["Previsto", "acerto"])
            .size()
            .reset_index(name="qtd")
        )
        grp["status_label"] = grp["acerto"].map({1: "Acerto ✅", 0: "Erro ❌"})

        fig_bar = px.bar(
            grp, x="Previsto", y="qtd", color="status_label",
            barmode="group", text="qtd",
            color_discrete_map={"Acerto ✅": C_GREEN, "Erro ❌": C_RED},
        )
        fig_bar.update_traces(
            textposition="outside",
            textfont=dict(color=C_TEXT, size=13),
            marker_line_width=0,
        )
        fig_bar = chart_layout(fig_bar, height=270)
        fig_bar.update_layout(
            legend_title_text="",
            xaxis_title="Classe Prevista",
            yaxis_title="Quantidade",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Feature distributions ──
    st.markdown('<div class="section-title">Distribuição das Variáveis do Dataset</div>', unsafe_allow_html=True)

    ca, cb, cc = st.columns(3)

    with ca:
        top_ind = df["industry"].value_counts().head(12).sort_values()
        fig_ind = go.Figure(
            go.Bar(
                y=top_ind.index,
                x=top_ind.values,
                orientation="h",
                marker=dict(
                    color=top_ind.values,
                    colorscale=[[0, C_PRIMARY], [1, C_BLUE]],
                    showscale=False,
                ),
                text=top_ind.values,
                textposition="outside",
                textfont=dict(color=C_TEXT, size=10),
                hovertemplate="<b>%{y}</b><br>%{x} empresas<extra></extra>",
            )
        )
        fig_ind.update_layout(
            title=dict(text="Top 12 Indústrias", font=dict(color=C_TEXT, size=13), x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C_TEXT, size=11),
            margin=dict(t=36, b=10, l=10, r=40),
            height=370,
            xaxis=dict(gridcolor=C_BORDER, showticklabels=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_ind, use_container_width=True)

    with cb:
        top_exc = df["exchange"].value_counts().head(10)
        colors_exc = [C_TEAL if i == 0 else C_BLUE for i in range(len(top_exc))]
        fig_exc = go.Figure(
            go.Bar(
                x=top_exc.index,
                y=top_exc.values,
                marker=dict(color=colors_exc),
                text=top_exc.values,
                textposition="outside",
                textfont=dict(color=C_TEXT, size=11),
                hovertemplate="<b>%{x}</b><br>%{y} empresas<extra></extra>",
            )
        )
        fig_exc.update_layout(
            title=dict(text="Bolsas de Valores", font=dict(color=C_TEXT, size=13), x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C_TEXT, size=11),
            margin=dict(t=36, b=10, l=10, r=10),
            height=370,
            xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=-30),
            yaxis=dict(gridcolor=C_BORDER),
        )
        st.plotly_chart(fig_exc, use_container_width=True)

    with cc:
        top_cur = df["currency"].value_counts().head(10)
        fig_cur = go.Figure(
            go.Bar(
                x=top_cur.index,
                y=top_cur.values,
                marker=dict(
                    color=top_cur.values,
                    colorscale=[[0, C_PURPLE], [1, "#A569BD"]],
                    showscale=False,
                ),
                text=top_cur.values,
                textposition="outside",
                textfont=dict(color=C_TEXT, size=11),
                hovertemplate="<b>%{x}</b><br>%{y} empresas<extra></extra>",
            )
        )
        fig_cur.update_layout(
            title=dict(text="Moedas de Reporte", font=dict(color=C_TEXT, size=13), x=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C_TEXT, size=11),
            margin=dict(t=36, b=10, l=10, r=10),
            height=370,
            xaxis=dict(gridcolor="rgba(0,0,0,0)", tickangle=-30),
            yaxis=dict(gridcolor=C_BORDER),
        )
        st.plotly_chart(fig_cur, use_container_width=True)


# ═════════════════════════════════════════════════════════════
# PAGE 2 ─ MODEL EVALUATION
# ═════════════════════════════════════════════════════════════

elif page == "Avaliação Detalhada do Modelo":

    # ── Header ──
    st.markdown(
        f"""
        <div style="padding: 24px 0 20px;">
            <h1 style="color:{C_TEXT}; font-size:26px; font-weight:800; margin:0; letter-spacing:-0.3px;">
                Avaliação Detalhada do Modelo — XGBoost
            </h1>
            <p style="color:{C_MUTED}; font-size:13px; margin:6px 0 0;">
                Análise detalhada de desempenho, matriz de confusão e previsões individuais
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Metric cards ──
    st.markdown('<div class="section-title">Métricas de Desempenho</div>', unsafe_allow_html=True)

    m_items = [
        ("Acurácia",  metrics["accuracy"],  "Previsões corretas / total"),
        ("Precisão",  metrics["precision"], "Confiabilidade das previsões +"),
        ("Recall",    metrics["recall"],    "Cobertura dos casos positivos"),
        ("F1-Score",  metrics["f1"],        "Equilíbrio precisão × recall"),
    ]

    mc1, mc2, mc3, mc4 = st.columns(4)
    for col, (label, val, desc) in zip([mc1, mc2, mc3, mc4], m_items):
        color = metric_color(val)
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color};">{val:.1%}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confusion Matrix + Metrics Radar ──
    col_cm, col_met = st.columns([1, 1])

    with col_cm:
        st.markdown('<div class="section-title">Matriz de Confusão</div>', unsafe_allow_html=True)

        labels_cm = ["Não Alto (0)", "Alto (1)"]
        tn, fp, fn, tp = cm.ravel()

        quad_labels = [
            ["Verdadeiro<br>Negativo", "Falso<br>Positivo"],
            ["Falso<br>Negativo",      "Verdadeiro<br>Positivo"],
        ]

        fig_cm = go.Figure(
            go.Heatmap(
                z=cm,
                x=labels_cm,
                y=labels_cm,
                colorscale=[
                    [0.0, "#17202A"],
                    [0.4, C_PRIMARY],
                    [1.0, C_TEAL],
                ],
                showscale=False,
                hovertemplate=(
                    "Real: <b>%{y}</b><br>Previsto: <b>%{x}</b><br>Qtd: <b>%{z}</b><extra></extra>"
                ),
            )
        )
        for i in range(2):
            for j in range(2):
                fig_cm.add_annotation(
                    x=labels_cm[j], y=labels_cm[i],
                    text=(
                        f"<b style='font-size:26px;'>{cm[i][j]}</b><br>"
                        f"<span style='font-size:11px; opacity:0.8;'>{quad_labels[i][j]}</span>"
                    ),
                    showarrow=False,
                    font=dict(color=C_TEXT),
                    align="center",
                )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C_TEXT),
            xaxis=dict(title="Classe Prevista", side="bottom", gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Classe Real", autorange="reversed", gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=50, l=80, r=20),
            height=340,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_met:
        st.markdown('<div class="section-title">Comparativo de Métricas</div>', unsafe_allow_html=True)

        m_names = ["Acurácia", "Precisão", "Recall", "F1-Score"]
        m_vals  = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
        bar_colors = [metric_color(v) for v in m_vals]

        fig_met = go.Figure(
            go.Bar(
                x=m_names,
                y=m_vals,
                marker=dict(color=bar_colors, opacity=0.88, line=dict(width=0)),
                text=[f"{v:.1%}" for v in m_vals],
                textposition="outside",
                textfont=dict(color=C_TEXT, size=14, family="monospace"),
                hovertemplate="<b>%{x}</b>: %{y:.1%}<extra></extra>",
                width=0.5,
            )
        )
        fig_met.add_hline(
            y=0.8,
            line_dash="dot",
            line_color=C_ORANGE,
            line_width=1.5,
            annotation_text="  80% — referência",
            annotation_font_color=C_ORANGE,
            annotation_font_size=11,
        )
        fig_met.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C_TEXT),
            yaxis=dict(
                range=[0, 1.15],
                tickformat=".0%",
                gridcolor=C_BORDER,
                title="Score",
            ),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=30, b=20, l=10, r=10),
            height=340,
            showlegend=False,
        )
        st.plotly_chart(fig_met, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature importance ──
    if feat_imp:
        st.markdown('<div class="section-title">Importância das Variáveis</div>', unsafe_allow_html=True)

        fi_df = (
            pd.DataFrame({"Feature": list(feat_imp.keys()), "Importância": list(feat_imp.values())})
            .sort_values("Importância")
        )
        feat_display = {"exchange": "Bolsa (exchange)", "currency": "Moeda (currency)", "industry": "Indústria (industry)"}
        fi_df["Feature"] = fi_df["Feature"].map(feat_display)
        fi_colors = [C_TEAL, C_BLUE, C_PURPLE][-len(fi_df):]

        fig_fi = go.Figure(
            go.Bar(
                y=fi_df["Feature"],
                x=fi_df["Importância"],
                orientation="h",
                marker=dict(color=fi_colors[::-1], opacity=0.88),
                text=[f"{v:.3f}" for v in fi_df["Importância"]],
                textposition="outside",
                textfont=dict(color=C_TEXT, size=13, family="monospace"),
                hovertemplate="<b>%{y}</b><br>Importância: %{x:.4f}<extra></extra>",
                width=0.5,
            )
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=C_TEXT),
            xaxis=dict(gridcolor=C_BORDER, title="Importância Relativa"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=20, l=10, r=60),
            height=200,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Predictions table ──
    st.markdown('<div class="section-title">Registros Analisados — Detalhe das Previsões</div>', unsafe_allow_html=True)

    # Filters
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        filt_status = st.selectbox("Status", ["Todos", "✅ Acertos", "❌ Erros"])
    with f2:
        filt_class = st.selectbox("Classe Prevista", ["Todas", "Alto", "Não Alto"])
    with f3:
        filt_ind = st.text_input("Filtrar por Indústria", placeholder="Ex: technology, banking…")

    disp = result_df.copy()
    if filt_status == "✅ Acertos":
        disp = disp[disp["acerto"] == 1]
    elif filt_status == "❌ Erros":
        disp = disp[disp["acerto"] == 0]
    if filt_class != "Todas":
        disp = disp[disp["Previsto"] == filt_class]
    if filt_ind:
        disp = disp[disp["industry"].str.contains(filt_ind, case=False, na=False)]

    # Summary row
    st.markdown("<br>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    n_ac = int(disp["acerto"].sum())
    n_er = len(disp) - n_ac
    rate = n_ac / len(disp) * 100 if len(disp) > 0 else 0

    summary_cards = [
        (sc1, C_BLUE,  f"{len(disp):,}", "Registros exibidos"),
        (sc2, C_GREEN, f"{n_ac:,}",      "Acertos"),
        (sc3, C_RED,   f"{n_er:,}",      "Erros"),
        (sc4, metric_color(rate / 100), f"{rate:.1f}%", "Taxa de acerto filtrada"),
    ]
    for col, color, val, lbl in summary_cards:
        with col:
            st.markdown(
                f"""
                <div class="mini-card" style="border-color:{color}33;">
                    <div class="mini-value" style="color:{color};">{val}</div>
                    <div class="mini-label">{lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Build display table
    tbl = disp.copy()
    tbl["Status"] = tbl["acerto"].map({1: "✅ Acerto", 0: "❌ Erro"})
    tbl = tbl.drop(columns=["acerto"])
    tbl.columns = ["Bolsa", "Moeda", "Indústria", "Real", "Previsto", "Status"]
    tbl = tbl.reset_index(drop=True)

    st.dataframe(
        tbl,
        use_container_width=True,
        height=420,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Real":   st.column_config.TextColumn("Real",   width="small"),
            "Previsto": st.column_config.TextColumn("Previsto", width="small"),
            "Bolsa": st.column_config.TextColumn("Bolsa", width="small"),
            "Moeda": st.column_config.TextColumn("Moeda", width="small"),
            "Indústria": st.column_config.TextColumn("Indústria"),
        },
    )

    # Download button
    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Exportar tabela (.csv)",
        data=csv,
        file_name="esg_predictions.csv",
        mime="text/csv",
    )