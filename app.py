"""Minimal Streamlit dashboard for the Smart Expense Categorizer."""
import json
from html import escape

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.data_preprocessing import auto_map_columns


st.set_page_config(page_title="Spendwise · Expense Dashboard", page_icon="💳", layout="wide", initial_sidebar_state="collapsed")

NAV_ITEMS = ["Overview", "Expenses", "Categorize", "Import CSV"]
CATEGORY_COLORS = {
    "Food": "#e3a18f",
    "Transport": "#86c2a9",
    "Shopping": "#9eafe4",
    "Bills": "#d8bd78",
    "Entertainment": "#c2a0d1",
}
CATEGORY_TINTS = {
    "Food": "rgba(227,161,143,.14)",
    "Transport": "rgba(134,194,169,.14)",
    "Shopping": "rgba(158,175,228,.14)",
    "Bills": "rgba(216,189,120,.14)",
    "Entertainment": "rgba(194,160,209,.14)",
}
CATEGORY_TEXT_COLORS = {
    "Food": "#985743",
    "Transport": "#34785d",
    "Shopping": "#4b5f9b",
    "Bills": "#80651f",
    "Entertainment": "#79518a",
}


@st.cache_resource
def load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {config.MODEL_PATH}")
    return joblib.load(config.MODEL_PATH)


@st.cache_data
def load_transactions():
    path = config.PROCESSED_DIR / "clean_five.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["date"])
    frame["merchant"] = frame.get("description_clean", frame["description_raw"]).fillna("").astype(str).str.title()
    return frame


@st.cache_data
def load_metadata():
    if not config.METADATA_PATH.exists():
        return {}
    with config.METADATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


st.markdown("""
<style>
  :root { --ink:#263746; --muted:#71808a; --blue:#698a96; --line:#d8e4e7; --surface:#ffffff; }
  .stApp, [data-testid="stAppViewContainer"] { background:#e3f2f4; color:var(--ink); }
  [data-testid="stHeader"] { display:none !important; }
  [data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none !important; }
  [data-testid="stAppViewBlockContainer"], [data-testid="stMainBlockContainer"], .block-container {
    padding-top:.8rem !important; padding-bottom:2rem; max-width:1440px;
  }
  [data-testid="stMain"] { padding-top:0 !important; }
  [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line); border-radius:11px; padding:1rem 1.1rem; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div:nth-child(1) [data-testid="stMetric"] { border-top:2px solid #9eafe4; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div:nth-child(2) [data-testid="stMetric"] { border-top:2px solid #86c2a9; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div:nth-child(3) [data-testid="stMetric"] { border-top:2px solid #e3a18f; }
  [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div:nth-child(4) [data-testid="stMetric"] { border-top:2px solid #d8bd78; }
  [data-testid="stMetricLabel"] p { color:#71808a; font-size:.82rem; }
  [data-testid="stMetricValue"] { color:#263746; font-size:1.55rem; font-weight:650; }
  [data-testid="stMetricDelta"] { color:#47775c; }
  .summary-card { background:#fff; border:1px solid #d8e4e7; border-top:3px solid var(--accent); border-radius:12px; padding:1rem 1.1rem .9rem; min-height:126px; }
  .summary-label { color:#71808a; font-size:.84rem; margin-bottom:.55rem; }
  .summary-value { color:#263746; font-size:1.55rem; line-height:1.15; font-weight:680; letter-spacing:-.02em; }
  .summary-note { color:#73838d; font-size:.8rem; margin-top:.7rem; }
  div[data-testid="stPlotlyChart"] { background:var(--surface); border:1px solid var(--line); border-radius:11px; padding:.35rem; }
  div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
  [data-testid="stSelectbox"] [data-baseweb="select"] > div,
  [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background:#ffffff; color:var(--ink); border-color:#cbdde1;
  }
  [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] > div { border-color:var(--line); }
  .brand-name { color:#263746; font-size:1.08rem; font-weight:700; white-space:nowrap; padding-top:.2rem; }
  .eyebrow { color:#71828c; font-size:.74rem; font-weight:620; letter-spacing:.055em; text-transform:uppercase; margin-bottom:.3rem; }
  .page-title { color:#263746; font-size:1.55rem; font-weight:680; letter-spacing:-.03em; margin:0; }
  .page-subtitle { color:#687b86; font-size:.9rem; margin:.2rem 0 1rem; }
  .section-title { color:#334653; font-size:1rem; font-weight:650; margin:.15rem 0 .15rem; }
  .section-title:before { content:""; display:inline-block; width:3px; height:.82em; margin-right:.48rem; border-radius:3px; vertical-align:-.04em; background:linear-gradient(180deg,#9eafe4 0%,#86c2a9 35%,#d8bd78 68%,#c2a0d1 100%); }
  .section-subtitle { color:#758791; font-size:.8rem; margin:0 0 .7rem; }
  .stButton button { border-radius:8px; font-weight:550; }
  button[kind="primary"] { background:#d8e8eb; border-color:#c6dce0; color:#365968; }
  button[kind="primary"]:hover { background:#cce0e4; border-color:#b5d1d7; color:#2c4d5c; }
  button[kind="secondary"] { background:transparent; border-color:transparent; color:#5e727d; }
  button[kind="secondary"]:hover { background:#d6e9ec; border-color:#c6dfe3; color:#344f5c; }
  hr { border-color:#cadde1; }
  [data-testid="stHorizontalBlock"] { align-items:center; }
  @media (max-width: 760px) { .page-title {font-size:1.45rem;} }
</style>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.info("Place a compatible trained model at `models/best_model.joblib`.")
    st.stop()

metadata = load_metadata()
transactions = load_transactions()

def page_heading(title, subtitle, eyebrow="FINANCE OVERVIEW"):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><h1 class="page-title">{title}</h1><p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def navigate(destination):
    st.session_state.page = destination


def currency(value):
    return f"₹{value:,.0f}"


def summary_card(label, value, note, accent):
    st.markdown(
        f'<div class="summary-card" style="--accent:{accent}">'
        f'<div class="summary-label">{escape(str(label))}</div>'
        f'<div class="summary-value">{escape(str(value))}</div>'
        f'<div class="summary-note">{escape(str(note))}</div></div>',
        unsafe_allow_html=True,
    )


def category_styler(frame, column):
    if column not in frame.columns:
        return frame

    def cell_style(category):
        name = str(category)
        color = CATEGORY_TEXT_COLORS.get(name, "#586a74")
        tint = CATEGORY_TINTS.get(name, "rgba(197,208,214,.12)")
        return f"color:{color}; background-color:{tint}; font-weight:600;"

    return frame.style.map(cell_style, subset=[column])


def month_summary(frame):
    monthly = frame.assign(month=frame["date"].dt.to_period("M").astype(str)).groupby("month")["amount"].sum().sort_index()
    return monthly


def delta_text(current, previous):
    if previous is None or previous == 0:
        return None
    return f"{(current - previous) / previous * 100:+.1f}% vs previous month"


if "page" not in st.session_state:
    st.session_state.page = NAV_ITEMS[0]

header_columns = st.columns([2.2, 1, 1, 1, 1])
with header_columns[0]:
    logo_column, name_column = st.columns([.55, 2.8], vertical_alignment="center")
    with logo_column:
        st.image("assets/spendwise-logo.svg", width=36)
    with name_column:
        st.markdown('<div class="brand-name">Spendwise</div>', unsafe_allow_html=True)
for column, item in zip(header_columns[1:], NAV_ITEMS):
    with column:
        st.button(item, key=f"nav_{item}", type="primary" if st.session_state.page == item else "secondary",
                  width="stretch", on_click=navigate, args=(item,))
st.divider()
page = st.session_state.page


if page == "Overview":
    page_heading("Dashboard Overview", "A simple view of your spending and recent transactions.")

    if transactions.empty:
        st.warning("No processed transactions found. Add `data/processed/clean_five.csv` to populate the dashboard.")
    else:
        monthly = month_summary(transactions)
        latest_month = monthly.index[-1]
        latest_value = float(monthly.iloc[-1])
        previous_value = float(monthly.iloc[-2]) if len(monthly) > 1 else None
        category_totals = transactions.groupby("category")["amount"].sum().sort_values(ascending=False)
        top_category = category_totals.index[0]

        k1, k2, k3, k4 = st.columns(4)
        summary_card("Total spend", currency(transactions["amount"].sum()),
                     f"{len(transactions):,} transactions", "#9eafe4")
        summary_card(f"Spend · {pd.Period(latest_month).strftime('%b %Y')}", currency(latest_value),
                     delta_text(latest_value, previous_value) or "Latest available month", "#86c2a9")
        summary_card("Top category", top_category, currency(category_totals.iloc[0]), "#e3a18f")
        summary_card("Average expense", currency(transactions["amount"].mean()), "Per transaction", "#d8bd78")

        chart_left, chart_right = st.columns([1.65, 1])
        with chart_left:
            st.markdown('<div class="section-title">Spending analytics</div><div class="section-subtitle">Monthly expense totals</div>', unsafe_allow_html=True)
            monthly_frame = transactions.assign(month=transactions["date"].dt.to_period("M").astype(str))
            monthly_frame = monthly_frame.groupby(["month", "category"], as_index=False)["amount"].sum()
            spending_chart = px.area(monthly_frame, x="month", y="amount", color="category",
                                     color_discrete_map=CATEGORY_COLORS,
                                     category_orders={"category": list(CATEGORY_COLORS)})
            spending_chart.update_traces(opacity=.78, hovertemplate="%{fullData.name}<br>%{x}<br>₹%{y:,.0f}<extra></extra>")
            spending_chart.update_layout(height=300, margin={"l": 8, "r": 12, "t": 10, "b": 8},
                                         paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", showlegend=True,
                                         xaxis_title=None, yaxis_title=None, hovermode="x unified",
                                         font={"color": "#586d78", "size": 11},
                                         legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 10, "color": "#5f737e"}})
            spending_chart.update_xaxes(showgrid=False, linecolor="#d8e4e7", tickfont={"color": "#687d87"})
            spending_chart.update_yaxes(showgrid=True, gridcolor="#e7eef0", zeroline=False, tickprefix="₹", separatethousands=True, tickfont={"color": "#687d87"})
            st.plotly_chart(spending_chart, width="stretch", config={"displayModeBar": False})

        with chart_right:
            st.markdown('<div class="section-title">Spend by category</div><div class="section-subtitle">Share of total expenses</div>', unsafe_allow_html=True)
            donut = px.pie(category_totals.rename_axis("category").reset_index(name="amount"),
                           values="amount", names="category", color="category", hole=.68,
                           color_discrete_map=CATEGORY_COLORS,
                           category_orders={"category": list(CATEGORY_COLORS)})
            donut.update_traces(textinfo="percent", textposition="inside", marker={"line": {"color": "#ffffff", "width": 3}},
                                hovertemplate="%{label}<br>₹%{value:,.0f} (%{percent})<extra></extra>")
            donut.update_layout(height=300, margin={"l": 8, "r": 8, "t": 10, "b": 8},
                                paper_bgcolor="#ffffff", showlegend=True, legend={"orientation": "h", "y": -.06, "x": .5, "xanchor": "center", "font": {"size": 10, "color": "#5f737e"}},
                                font={"color": "#5f737e", "size": 11}, annotations=[{"text": f"Total<br><b>{currency(transactions['amount'].sum())}</b>", "x": .5, "y": .5, "showarrow": False, "font": {"size": 13, "color": "#263746"}}])
            st.plotly_chart(donut, width="stretch", config={"displayModeBar": False})

        recent_left, recent_right = st.columns([4, 1.1], vertical_alignment="bottom")
        with recent_left:
            st.markdown('<div class="section-title">Recent expenses</div><div class="section-subtitle">Latest transactions in your dataset</div>', unsafe_allow_html=True)
        with recent_right:
            search = st.text_input("Search expenses", placeholder="Search merchant", label_visibility="collapsed")
        recent = transactions.sort_values("date", ascending=False).copy()
        if search.strip():
            recent = recent[recent["merchant"].str.contains(search.strip(), case=False, na=False)]
        display = recent[["date", "merchant", "category", "amount"]].head(10).rename(
            columns={"date": "Date", "merchant": "Merchant", "category": "Category", "amount": "Amount"})
        st.dataframe(category_styler(display, "Category"), width="stretch", hide_index=True,
                     column_config={"Date": st.column_config.DateColumn(format="DD MMM YYYY"),
                                    "Amount": st.column_config.NumberColumn(format="₹ %.2f")})

elif page == "Expenses":
    page_heading("Expenses", "Search and review transactions in the processed dataset.", "TRANSACTION HISTORY")
    if transactions.empty:
        st.info("No processed transactions are available.")
    else:
        filters = st.columns([2, 1.2, 1.2])
        term = filters[0].text_input("Search merchant", placeholder="e.g. Myntra")
        categories = sorted(transactions["category"].dropna().unique())
        selected = filters[1].multiselect("Categories", categories, placeholder="All categories")
        max_amount = float(transactions["amount"].max())
        amount_range = filters[2].slider("Amount range", 0.0, max_amount, (0.0, max_amount), step=max(max_amount / 100, 1.0))
        filtered = transactions.copy()
        if term.strip():
            filtered = filtered[filtered["merchant"].str.contains(term.strip(), case=False, na=False)]
        if selected:
            filtered = filtered[filtered["category"].isin(selected)]
        filtered = filtered[filtered["amount"].between(*amount_range)]
        st.caption(f"{len(filtered):,} matching expenses · {currency(filtered['amount'].sum())} total")
        display = filtered.sort_values("date", ascending=False)[["date", "merchant", "category", "amount"]].rename(
            columns={"date": "Date", "merchant": "Merchant", "category": "Category", "amount": "Amount"})
        st.dataframe(category_styler(display, "Category"), width="stretch", hide_index=True,
                     column_config={"Date": st.column_config.DateColumn(format="DD MMM YYYY"),
                                    "Amount": st.column_config.NumberColumn(format="₹ %.2f")})
        st.download_button("Export CSV", display.to_csv(index=False), "expenses.csv", "text/csv")

elif page == "Categorize":
    page_heading("Categorize an expense", "Get a category suggestion from the saved text classification model.", "SMART CATEGORIZATION")
    with st.container(border=True):
        description = st.text_input("Transaction description", placeholder="e.g. Transaction at Cafe Coffee Day")
        st.caption("The current model uses the description text to predict a category.")
        if st.button("Predict category", type="primary", disabled=not description.strip()):
            prediction = model.predict([description])[0]
            st.success(f"Suggested category: **{prediction}**")
            if hasattr(model, "predict_proba"):
                score = float(model.predict_proba([description]).max())
                st.caption(f"Model score: {score:.0%}. This score is not a calibrated probability.")
        st.caption("Predictions are suggestions. The included dataset appears synthetic, so real-world accuracy may differ.")

elif page == "Import CSV":
    page_heading("Import transactions", "Categorize many transactions and export the results as a CSV.", "BULK CATEGORIZATION")
    st.write("Upload a CSV containing description/merchant and amount columns. Date and category columns are optional.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            mapping = auto_map_columns(raw)
            if "description" not in mapping or "amount" not in mapping:
                st.error(f"Could not find description and amount columns. Found: {list(raw.columns)}")
            else:
                mapped = pd.DataFrame({field: raw[column] for field, column in mapping.items()})
                mapped["description"] = mapped["description"].fillna("").astype(str)
                mapped["amount"] = pd.to_numeric(mapped["amount"], errors="coerce")
                valid = mapped["description"].str.strip().ne("") & mapped["amount"].notna()
                result = mapped.loc[valid].copy()
                if not result.empty:
                    result["predicted_category"] = model.predict(result["description"].tolist())
                st.metric("Categorized rows", f"{len(result):,}")
                st.dataframe(category_styler(result, "predicted_category"), width="stretch", hide_index=True)
                st.download_button("Download categorized CSV", result.to_csv(index=False),
                                   "categorized_expenses.csv", "text/csv", disabled=result.empty)
                if not valid.all():
                    st.caption(f"Skipped {int((~valid).sum())} rows with an empty description or invalid amount.")
        except Exception as exc:
            st.error(f"Could not process this CSV: {exc}")

