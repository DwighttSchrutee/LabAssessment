"""
app.py — Periscope Labs Loan Default Predictor
Part D: Query Interface (Streamlit)

Run with:
    streamlit run app.py

This app loads pre-computed test predictions and SHAP values generated
by the notebook. Make sure you run the notebook first.
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Risk — Periscope Labs",
    page_icon="🏦",
    layout="wide",
)

DATA_PATH        = "data/test_predictions.csv"
SHAP_PATH        = "data/shap_values.csv"
MODEL_PATH       = "models/model2_behavioral.pkl"
EXPLANATION_PATH = "explanations/high_risk_explanations.md"

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(
            "❌ data/test_predictions.csv not found. "
            "Please run the notebook first to generate predictions."
        )
        st.stop()
    df = pd.read_csv(DATA_PATH)
    shap_df = pd.read_csv(SHAP_PATH) if os.path.exists(SHAP_PATH) else None
    return df, shap_df

df, shap_df = load_data()

FEATURE_COLS = [c for c in df.columns
                if c not in ('borrower_id', 'actual_default', 'risk_score')]

READABLE = {
    'int_rate': 'Interest Rate (%)',
    'dti': 'Debt-to-Income Ratio (%)',
    'fico_range_low': 'FICO Score',
    'grade_num': 'Loan Grade (0=A, 6=G)',
    'loan_amnt': 'Loan Amount ($)',
    'term': 'Loan Term (months)',
    'emp_length': 'Employment Length (years)',
    'log_annual_inc': 'Log Annual Income',
    'revol_util': 'Revolving Utilization (%)',
    'loan_to_income': 'Loan-to-Income Ratio',
    'credit_age_months': 'Credit History Age (months)',
    'mths_since_last_delinq_filled': 'Months Since Last Delinquency',
    'ever_120dpd': '120+ Day Delinquency Flag',
    'revol_bal_to_limit': 'Revolving Balance / Credit Limit',
    'pct_tl_nvr_dlq': '% Accounts Never Delinquent',
    'inq_last_6mths': 'Recent Inquiries (6 months)',
    'pub_rec': 'Public Derogatory Records',
}


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar + title
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Default Risk Dashboard")
st.caption("Periscope Labs — Mini Credit Risk Predictor | Model: Gradient Boosting + Behavioral Features")

st.sidebar.title("Query Interface")
st.sidebar.markdown(
    "Select a query type below or type naturally in the search box."
)

query_type = st.sidebar.radio(
    "Choose query:",
    [
        "🔴 Top 10 Highest-Risk Borrowers",
        "🔍 Explain a Specific Borrower",
        "📊 Portfolio Statistics",
        "💬 Natural Language Query",
    ],
)

# ─────────────────────────────────────────────────────────────────────────────
# Query 1 — Top 10 Highest-Risk Borrowers
# ─────────────────────────────────────────────────────────────────────────────
if query_type == "🔴 Top 10 Highest-Risk Borrowers":
    st.header("Top 10 Highest-Risk Borrowers")
    st.markdown(
        "These are the 10 borrowers in the test set with the highest predicted "
        "default probability. A credit officer should prioritise outreach to these accounts."
    )

    top10 = (
        df[['borrower_id', 'risk_score', 'actual_default',
            'loan_amnt', 'int_rate', 'dti', 'fico_range_low',
            'grade_num', 'loan_to_income', 'revol_util']]
        .sort_values('risk_score', ascending=False)
        .head(10)
        .copy()
    )

    top10['risk_score_pct'] = (top10['risk_score'] * 100).round(1).astype(str) + '%'
    top10['grade'] = top10['grade_num'].apply(
        lambda x: 'ABCDEFG'[min(int(x), 6)] if pd.notna(x) else 'N/A'
    )
    top10['actually_defaulted'] = top10['actual_default'].map({1: '✅ Yes', 0: '❌ No'})

    display_cols = {
        'borrower_id': 'Borrower ID',
        'risk_score_pct': 'Predicted Risk',
        'actually_defaulted': 'Actually Defaulted?',
        'loan_amnt': 'Loan Amount ($)',
        'int_rate': 'Interest Rate (%)',
        'dti': 'DTI (%)',
        'fico_range_low': 'FICO',
        'grade': 'Grade',
    }
    st.dataframe(
        top10[list(display_cols.keys())].rename(columns=display_cols),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader("Risk Score Distribution (Test Set)")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df['risk_score'], bins=50, color='salmon', edgecolor='white', alpha=0.8)
    ax.axvline(df['risk_score'].quantile(0.9), color='red', linestyle='--',
               label='90th percentile')
    ax.set_xlabel('Predicted Default Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predicted Default Risk')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Query 2 — Explain a Specific Borrower
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "🔍 Explain a Specific Borrower":
    st.header("Borrower Risk Explanation")
    st.markdown("Enter a Borrower ID from the test set to get a plain-English explanation.")

    col1, col2 = st.columns([2, 1])
    with col1:
        borrower_input = st.text_input(
            "Borrower ID",
            placeholder="e.g. 1060773",
            help="Use an ID from the test set. Try one of the top-risk IDs from Query 1."
        )

    with col2:
        st.markdown("**Quick picks (high risk):**")
        top5_ids = df.sort_values('risk_score', ascending=False).head(5)['borrower_id'].tolist()
        for bid in top5_ids:
            if st.button(str(int(bid)), key=f'btn_{bid}'):
                borrower_input = str(int(bid))

    if borrower_input:
        try:
            bid_val = float(borrower_input)
            row = df[df['borrower_id'] == bid_val]

            if row.empty:
                st.error(f"Borrower ID {borrower_input} not found in test set.")
            else:
                row = row.iloc[0]
                risk = row['risk_score']
                actual = row['actual_default']

                # Risk badge
                risk_color = '#d32f2f' if risk >= 0.5 else '#f57c00' if risk >= 0.3 else '#388e3c'
                st.markdown(
                    f"<h2 style='color:{risk_color}'>Predicted Default Risk: {risk:.1%}</h2>",
                    unsafe_allow_html=True,
                )

                outcome_str = "✅ Actually defaulted" if actual == 1 else "✅ Did not default"
                st.caption(f"Ground truth (test set): {outcome_str}")

                st.markdown("---")

                # SHAP-based top reasons
                if shap_df is not None:
                    shap_row_match = shap_df[shap_df['borrower_id'] == bid_val]
                    if not shap_row_match.empty:
                        shap_row = shap_row_match.iloc[0].drop('borrower_id')
                        top_risk_feats = shap_row.nlargest(5)

                        st.subheader("🔴 Top Risk Factors")
                        for feat, shap_val in top_risk_feats.items():
                            if shap_val <= 0:
                                continue
                            val = row.get(feat, None)
                            med = df[feat].median() if feat in df.columns else None
                            fname = READABLE.get(feat, feat.replace('_', ' ').title())

                            direction = 'above' if (val and med and val > med) else 'below'

                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(f"**{fname}**: `{val:.2f}` (portfolio median: `{med:.2f}`)")
                                st.progress(min(abs(shap_val) * 5, 1.0))
                            with col_b:
                                st.markdown(f"Impact: `+{shap_val:.3f}`")
                            st.markdown("")

                        top_protective = shap_row.nsmallest(3)
                        protective_feats = top_protective[top_protective < 0]
                        if len(protective_feats) > 0:
                            st.subheader("🟢 Protective Factors")
                            for feat, shap_val in protective_feats.items():
                                val = row.get(feat, None)
                                fname = READABLE.get(feat, feat.replace('_', ' ').title())
                                st.markdown(f"- **{fname}**: `{val:.2f}` (reduces risk by `{abs(shap_val):.3f}`)")

                st.markdown("---")
                st.subheader("Full Borrower Profile")
                display_feats = ['loan_amnt', 'term', 'int_rate', 'dti', 'fico_range_low',
                                 'grade_num', 'emp_length', 'revol_util', 'loan_to_income',
                                 'credit_age_months', 'inq_last_6mths', 'pub_rec', 'ever_120dpd']
                profile = {READABLE.get(f, f): round(row.get(f, np.nan), 2)
                           for f in display_feats if f in row.index}
                st.table(pd.DataFrame.from_dict(profile, orient='index', columns=['Value']))

        except ValueError:
            st.error("Please enter a valid numeric Borrower ID.")

# ─────────────────────────────────────────────────────────────────────────────
# Query 3 — Portfolio Statistics
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "📊 Portfolio Statistics":
    st.header("Portfolio Statistics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Borrowers (test)", f"{len(df):,}")
    col2.metric("Actual Default Rate", f"{df['actual_default'].mean():.1%}")
    col3.metric("Avg Predicted Risk", f"{df['risk_score'].mean():.1%}")
    col4.metric("High-Risk Borrowers (>50%)", f"{(df['risk_score'] > 0.5).sum():,}")

    st.markdown("---")

    st.subheader("Default Rate by Loan Amount")
    loan_threshold = st.slider("Loan amount threshold ($)", 5000, 35000, 20000, step=5000)
    above = df[df['loan_amnt'] > loan_threshold]
    below = df[df['loan_amnt'] <= loan_threshold]

    c1, c2 = st.columns(2)
    c1.metric(
        f"Avg default rate — loans > ${loan_threshold:,}",
        f"{above['actual_default'].mean():.1%}",
        delta=f"{(above['actual_default'].mean() - df['actual_default'].mean()):.1%} vs overall",
        delta_color="inverse",
    )
    c2.metric(
        f"Avg default rate — loans ≤ ${loan_threshold:,}",
        f"{below['actual_default'].mean():.1%}",
        delta=f"{(below['actual_default'].mean() - df['actual_default'].mean()):.1%} vs overall",
        delta_color="inverse",
    )

    st.markdown("---")
    st.subheader("Default Rate by Loan Grade")

    grade_map_inv = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
    df['grade_letter'] = df['grade_num'].apply(lambda x: grade_map_inv.get(int(x), '?'))
    grade_stats = (
        df.groupby('grade_letter')['actual_default']
        .agg(['mean', 'count'])
        .reset_index()
        .rename(columns={'mean': 'default_rate', 'count': 'n_loans'})
        .sort_values('grade_letter')
    )

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3))
    colors = ['#4caf50','#8bc34a','#ffc107','#ff9800','#f44336','#b71c1c','#7b0000']
    ax.bar(grade_stats['grade_letter'], grade_stats['default_rate'] * 100,
           color=colors[:len(grade_stats)])
    ax.set_xlabel('Loan Grade (A = best, G = worst)')
    ax.set_ylabel('Default Rate (%)')
    ax.set_title('Default Rate by LC Loan Grade')
    st.pyplot(fig, use_container_width=True)

    st.dataframe(
        grade_stats.rename(columns={
            'grade_letter': 'Grade',
            'default_rate': 'Default Rate',
            'n_loans': '# Loans'
        }).style.format({'Default Rate': '{:.1%}'}),
        use_container_width=True,
        hide_index=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Query 4 — Natural Language Query (keyword matching)
# ─────────────────────────────────────────────────────────────────────────────
elif query_type == "💬 Natural Language Query":
    st.header("Natural Language Query")
    st.markdown(
        "Type a question in plain English. The system uses keyword matching to route "
        "your query — no LLM required."
    )
    st.info(
        "**Example queries:**\n"
        "- Show me the top 10 riskiest borrowers\n"
        "- What is the default rate for loans above $20,000?\n"
        "- Explain borrower 1060773\n"
        "- How many borrowers have a risk score above 70%?\n"
        "- What is the average DTI of defaulters?"
    )

    user_query = st.text_input("Your query:", placeholder="Type your question here...")

    if user_query:
        q = user_query.lower()

        # ── Route: top / highest risk ──
        if any(kw in q for kw in ['top', 'highest', 'riskiest', 'most risky', 'most at risk']):
            n_match = re.search(r'top (\d+)', q)
            n = int(n_match.group(1)) if n_match else 10

            st.subheader(f"Top {n} Highest-Risk Borrowers")
            top_n = (
                df[['borrower_id', 'risk_score', 'actual_default', 'loan_amnt',
                     'int_rate', 'dti', 'fico_range_low']]
                .sort_values('risk_score', ascending=False)
                .head(n)
            )
            top_n['risk_score'] = (top_n['risk_score'] * 100).round(1).astype(str) + '%'
            st.dataframe(top_n, hide_index=True, use_container_width=True)

        # ── Route: count / how many ── (MOVED ABOVE explain to avoid 'borrower' keyword clash)
        elif any(kw in q for kw in ['how many', 'count', 'number of']):
            pct_match = re.search(r'(\d+)%', q)
            if pct_match and any(kw in q for kw in ['risk', 'score', 'probability']):
                pct = int(pct_match.group(1)) / 100
                direction = 'above' if any(kw in q for kw in ['above', 'over', 'greater']) else 'below'
                count = (df['risk_score'] > pct).sum() if direction == 'above' else (df['risk_score'] < pct).sum()
                st.metric(f"Borrowers with risk {direction} {pct:.0%}", f"{count:,}")
            else:
                st.metric("Total borrowers in test set", f"{len(df):,}")
                st.metric("Actual defaulters", f"{df['actual_default'].sum():,}")

        # ── Route: explain borrower ──
        elif any(kw in q for kw in ['explain', 'why', 'reason', 'borrower']):
            id_match = re.search(r'\b(\d{4,})\b', q)
            if id_match:
                bid = float(id_match.group(1))
                row = df[df['borrower_id'] == bid]
                if not row.empty:
                    row = row.iloc[0]
                    st.subheader(f"Explanation for Borrower {int(bid)}")
                    st.metric("Predicted Default Risk", f"{row['risk_score']:.1%}")

                    if shap_df is not None:
                        shap_match = shap_df[shap_df['borrower_id'] == bid]
                        if not shap_match.empty:
                            shap_row = shap_match.iloc[0].drop('borrower_id')
                            top_feats = shap_row.nlargest(4)
                            st.markdown("**Top risk factors (by SHAP impact):**")
                            for feat, val in top_feats.items():
                                if val > 0:
                                    fname = READABLE.get(feat, feat)
                                    fval = row.get(feat, 'N/A')
                                    st.markdown(f"- {fname}: `{fval:.2f}` → impact `+{val:.3f}`")
                else:
                    st.warning(f"Borrower ID {int(bid)} not found in test set.")
            else:
                st.warning("Please include a numeric Borrower ID in your query (e.g. 'explain borrower 1060773').")

        # ── Route: default rate / statistics ──
        elif any(kw in q for kw in ['default rate', 'rate', 'average', 'avg', 'mean', 'statistics', 'stats']):
            amount_match = re.search(r'\$?([\d,]+)', q)

            if amount_match and any(kw in q for kw in ['above', 'over', 'greater', 'more than']):
                threshold = float(amount_match.group(1).replace(',', ''))
                subset = df[df['loan_amnt'] > threshold]
                rate = subset['actual_default'].mean()
                st.metric(
                    f"Default rate for loans above ${threshold:,.0f}",
                    f"{rate:.1%}",
                    delta=f"{(rate - df['actual_default'].mean()):.1%} vs overall {df['actual_default'].mean():.1%}",
                    delta_color="inverse"
                )
                st.caption(f"Based on {len(subset):,} loans in the test set.")

            elif 'dti' in q and any(kw in q for kw in ['default', 'defaulter', 'defaulted']):
                defaulters = df[df['actual_default'] == 1]['dti']
                non_def = df[df['actual_default'] == 0]['dti']
                c1, c2 = st.columns(2)
                c1.metric("Avg DTI — Defaulters", f"{defaulters.mean():.1f}%")
                c2.metric("Avg DTI — Non-Defaulters", f"{non_def.mean():.1f}%")

            elif 'fico' in q:
                c1, c2 = st.columns(2)
                c1.metric("Avg FICO — Defaulters", f"{df[df['actual_default']==1]['fico_range_low'].mean():.0f}")
                c2.metric("Avg FICO — Non-Defaulters", f"{df[df['actual_default']==0]['fico_range_low'].mean():.0f}")

            else:
                st.metric("Overall Default Rate (test set)", f"{df['actual_default'].mean():.1%}")
                st.metric("Average Predicted Risk", f"{df['risk_score'].mean():.1%}")

        else:
            st.warning(
                "Query not recognised. Try including keywords like: "
                "'top 10', 'default rate', 'explain borrower [ID]', or 'how many'."
            )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Periscope Labs Take-Home Assignment | Model: GradientBoostingClassifier | "
    "Dataset: Lending Club 2007–2018 (50k subsample) | Seed: 42"
)
