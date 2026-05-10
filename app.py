

import os
import pandas as pd
import streamlit as st

# -------------------------- CONFIG --------------------------
st.set_page_config(page_title="Loan Default Risk", page_icon="🏦", layout="wide")

DATA_PATH = "data/test_predictions.csv"
SHAP_PATH = "data/shap_values.csv"

# -------------------------- LOAD DATA --------------------------
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ data/test_predictions.csv not found.\nPlease run the notebook first!")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    shap_df = pd.read_csv(SHAP_PATH) if os.path.exists(SHAP_PATH) else None
    return df, shap_df

df, shap_df = load_data()

# -------------------------- SIDEBAR --------------------------
st.sidebar.title("Query Interface")
query = st.sidebar.radio("Choose a query:", 
    ["1. Show top 10 highest-risk borrowers", 
     "2. Why was borrower [ID] flagged?", 
     "3. Average default rate for loans above $20,000"])

st.title("🏦 Loan Default Risk Dashboard")
st.caption("Periscope Labs — Mini Credit Risk Predictor")

# -------------------------- QUERY 1 --------------------------
if query == "1. Show top 10 highest-risk borrowers":
    st.header("Top 10 Highest-Risk Borrowers")
    st.markdown("These borrowers have the highest predicted default risk.")
    
    top10 = (df[['borrower_id', 'risk_score', 'actual_default', 
                 'loan_amnt', 'int_rate', 'dti']]
             .sort_values('risk_score', ascending=False)
             .head(10)
             .copy())
    
    top10['Risk %'] = (top10['risk_score'] * 100).round(1).astype(str) + '%'
    top10['Actually Defaulted'] = top10['actual_default'].map({1: 'Yes', 0: 'No'})
    
    st.dataframe(top10[['borrower_id', 'Risk %', 'Actually Defaulted', 
                        'loan_amnt', 'int_rate', 'dti']], 
                 use_container_width=True, hide_index=True)

# -------------------------- QUERY 2 --------------------------
elif query == "2. Why was borrower [ID] flagged?":
    st.header("Why was borrower [ID] flagged?")
    borrower_input = st.text_input("Enter Borrower ID", placeholder="1060773")
    
    if borrower_input:
        try:
            bid = float(borrower_input.strip())
            row = df[df['borrower_id'] == bid]
            if row.empty:
                st.error(f"Borrower ID {int(bid)} not found in test set.")
            else:
                row = row.iloc[0]
                st.subheader(f"Borrower {int(bid)}")
                st.metric("Predicted Default Risk", f"{row['risk_score']:.1%}")
                
                # Show simple Part C style explanation using SHAP
                if shap_df is not None:
                    shap_match = shap_df[shap_df['borrower_id'] == bid]
                    if not shap_match.empty:
                        shap_row = shap_match.iloc[0].drop('borrower_id')
                        top_risk = shap_row.nlargest(4)
                        st.markdown("**Top risk factors:**")
                        for feat, shap_val in top_risk.items():
                            if shap_val > 0:
                                val = row.get(feat, 'N/A')
                                st.markdown(f"- **{feat}** = `{val}` → impact `+{shap_val:.3f}`")
                else:
                    st.info("SHAP values not available.")
        except:
            st.error("Please enter a valid numeric Borrower ID.")

# -------------------------- QUERY 3 --------------------------
else:
    st.header("Portfolio Statistic")
    st.subheader("Average default rate for loans above $20,000")
    
    big_loans = df[df['loan_amnt'] > 20000]
    rate = big_loans['actual_default'].mean()
    overall = df['actual_default'].mean()
    
    st.metric("Default Rate (loans > $20k)", f"{rate:.1%}", 
              delta=f"{rate - overall:.1%} vs overall")
    st.caption(f"Based on {len(big_loans):,} loans in the test set.")

# -------------------------- FOOTER --------------------------
st.markdown("---")
