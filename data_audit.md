# Data Audit — Lending Club Loan Default Predictor

## Selected Features (~15 columns used in modeling)

| Column | Description | Available at Application? |
|---|---|---|
| `loan_amnt` | Requested loan amount | ✅ Yes |
| `term` | Repayment period (36 or 60 months) | ✅ Yes |
| `int_rate` | Interest rate assigned by LC after risk assessment | ✅ Yes |
| `grade` / `sub_grade` | LC's own risk grade (A–G) | ✅ Yes |
| `emp_length` | Borrower's years of employment | ✅ Yes |
| `home_ownership` | RENT / OWN / MORTGAGE | ✅ Yes |
| `annual_inc` | Self-reported annual income | ✅ Yes |
| `verification_status` | Whether income was verified | ✅ Yes |
| `purpose` | Stated loan purpose (debt consolidation, etc.) | ✅ Yes |
| `dti` | Debt-to-income ratio at application | ✅ Yes |
| `delinq_2yrs` | # of delinquencies in past 2 years | ✅ Yes |
| `fico_range_low` | Lower FICO score band | ✅ Yes |
| `inq_last_6mths` | Hard credit inquiries in last 6 months | ✅ Yes |
| `open_acc` | Number of open credit lines | ✅ Yes |
| `pub_rec` | Number of public derogatory records | ✅ Yes |
| `revol_bal` | Current revolving balance | ✅ Yes |
| `revol_util` | Revolving credit utilization % | ✅ Yes |
| `total_acc` | Total number of credit accounts ever | ✅ Yes |
| `earliest_cr_line` | Date of oldest credit account (→ credit age) | ✅ Yes |
| `mths_since_last_delinq` | Months since last delinquency | ✅ Yes (if applicable) |

## Leakage Columns — DO NOT USE as Features

These columns only exist **after** the loan outcome is known:

| Column | Why It Leaks |
|---|---|
| `total_pymnt` | Total payments received — only known after repayment history |
| `recoveries` | Post-default recovery amount — only exists after charge-off |
| `collection_recovery_fee` | Fee from debt collector — only after default and collection |
| `last_pymnt_d` | Date of last payment — depends on how long the borrower paid |
| `last_pymnt_amnt` | Amount of last payment — a direct signal of distress/default |
| `out_prncp` | Remaining principal — decreases with payments, reveals repayment status |
| `total_rec_prncp` | Total principal received — accumulates only during repayment |
| `total_rec_int` | Total interest received — same issue as above |
| `total_rec_late_fee` | Late fees charged — only exist if borrower was actually late |
| `last_fico_range_high/low` | Post-origination credit score — updated periodically after loan begins |
| `next_pymnt_d` | Scheduled next payment date — N/A if loan is in default |

**The test**: "Would a loan officer have this number on the day the application is submitted?" If no → leakage.

---

## Class Imbalance

After dropping `Current` loans (no outcome yet) and defining:

> **Default** = `loan_status` ∈ `['Charged Off', 'Default', 'Late (31-120 days)']`

Expected distribution in the dataset (calculated during EDA in notebook):

- **Non-default (Fully Paid)**: ~80%
- **Default (Charged Off / Default / Late)**: ~20%

This is a **moderately imbalanced** dataset. The minority class (defaulters) is ~20%, which is manageable without aggressive resampling. However, we use:
- **Stratified train/test split** to preserve the ratio
- **`class_weight='balanced'`** in models where applicable
- **AUC-ROC + Precision@Top10%** as primary metrics (not accuracy, which is misleading here)

---

## Notes on Data Quality

- `emp_length` is stored as a string (`"10+ years"`, `"< 1 year"`) — needs parsing
- `earliest_cr_line` is a date string — converted to `credit_age_months` feature
- ~15-20% of behavioral columns (`mths_since_last_delinq`, `mths_since_last_record`, etc.) are missing — these are legitimately missing when the borrower has *never* been delinquent. Impute with a large sentinel (e.g., 999) or indicator flag
- `annual_inc` has outliers — a small number of borrowers report $10M+ income; cap or log-transform
