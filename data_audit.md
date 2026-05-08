# Data Audit — Lending Club Loan Default Predictor

## Selected Features (~20 source columns used in modeling, expands to 29 features after one-hot encoding)

| Column                   | Description                                                                                                | Available at Application? |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- | ------------------------- |
| `loan_amnt`              | Requested loan amount                                                                                      | ✅ Yes                    |
| `term`                   | Repayment period (36 or 60 months)                                                                         | ✅ Yes                    |
| `int_rate`               | Interest rate assigned by LC after risk assessment                                                         | ✅ Yes                    |
| `grade` / `sub_grade`    | LC's own risk grade (A–G)                                                                                  | ✅ Yes                    |
| `emp_length`             | Borrower's years of employment (parsed from string e.g. "10+ years" → 10)                                  | ✅ Yes                    |
| `home_ownership`         | RENT / OWN / MORTGAGE (one-hot encoded)                                                                    | ✅ Yes                    |
| `annual_inc`             | Self-reported annual income (log-transformed to handle outliers)                                           | ✅ Yes                    |
| `verification_status`    | Whether income was verified                                                                                | ✅ Yes                    |
| `purpose`                | Stated loan purpose — top 8 categories kept, rest → "other" (one-hot encoded)                              | ✅ Yes                    |
| `dti`                    | Debt-to-income ratio at application                                                                        | ✅ Yes                    |
| `installment`            | Monthly payment amount                                                                                     | ✅ Yes                    |
| `delinq_2yrs`            | # of delinquencies in past 2 years                                                                         | ✅ Yes                    |
| `fico_range_low`         | Lower FICO score band at origination                                                                       | ✅ Yes                    |
| `inq_last_6mths`         | Hard credit inquiries in last 6 months                                                                     | ✅ Yes                    |
| `open_acc`               | Number of open credit lines                                                                                | ✅ Yes                    |
| `pub_rec`                | Number of public derogatory records                                                                        | ✅ Yes                    |
| `revol_bal`              | Current revolving balance (log-transformed)                                                                | ✅ Yes                    |
| `revol_util`             | Revolving credit utilization %                                                                             | ✅ Yes                    |
| `total_acc`              | Total number of credit accounts ever                                                                       | ✅ Yes                    |
| `earliest_cr_line`       | Date of oldest credit account (→ used to derive `credit_age_months`)                                       | ✅ Yes                    |
| `mths_since_last_delinq` | Months since last delinquency — missing when borrower has never been delinquent; imputed with sentinel 999 | ✅ Yes (if applicable)    |

## Leakage Columns — DO NOT USE as Features

These columns only exist **after** the loan outcome is known:

| Column                     | Why It Leaks                                                            |
| -------------------------- | ----------------------------------------------------------------------- |
| `total_pymnt`              | Total payments received — only known after repayment history            |
| `recoveries`               | Post-default recovery amount — only exists after charge-off             |
| `collection_recovery_fee`  | Fee from debt collector — only after default and collection             |
| `last_pymnt_d`             | Date of last payment — depends on how long the borrower paid            |
| `last_pymnt_amnt`          | Amount of last payment — a direct signal of distress/default            |
| `out_prncp`                | Remaining principal — decreases with payments, reveals repayment status |
| `total_rec_prncp`          | Total principal received — accumulates only during repayment            |
| `total_rec_int`            | Total interest received — same issue as above                           |
| `total_rec_late_fee`       | Late fees charged — only exist if borrower was actually late            |
| `last_fico_range_high/low` | Post-origination credit score — updated periodically after loan begins  |
| `next_pymnt_d`             | Scheduled next payment date — N/A if loan is in default                 |

**The test**: "Would a loan officer have this number on the day the application is submitted?" If no → leakage.

---

## Class Imbalance

After dropping `Current` loans (no outcome yet) and defining:

> **Default** = `loan_status` ∈ `['Charged Off', 'Default', 'Late (31-120 days)']`

Actual distribution in the 50,000-row subsample (seed=42):

- **Non-default (Fully Paid)**: 39,557 rows — **79.1%**
- **Default (Charged Off / Default / Late)**: 10,443 rows — **20.9%**
- **Default rate: 0.209**

The class ratio is preserved exactly in both train (0.209) and test (0.209) splits via stratification.

This is a **moderately imbalanced** dataset. The 20/80 split is mild enough to handle without resampling. Our approach:

- **Stratified 80/20 train/test split** to preserve the class ratio
- **AUC-ROC + Precision@Top10%** as primary metrics (not accuracy, which is misleading at 80/20 imbalance — a model predicting "never default" scores 79.1% accuracy and is useless)
- No oversampling (SMOTE) or undersampling applied — the imbalance does not warrant it at this ratio

---

## Notes on Data Quality

- `emp_length` is stored as a string (`"10+ years"`, `"< 1 year"`) — parsed to integer (0–10), missing values imputed with median
- `int_rate` is occasionally stored as a string with `%` suffix — stripped and cast to float
- `earliest_cr_line` is a date string (`"Jan-2000"`) — converted to `credit_age_months` by differencing against `issue_d`
- `mths_since_last_delinq` is missing for ~50% of borrowers — this is legitimate (it means "never delinquent"), not random missingness. Imputed with sentinel value 999
- `annual_inc` has extreme outliers (some borrowers report $10M+) — log-transformed (`log1p`) to compress the scale
- `revol_bal` similarly right-skewed — log-transformed
- `total_rev_hi_lim` (used to compute `revol_bal_to_limit`) has some zero values — handled by adding 1 to denominator before dividing
