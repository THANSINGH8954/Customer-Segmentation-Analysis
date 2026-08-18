"""
generate_data.py -- Customer Segmentation Analysis
----------------------------------------------------
Generates a synthetic-but-realistic customer dataset combining demographics
and purchasing-behavior (RFM-style) metrics, with 5 latent "true" customer
archetypes baked in (so clustering has real, recoverable structure) plus
common real-world data quality issues (missing values, duplicates,
inconsistent text, a few outliers) for the cleaning step.

Run:
    python scripts/generate_data.py
Output:
    data/customer_data_raw.csv
"""

import os
import numpy as np
import pandas as pd

SEED = 21
rng = np.random.default_rng(SEED)

N_PER_SEGMENT = {
    "Budget Shoppers": 340,
    "High-Value Loyalists": 220,
    "Occasional Big Spenders": 180,
    "New / Low-Engagement": 260,
    "Young Digital Shoppers": 200,
}

GENDERS = ["Female", "Male", "Non-Binary"]
CITIES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Austin",
          "Seattle", "Denver", "Miami", "Boston", "Atlanta", "Portland"]
CHANNELS = ["Online", "In-Store", "Mobile App"]
MEMBERSHIP = ["Basic", "Silver", "Gold", "Platinum"]

# Segment archetypes: (age_mean, age_sd, income_mean, income_sd,
#                       recency_days_mean, recency_sd,
#                       frequency_mean, frequency_sd,
#                       monetary_mean, monetary_sd,
#                       avg_basket_mean, avg_basket_sd,
#                       tenure_months_mean, tenure_sd)
ARCHETYPES = {
    "Budget Shoppers": dict(age=(42, 12), income=(38000, 9000), recency=(25, 12),
                             frequency=(14, 5), monetary=(650, 220), basket=(28, 9), tenure=(30, 14)),
    "High-Value Loyalists": dict(age=(46, 10), income=(95000, 18000), recency=(10, 6),
                                  frequency=(28, 7), monetary=(4200, 1100), basket=(140, 35), tenure=(48, 16)),
    "Occasional Big Spenders": dict(age=(50, 11), income=(110000, 22000), recency=(60, 25),
                                     frequency=(5, 2), monetary=(2100, 700), basket=(380, 90), tenure=(24, 12)),
    "New / Low-Engagement": dict(age=(31, 9), income=(45000, 12000), recency=(80, 30),
                                  frequency=(2, 1), monetary=(120, 60), basket=(55, 20), tenure=(4, 3)),
    "Young Digital Shoppers": dict(age=(25, 4), income=(42000, 8000), recency=(15, 8),
                                    frequency=(18, 6), monetary=(900, 300), basket=(35, 12), tenure=(14, 8)),
}

rows = []
cust_id = 10000
for segment, n in N_PER_SEGMENT.items():
    a = ARCHETYPES[segment]
    for _ in range(n):
        cust_id += 1
        age = int(np.clip(rng.normal(*a["age"]), 18, 80))
        income = round(max(15000, rng.normal(*a["income"])), 2)
        recency = int(np.clip(rng.normal(*a["recency"]), 0, 365))
        frequency = int(max(0, round(rng.normal(*a["frequency"]))))
        monetary = round(max(0, rng.normal(*a["monetary"])), 2)
        avg_basket = round(max(5, rng.normal(*a["basket"])), 2)
        tenure = int(np.clip(rng.normal(*a["tenure"]), 1, 96))

        gender = rng.choice(GENDERS, p=[0.48, 0.48, 0.04])
        city = rng.choice(CITIES)
        channel = rng.choice(CHANNELS, p=[0.5, 0.3, 0.2])
        membership = rng.choice(MEMBERSHIP, p=[0.4, 0.3, 0.2, 0.1])

        rows.append({
            "CustomerID": f"CUST-{cust_id}",
            "Age": age,
            "Gender": gender,
            "City": city,
            "AnnualIncome": income,
            "MembershipTier": membership,
            "PreferredChannel": channel,
            "TenureMonths": tenure,
            "RecencyDays": recency,
            "Frequency": frequency,
            "MonetaryTotal": monetary,
            "AvgBasketValue": avg_basket,
            "_TrueSegment": segment,  # ground truth, kept only for validation notes
        })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Inject realistic messiness
# ---------------------------------------------------------------------------
dupe_idx = df.sample(frac=0.015, random_state=SEED).index
df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

for col, frac in [("AnnualIncome", 0.02), ("City", 0.015), ("Gender", 0.01), ("MembershipTier", 0.012)]:
    idx = df.sample(frac=frac, random_state=SEED + 1).index
    df.loc[idx, col] = np.nan

idx_upper = df.sample(frac=0.05, random_state=SEED + 2).index
df.loc[idx_upper, "PreferredChannel"] = df.loc[idx_upper, "PreferredChannel"].str.upper()
idx_ws = df.sample(frac=0.05, random_state=SEED + 3).index
df.loc[idx_ws, "MembershipTier"] = df.loc[idx_ws, "MembershipTier"].apply(
    lambda x: f"  {x.lower()}  " if isinstance(x, str) else x
)

# a few implausible outliers (data entry errors) to be caught in cleaning
idx_out = df.sample(n=6, random_state=SEED + 4).index
df.loc[idx_out, "Age"] = rng.choice([2, 5, 130, 145], size=len(idx_out))

idx_neg = df.sample(n=4, random_state=SEED + 5).index
df.loc[idx_neg, "AnnualIncome"] = -df.loc[idx_neg, "AnnualIncome"]

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
# Ground-truth segment label is saved separately (not exposed in the main
# raw file) so the notebook's clustering results can be validated against it
# without "cheating" by using it as a feature.
ground_truth = df[["CustomerID", "_TrueSegment"]].drop_duplicates(subset="CustomerID")
ground_truth.to_csv("data/ground_truth_segments_FOR_VALIDATION_ONLY.csv", index=False)

df = df.drop(columns=["_TrueSegment"])
df.to_csv("data/customer_data_raw.csv", index=False)
print(f"Generated {len(df):,} customer rows -> data/customer_data_raw.csv")
print(df.head())
