from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load a specific CSV file
repo_path = Path(__file__).parent.parent.parent
datafile = repo_path / "outputs/sweep_two_player_20260122_133334/results.csv"
output_dir = datafile.parent
figure_dir = repo_path / "figures"


df = pd.read_csv(datafile)

print(f"Loaded: {datafile.name}")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
# print("\nFirst few rows:")
# df.head()
print(len(df))

# Filter outliers from confidence_diff using IQR
Q1 = df["confidence_diff"].quantile(0.25)
Q3 = df["confidence_diff"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_filtered = df[
    (df["confidence_diff"] >= lower_bound) & (df["confidence_diff"] <= upper_bound)
].copy()

df_outliers = df[
    (df["confidence_diff"] < lower_bound) | (df["confidence_diff"] > upper_bound)
].copy()

print(df_outliers["agent_confidence"])

df_correct = df[(df["agent_confidence"] > 0) & (df["baseline_confidence"] > 0)].copy()

sns.barplot(
    data=df_correct,
    x="discount_factor",
    y="confidence_diff",
    hue="round",
    capsize=0.1,
    errorbar="se",
)
plt.xlabel("Discount Factor", fontsize=12)
plt.ylabel("Confidence Difference", fontsize=12)
plt.title("Confidence Difference vs Discount Factor by Round", fontsize=14)
plt.tight_layout()
plt.savefig(
    figure_dir / "confidence_diff_vs_discount_factor_by_round.pdf",
    format="pdf",
    dpi=1200,
)

sns.scatterplot(
    data=df,
    x="user_belief_agent_correct",
    y="agent_payoff",
    hue="round",
    s=100,
    alpha=0.6,
)
plt.show()
