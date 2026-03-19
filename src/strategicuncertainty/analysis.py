import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load a specific CSV file
repo_path = Path(__file__).parent.parent.parent
output_dir = repo_path / "outputs/experiments/sweep_two_player_20260122_133334_figure"
csv_candidates = sorted(
    output_dir.rglob("results*.csv"),
    key=lambda candidate: candidate.stat().st_mtime,
    reverse=True,
)
if csv_candidates:
    datafile = csv_candidates[0]
else:
    raise FileNotFoundError(f"No results CSV found under {output_dir}")

df = pd.read_csv(datafile)
# df["round"] = df["round"] + 1
df["Delegated"] = df["agent_payoff"] * 10
df = df.round(4)

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

timestamp = time.strftime("%Y%m%d_%H%M%S")

df_correct = df[(df["agent_confidence"] > 0) & (df["baseline_confidence"] > 0)]

df_correct_filtered = df_filtered[
    (df_filtered["agent_confidence"] > 0) & (df_filtered["baseline_confidence"] > 0)
].copy()

plotting_data = df_correct

sns.barplot(
    data=plotting_data[plotting_data["round"] == 0],
    x="discount_factor",
    y="confidence_diff",
    # hue="round",
    capsize=0.1,
    errorbar=("sd", 0.25),
)
plt.xlabel("Discount Factor", fontsize=12)
plt.ylabel("Confidence Difference", fontsize=12)
# plt.title("Confidence Difference vs Discount Factor by Round", fontsize=14)
plt.tight_layout()

plt.savefig(
    output_dir / f"Cconfidence_diff_vs_discount_factor_by_round_{timestamp}.pdf",
    format="pdf",
    dpi=1200,
)
# plt.show()

dfrnd1 = plotting_data[plotting_data["round"] == 0]
dfrnd2 = plotting_data[plotting_data["round"] == 1]

heatmap_data_rnd1 = dfrnd1.copy()
heatmap_data_rnd2 = dfrnd2.copy()

rnd2_bins_honesty = []
rnd2_bins_ability = []
for i in range(len(heatmap_data_rnd2)):
    if heatmap_data_rnd2["prior_agent_ability"].to_list()[i] <= 0.2:
        rnd2_bins_ability.append(0.1)
    elif heatmap_data_rnd2["prior_agent_ability"].to_list()[i] <= 0.4:
        rnd2_bins_ability.append(0.3)
    elif heatmap_data_rnd2["prior_agent_ability"].to_list()[i] <= 0.6:
        rnd2_bins_ability.append(0.5)
    elif heatmap_data_rnd2["prior_agent_ability"].to_list()[i] <= 0.8:
        rnd2_bins_ability.append(0.7)
    else:
        rnd2_bins_ability.append(0.9)

    if heatmap_data_rnd2["prior_agent_honesty"].to_list()[i] <= 0.2:
        rnd2_bins_honesty.append(0.1)
    elif heatmap_data_rnd2["prior_agent_honesty"].to_list()[i] <= 0.4:
        rnd2_bins_honesty.append(0.3)
    elif heatmap_data_rnd2["prior_agent_honesty"].to_list()[i] <= 0.6:
        rnd2_bins_honesty.append(0.5)
    elif heatmap_data_rnd2["prior_agent_honesty"].to_list()[i] <= 0.8:
        rnd2_bins_honesty.append(0.7)
    else:
        rnd2_bins_honesty.append(0.9)

heatmap_data_rnd2["binned_ability"] = rnd2_bins_ability
heatmap_data_rnd2["binned_honesty"] = rnd2_bins_honesty

heatmap_df_rnd1 = heatmap_data_rnd1.pivot_table(
    index="prior_agent_ability",
    columns="prior_agent_honesty",
    values="Delegated",
    aggfunc="mean",
    observed=False,
)

heatmap_df_rnd2 = heatmap_data_rnd2.pivot_table(
    index="binned_ability",
    columns="binned_honesty",
    values="Delegated",
    aggfunc="mean",
    observed=False,
)


fig, ax = plt.subplots(1, 2, figsize=(13, 6))
sns.heatmap(heatmap_df_rnd1, cmap="viridis", ax=ax[0], annot=True, fmt=".2f")
ax[0].invert_yaxis()
ax[0].set_xlabel("Belief in Agent Honesty", fontsize=12)
ax[0].set_ylabel("Belief in Agent Ability", fontsize=12)
ax[0].set_title("Round 1", fontsize=14)

sns.heatmap(heatmap_df_rnd2, cmap="viridis", ax=ax[1], annot=True, fmt=".2f")
ax[1].invert_yaxis()
ax[1].set_xlabel("Belief in Agent Honesty", fontsize=12)
ax[1].set_ylabel("Belief in Agent Ability", fontsize=12)
ax[1].set_title("Round 2", fontsize=14)

fig.suptitle("Delegation Rate v. Reputation", fontsize=16)
plt.tight_layout()

# plt.savefig(
#     output_dir / f"Cdelegation_rate_vs_priors_by_round_{timestamp}.pdf",
#     format="pdf",
#     dpi=1200,
# )
plt.show()
