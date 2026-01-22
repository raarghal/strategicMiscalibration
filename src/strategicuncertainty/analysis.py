from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load a specific CSV file
repo_path = Path(__file__).parent.parent.parent
datafile = repo_path / "outputs/sweep_two_player_20260122_085619.csv"
output_dir = datafile.parent
figure_dir = repo_path / "figures"


df = pd.read_csv(datafile)

print(f"Loaded: {datafile.name}")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print("\nFirst few rows:")
df.head()

sns.scatterplot(
    data=df, x="discount_factor", y="confidence_diff", hue="round", s=100, alpha=0.6
)
plt.xlabel("Discount Factor", fontsize=12)
plt.ylabel("Confidence Difference", fontsize=12)
plt.title("Confidence Difference vs Discount Factor by Round", fontsize=14)
plt.tight_layout()
plt.show()
