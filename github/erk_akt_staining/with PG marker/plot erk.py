import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load your CSV files (Replace with your actual file names)
df1 = pd.read_csv(r"phm toptosos\optosos_light perk\optosos_light perk-nuclei_normalized_summary.csv")
df2 = pd.read_csv(r"phm toptosos\optosos_dark perk\optosos_dark perk-nuclei_normalized_summary.csv")

# 2. Extract your intensity column (Replace 'Mean' with your actual column name)
# We add a label column so the computer knows which group is which
data1 = pd.DataFrame(
    {"Intensity": df1["Normalized_Ratio_GreenROI_Over_DapiNuclei"], "Group": "phm optosos light perk"}
)  # Change 'Control' to your group name
data2 = pd.DataFrame(
    {"Intensity": df2["Normalized_Ratio_GreenROI_Over_DapiNuclei"], "Group": "phm optosos dark perk"}
)  # Change 'Treated' to your group name

# 3. Combine the datasets into one table
combined_df = pd.concat([data1, data2], ignore_index=True)

# 4. Set up the plot style
sns.set_theme(style="ticks")
plt.figure(figsize=(5, 6))

# 5. Draw the bar chart (Shows the average/mean value)
sns.barplot(
    data=combined_df,
    x="Group",
    y="Intensity",
    errorbar="se",  # Adds Standard Error bars
    capsize=0.1,  # Puts horizontal caps on error bars
    color="lightgray",  # Neutral background color for bars
    edgecolor="black",
    linewidth=1.5,
)

# 6. Overlay the individual scatter points
sns.stripplot(
    data=combined_df,
    x="Group",
    y="Intensity",
    color="black",  # Color of the points
    size=6,  # Size of the points
    jitter=0.2,  # Spreads points out horizontally so they don't overlap
    alpha=0.7,  # Makes points slightly translucent
)

# 7. Clean up labels and aesthetics
plt.ylabel("relative erk intensity", fontsize=12, fontweight="bold")
sns.despine()  # Removes top and right borders for a clean look

# 8. Save and display
plt.tight_layout()
plt.savefig("optosos perk.svg", dpi=300)
plt.show()