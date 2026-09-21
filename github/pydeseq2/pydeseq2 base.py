import pandas as pd
from pydeseq2.dds import DeseqDataSet
import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

SAVE=True
inference = DefaultInference(n_cpus=8)

counts_df = pd.read_excel(r"read_count\gene_expression_read_count_only.xlsx", index_col=0).astype(int).T
metadata = pd.read_excel(r"read_count\metadata.xlsx").set_index("sample")

samples_to_keep = ~metadata.condition.isna()
counts_df = counts_df.loc[samples_to_keep]
metadata = metadata.loc[samples_to_keep]


# 🛠️ NEW OUTLIER FILTER: Keep genes where at least 2 samples have a count >= 3
# This completely eliminates [0, 0, 17] single-sample spikes.
min_count = 0.01
min_samples = 2
genes_to_keep = counts_df.columns[(counts_df >= min_count).sum(axis=0) >= min_samples]
counts_df = counts_df[genes_to_keep]


dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata,
    design="~condition",
    refit_cooks=True,
    inference=inference,
    # n_cpus=8, # n_cpus can be specified here or in the inference object
)  # runs normalization and dispersion estimation


dds.deseq2()

if SAVE:
    with open("dds.pkl", "wb") as f:
        pkl.dump(dds, f)

# PyDESeq2 automatically isolates 'condition' when running this contrast 
# while keeping the batch math corrected in the background.
# wt, light, dark
gp1 = "wt"
gp2 = "dark"
ds = DeseqStats(dds, contrast=["condition", gp1, gp2], inference=inference)

ds.summary(lfc_null=0.1, alt_hypothesis="greaterAbs")
ds.plot_MA(s=20)

if SAVE:
    with open("ds.pkl", "wb") as f:
        pkl.dump(ds, f)

# 🛠️ NEW ADDITION: Extract the results DataFrame and export to Excel
results_df = ds.results_df

# Optional: Sort the genes by significance (most significant at the top)
results_df = results_df.sort_values(by="padj", ascending=True)

# Export to Excel (the gene names are saved automatically as the row index)
results_df.to_excel(f"stat {gp1} vs {gp2}.xlsx")

# ==============================================================================
# VISUAL RECONSTRUCTION: VOLCANO PLOT OF DEGs
# ==============================================================================
# Extract metrics directly from your sorted results_df
# (Note: padj represents the Benjamini-Hochberg false discovery rate / Q-value)
plot_df = results_df.copy()
plot_df['log2FC'] = plot_df['log2FoldChange']
plot_df['-log10Q'] = -np.log10(plot_df['padj'])

# Clean any missing or infinite values generated from low-expression genes
plot_df = plot_df.dropna(subset=['log2FC', '-log10Q'])

# Define your biological thresholds (matching the vertical dashed layout)
FC_cutoff = 0.58           # Corresponds to approximately a 1.5-fold change boundary
Q_cutoff = -np.log10(0.05) # Corresponds to a Q-value (padj) < 0.05 significance cutoff

# Categorize genes into expression categories
plot_df['group'] = 'no-DEGS'
plot_df.loc[(plot_df['log2FC'] > FC_cutoff) & (plot_df['-log10Q'] > Q_cutoff), 'group'] = 'Up'
plot_df.loc[(plot_df['log2FC'] < -FC_cutoff) & (plot_df['-log10Q'] > Q_cutoff), 'group'] = 'Down'

# Configure strict font guidelines matching your graphic workflow
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)

# Exact color mapping matching the reference sample image assets
color_map = {
    'Up': '#E53935',       # Vibrant Red
    'no-DEGS': '#94A7C1',  # Soft Blue-Gray
    'Down': '#66BB6A'      # Apple Green
}

# Ensure specific layout layering by printing no-DEGS beneath active markers
for group_name in ['no-DEGS', 'Up', 'Down']:
    if group_name in plot_df['group'].values:
        group_data = plot_df[plot_df['group'] == group_name]
        ax.scatter(
            group_data['log2FC'],
            group_data['-log10Q'],
            color=color_map[group_name],
            alpha=0.8,
            s=16,
            linewidths=0,
            label=group_name,
            zorder=2 if group_name == 'no-DEGS' else 3
        )

# Construct triple structural boundary lines (Dashed layout)
ax.axvline(x=0, color='#666666', linestyle='--', linewidth=0.8, dashes=(5, 5), zorder=1)
ax.axvline(x=FC_cutoff, color='#666666', linestyle='--', linewidth=0.8, dashes=(5, 5), zorder=1)
ax.axvline(x=-FC_cutoff, color='#666666', linestyle='--', linewidth=0.8, dashes=(5, 5), zorder=1)
ax.axhline(y=Q_cutoff, color='#E0E0E0', linestyle=':', linewidth=0.8, zorder=1) # Optional baseline

# Replicate strict outward spine ticks frame formatting
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Long outward ticks padding rules
ax.tick_params(direction='out', length=6, width=0.8, labelsize=11, pad=5)

# Enforce crisp, balanced symmetric X limits centered around zero
max_fold = max(abs(plot_df['log2FC'].min()), abs(plot_df['log2FC'].max()))
x_limit = np.ceil(max_fold) if max_fold > 1 else 2
ax.set_xlim(-x_limit, x_limit)

# Standardize axis parameters to mimic the source composition
ax.set_xlabel(f"log2({gp1}/{gp2})", fontsize=13, labelpad=8)
ax.set_ylabel("-log10(Qvalue)", fontsize=13, labelpad=8)
ax.set_title("Volcano map of DEGs", fontsize=14, pad=18)

# Accurate label square marker legend tracking layout
ax.legend(
    loc='upper right',
    frameon=False,
    markerscale=2.0,
    handletextpad=0.5,
    fontsize=11
)

plt.tight_layout()
# Saves dynamically based on the exact variable comparison string
plt.savefig(f"Volcano_Plot_{gp1}_vs_{gp2}.png", bbox_inches='tight')
plt.show()