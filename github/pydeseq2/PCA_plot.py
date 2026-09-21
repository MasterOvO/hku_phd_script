import pandas as pd
from pydeseq2.dds import DeseqDataSet
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from pydeseq2.preprocessing import deseq2_norm




data = pd.read_excel(r"pydeseq2\gene_expression_read_count_only.xlsx", index_col=0).astype(int).T
metadata = pd.read_excel(r"pydeseq2\metadata.xlsx").set_index("sample")

norm_counts, size_factors = deseq2_norm(data)

X = norm_counts

X.columns = X.columns.astype(str)


pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

# Put into a DataFrame for plotting
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=X.index)
pca_df["condition"] = metadata.loc[pca_df.index, "condition"]

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=pca_df,
    x="PC1", y="PC2",
    hue="condition",
    style="condition",
    s=100
)


for i, sample in enumerate(pca_df.index):
    plt.text(
        pca_df["PC1"][i] + 0.2,   # small offset so text doesn't overlap the point
        pca_df["PC2"][i] + 0.2,
        sample,
        fontsize=9
    )

plt.title("PCA of RNA-seq samples")
plt.savefig("PCA.png")
plt.show()

