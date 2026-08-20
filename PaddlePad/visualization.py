# ============================================================
# PaddlePad ML - Visualization
# ============================================================

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# ============================================================
# LEVEL 1: SKILL CLUSTERS
# ============================================================

def plot_skill_clusters(
    clustering_data,
    cluster_labels,
    clustering_features,
    title="PaddlePad Skill Clusters"
):
    """
    Reduce the standardized skill feature space to two
    dimensions using PCA for visualization only.

    PCA is NOT used for K-Means.
    """

    X = clustering_data[
        clustering_features
    ].copy()

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(
        X
    )

    plt.figure(
        figsize=(9, 7)
    )

    for cluster in sorted(
        set(cluster_labels)
    ):

        mask = (
            cluster_labels
            ==
            cluster
        )

        plt.scatter(
            X_pca[
                mask,
                0
            ],
            X_pca[
                mask,
                1
            ],
            label=f"Skill Cluster {cluster}"
        )

    variance = (
        pca.explained_variance_ratio_
        * 100
    )

    plt.xlabel(
        f"PC1 ({variance[0]:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({variance[1]:.1f}% variance)"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    return pca


# ============================================================
# LEVEL 2: PLAYSTYLE CLUSTERS
# ============================================================

def plot_playstyle_clusters(
    playstyle_data,
    cluster_labels,
    playstyle_features,
    title
):
    """
    PCA visualization of one playstyle clustering branch.

    PCA is used only for visualization.
    """

    X = playstyle_data[
        playstyle_features
    ].copy()

    pca = PCA(
        n_components=2
    )

    X_pca = pca.fit_transform(
        X
    )

    plt.figure(
        figsize=(9, 7)
    )

    for cluster in sorted(
        set(cluster_labels)
    ):

        mask = (
            cluster_labels
            ==
            cluster
        )

        plt.scatter(
            X_pca[
                mask,
                0
            ],
            X_pca[
                mask,
                1
            ],
            label=f"Playstyle Cluster {cluster}"
        )

    variance = (
        pca.explained_variance_ratio_
        * 100
    )

    plt.xlabel(
        f"PC1 ({variance[0]:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({variance[1]:.1f}% variance)"
    )

    plt.title(
        title
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    return pca