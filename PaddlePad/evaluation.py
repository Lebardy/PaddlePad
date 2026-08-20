# ============================================================
# PaddlePad ML - Evaluation
# ============================================================

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)


# ============================================================
# GENERIC CLUSTER QUALITY
# ============================================================

def calculate_cluster_metrics(
    X,
    labels
):
    """
    Calculate standard clustering-quality metrics.

    Silhouette:
        Higher is better.

    Davies-Bouldin:
        Lower is better.

    Calinski-Harabasz:
        Higher is generally better.
    """

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError(
            "At least two clusters are required."
        )

    return {
        "silhouette": silhouette_score(
            X,
            labels
        ),
        "davies_bouldin": davies_bouldin_score(
            X,
            labels
        ),
        "calinski_harabasz": calinski_harabasz_score(
            X,
            labels
        )
    }


# ============================================================
# LEVEL 1: SKILL CLUSTER EVALUATION
# ============================================================

def evaluate_skill_clustering(
    clustering_data,
    skill_clustered,
    clustering_features
):
    """
    Evaluate the first-level skill clustering.
    """

    X = clustering_data[
        clustering_features
    ].copy()

    labels = skill_clustered[
        "skill_cluster"
    ].to_numpy()

    metrics = calculate_cluster_metrics(
        X,
        labels
    )

    print(
        "\n=== SKILL CLUSTER QUALITY ==="
    )

    print(
        f"Silhouette Score: "
        f"{metrics['silhouette']:.3f}"
    )

    print(
        f"Davies-Bouldin Index: "
        f"{metrics['davies_bouldin']:.3f}"
    )

    print(
        f"Calinski-Harabasz Score: "
        f"{metrics['calinski_harabasz']:.1f}"
    )

    return metrics


# ============================================================
# LEVEL 2: PLAYSTYLE CLUSTER EVALUATION
# ============================================================

def evaluate_playstyle_clustering(
    playstyle_data,
    playstyle_clustered,
    playstyle_features,
    skill_group
):
    """
    Evaluate second-level playstyle clustering for
    one skill group.
    """

    X = playstyle_data[
        playstyle_features
    ].copy()

    labels = playstyle_clustered[
        "playstyle_cluster"
    ].to_numpy()

    metrics = calculate_cluster_metrics(
        X,
        labels
    )

    print(
        "\n=== PLAYSTYLE CLUSTER QUALITY ==="
    )

    print(
        f"Skill group: "
        f"{skill_group}"
    )

    print(
        f"Silhouette Score: "
        f"{metrics['silhouette']:.3f}"
    )

    print(
        f"Davies-Bouldin Index: "
        f"{metrics['davies_bouldin']:.3f}"
    )

    print(
        f"Calinski-Harabasz Score: "
        f"{metrics['calinski_harabasz']:.1f}"
    )

    return metrics


# ============================================================
# HIDDEN SYNTHETIC LABELS
# ============================================================

def build_hidden_player_profiles(
    raw_logs
):
    """
    Recover one hidden synthetic profile per player.

    IMPORTANT:
    These labels are NEVER given to the ML pipeline.

    They exist only for evaluating the synthetic experiment.
    """

    hidden_profiles = (
        raw_logs[
            [
                "player_id",
                "_hidden_profile"
            ]
        ]
        .drop_duplicates(
            subset="player_id"
        )
        .copy()
    )

    return hidden_profiles


# ============================================================
# BROAD HIDDEN GROUP
# ============================================================

def convert_hidden_profile_to_broad_group(
    hidden_profile
):
    """
    Convert the detailed hidden synthetic archetype into
    a broad performance group.

    Beginner hidden profiles:
        -> Developing

    Everything else:
        -> Higher
    """

    if hidden_profile.startswith(
        "Beginner"
    ):
        return "Developing"

    return "Higher"


# ============================================================
# LEVEL 1 VS HIDDEN STRUCTURE
# ============================================================

def evaluate_skill_against_hidden_profiles(
    hidden_profiles,
    skill_clustered
):
    """
    Compare Level 1 discovered skill clusters against the
    broad hidden synthetic structure.

    This is external evaluation only.
    """

    comparison = (
        skill_clustered[
            [
                "player_id",
                "skill_cluster"
            ]
        ]
        .merge(
            hidden_profiles,
            on="player_id",
            how="inner",
            validate="one_to_one"
        )
    )

    comparison[
        "hidden_group"
    ] = (
        comparison[
            "_hidden_profile"
        ]
        .apply(
            convert_hidden_profile_to_broad_group
        )
    )

    # --------------------------------------------------------
    # Encode hidden broad groups
    # --------------------------------------------------------

    hidden_numeric = (
        comparison[
            "hidden_group"
        ]
        .map(
            {
                "Developing": 0,
                "Higher": 1
            }
        )
    )

    discovered_numeric = (
        comparison[
            "skill_cluster"
        ]
    )

    ari = adjusted_rand_score(
        hidden_numeric,
        discovered_numeric
    )

    print(
        "\n=== SKILL CLUSTER VS HIDDEN STRUCTURE ==="
    )

    print(
        f"Hidden-group ARI: "
        f"{ari:.3f}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "Hidden synthetic profiles are used only "
        "for evaluation and are never given to K-Means."
    )

    return (
        comparison,
        ari
    )


# ============================================================
# PLAYSTYLE STABILITY
# ============================================================

def evaluate_playstyle_stability(
    playstyle_data,
    playstyle_features,
    random_states,
    selected_k
):
    """
    Evaluate whether a fixed K produces stable playstyle
    assignments across different random seeds.
    """

    X = playstyle_data[
        playstyle_features
    ].copy()

    results = []

    reference_labels = None

    for seed in random_states:

        model = KMeans(
            n_clusters=selected_k,
            random_state=seed,
            n_init=20
        )

        labels = model.fit_predict(
            X
        )

        silhouette = silhouette_score(
            X,
            labels
        )

        if reference_labels is None:

            reference_labels = labels

            ari_vs_reference = 1.0

        else:

            ari_vs_reference = (
                adjusted_rand_score(
                    reference_labels,
                    labels
                )
            )

        results.append(
            {
                "seed": seed,
                "k": selected_k,
                "silhouette": silhouette,
                "ari_vs_reference": ari_vs_reference
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n=== PLAYSTYLE FIXED-K STABILITY ==="
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print(
        f"\nAverage ARI: "
        f"{results_df['ari_vs_reference'].mean():.3f}"
    )

    print(
        f"Average Silhouette: "
        f"{results_df['silhouette'].mean():.3f}"
    )

    print(
        f"Silhouette Std: "
        f"{results_df['silhouette'].std():.3f}"
    )

    return results_df