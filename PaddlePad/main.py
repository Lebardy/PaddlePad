# ============================================================
# PaddlePad ML - Main Test Pipeline
# ============================================================
#
# Current architecture:
#
# Match Logs
#     ↓
# Player Profiles
#     ↓
# Feature Engineering
#     ↓
# ┌──────────────────────────────┐
# │ Skill Feature Space          │
# │ 10 standardized features     │
# └──────────────┬───────────────┘
#                ↓
#          K-Means #1
#                ↓
#        Discovered Skill Groups
#                ↓
# ┌──────────────────────────────┐
# │ Playstyle Feature Space      │
# │ 13 standardized features,    │
# │ 7 residualized against skill │
# └──────────────┬───────────────┘
#                ↓
#          K-Means #2
#                ↓
#        Discovered Playstyles
#                ↓
#      Archetype Interpretation
#                ↓
#       Final Player Profiles
#
# Hidden synthetic labels are used ONLY for evaluation.
# PCA is used ONLY for visualization.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from data_generator import (
    generate_unlabeled_match_logs
)

from player_profiles import (
    aggregate_player_profiles
)

from feature_engineering import (
    create_ml_features,
    prepare_for_ml,
    scale_ml_features,
    create_playstyle_features,
    prepare_playstyle_features,
    residualize_playstyle_features,
    scale_playstyle_features,
    print_feature_information,
    print_feature_directions,
    print_feature_summary,
    print_scaling_summary
)

from skill_model import (
    build_skill_model,
    print_skill_summary,
    print_skill_profiles
)

from clustering import (
    CLUSTERING_FEATURES,
    PLAYSTYLE_CLUSTERING_FEATURES,
    prepare_clustering_data,
    print_clustering_input_summary,
    print_clustering_samples,
    test_skill_k_values,
    cluster_skill_groups,
    print_skill_cluster_profiles,
    interpret_skill_clusters,
    apply_skill_cluster_labels,
    print_skill_group_summary,
    test_playstyle_k_values,
    cluster_playstyles,
    print_playstyle_cluster_profiles,
    test_playstyle_stability,
    interpret_playstyle_clusters,
    apply_playstyle_archetypes,
    print_playstyle_archetype_summary,
    print_playstyle_evidence,
    build_final_player_profiles,
    print_final_player_profiles
)

from evaluation import (
    evaluate_skill_clustering,
    evaluate_playstyle_clustering,
    build_hidden_player_profiles,
    evaluate_skill_against_hidden_profiles
)

from visualization import (
    plot_skill_clusters,
    plot_playstyle_clusters
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # STEP 1: GENERATE MATCH DATA
    # ========================================================

    print(
        "Generating UNLABELED synthetic match data..."
    )

    raw_logs = (
        generate_unlabeled_match_logs(
            num_players=240,
            matches_per_player=10,
            random_state=42
        )
    )

    # Keep hidden labels separately for evaluation only.
    hidden_player_profiles = (
        build_hidden_player_profiles(
            raw_logs
        )
    )

    # Remove hidden labels before ML processing.
    raw_logs = raw_logs.drop(
        columns=[
            "_hidden_profile"
        ]
    )

    # ========================================================
    # VERIFY MATCH DATA
    # ========================================================

    print(
        f"\nGenerated {len(raw_logs)} unlabeled match logs."
    )

    print(
        f"Players: {raw_logs['player_id'].nunique()}"
    )

    print(
        f"Matches per player: "
        f"{raw_logs.groupby('player_id').size().iloc[0]}"
    )

    print(
        "\n=== MATCH DATA ==="
    )

    print(
        raw_logs
        .head(5)
        .to_string(index=False)
    )

    # ========================================================
    # STEP 2: AGGREGATE MATCHES INTO PLAYER PROFILES
    # ========================================================

    print(
        "\nAggregating matches into player profiles..."
    )

    player_profiles = (
        aggregate_player_profiles(
            raw_logs
        )
    )

    print(
        "\n=== PLAYER PROFILE DATASET ==="
    )

    print(
        f"Players: {len(player_profiles)}"
    )

    print(
        f"Columns: {len(player_profiles.columns)}"
    )

    print(
        "\n=== FIRST 5 PLAYER PROFILES ==="
    )

    print(
        player_profiles
        .head(5)
        .to_string(index=False)
    )

    # ========================================================
    # STEP 3: CREATE SKILL FEATURES
    # ========================================================

    print(
        "\nCreating ML feature dataset..."
    )

    ml_features = (
        create_ml_features(
            player_profiles
        )
    )

    print_feature_information(
        ml_features
    )

    print_feature_summary(
        ml_features
    )

    print(
        "\nPreparing ML-oriented feature directions..."
    )

    ml_oriented_features = (
        prepare_for_ml(
            ml_features
        )
    )

    print_feature_directions(
        ml_features,
        ml_oriented_features
    )

    print(
        "\nScaling ML features..."
    )

    (
        scaled_features,
        scaler
    ) = scale_ml_features(
        ml_oriented_features
    )

    print_scaling_summary(
        scaled_features
    )

    print(
        "\n=== FIRST 5 SCALED PROFILES ==="
    )

    print(
        scaled_features
        .head(5)
        .to_string(index=False)
    )

    # ========================================================
    # STEP 4: STATISTICAL SKILL SCORE
    # ========================================================
    #
    # This is an independent reference score.
    # It is NOT provided to K-Means.
    # ========================================================

    print(
        "\nCalculating player skill scores..."
    )

    skill_profiles = (
        build_skill_model(
            player_profiles
        )
    )

    print_skill_summary(
        skill_profiles
    )

    print_skill_profiles(
        skill_profiles,
        number_of_players=10
    )

    # ========================================================
    # STEP 5: PREPARE LEVEL 1 SKILL CLUSTERING DATA
    # ========================================================

    print(
        "\nPreparing skill clustering dataset..."
    )

    clustering_data = (
        prepare_clustering_data(
            skill_profiles,
            scaled_features
        )
    )

    print_clustering_input_summary(
        clustering_data
    )

    print_clustering_samples(
        clustering_data,
        number_of_players=5
    )

    # ========================================================
    # STEP 6: LEVEL 1 K-MEANS - SKILL GROUPS
    # ========================================================

    print(
        "\nTesting first-level skill clustering..."
    )

    (
        best_skill_k,
        skill_k_results
    ) = test_skill_k_values(
        clustering_data,
        k_min=2,
        k_max=5,
        random_state=42
    )

    print(
        "\nRunning final skill clustering..."
    )

    (
        skill_clustered,
        skill_kmeans
    ) = cluster_skill_groups(
        clustering_data,
        best_skill_k,
        random_state=42
    )

    # --------------------------------------------------------
    # Level 1 evaluation
    # --------------------------------------------------------

    skill_metrics = evaluate_skill_clustering(
        clustering_data,
        skill_clustered,
        CLUSTERING_FEATURES
    )

    hidden_comparison, hidden_ari = (
        evaluate_skill_against_hidden_profiles(
            hidden_player_profiles,
            skill_clustered
        )
    )

    # --------------------------------------------------------
    # Interpret Level 1 groups
    # --------------------------------------------------------

    skill_cluster_labels = (
        interpret_skill_clusters(
            skill_clustered
        )
    )

    skill_clustered = (
        apply_skill_cluster_labels(
            skill_clustered,
            skill_cluster_labels
        )
    )

    print_skill_group_summary(
        skill_clustered,
        skill_cluster_labels
    )

    print_skill_cluster_profiles(
        clustering_data,
        skill_clustered,
        scaler
    )

    print(
        "\n=== SAMPLE SKILL CLUSTER ASSIGNMENTS ==="
    )

    print(
        skill_clustered[
            [
                "player_id",
                "skill_score",
                "skill_cluster",
                "skill_group"
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    # ========================================================
    # STEP 7: CREATE PLAYSTYLE FEATURE SPACE
    # ========================================================

    print(
        "\nCreating dedicated playstyle features..."
    )

    playstyle_features = (
        create_playstyle_features(
            player_profiles
        )
    )

    print(
        "\n=== PLAYSTYLE FEATURES ==="
    )

    print(
        playstyle_features
        .head(5)
        .to_string(index=False)
    )

    print(
        f"\nPlaystyle features: "
        f"{len(playstyle_features.columns) - 1}"
    )

    # ========================================================
    # STEP 7b: RESIDUALIZE SKILL-CORRELATED PLAYSTYLE FEATURES
    # ========================================================
    #
    # Several playstyle features (aggression, drop efficiency,
    # error-to-winner ratio, drop usage rate, and several
    # standard deviations) are strongly correlated with skill
    # within a skill group. Residualizing removes the linear
    # relationship with skill_score so Level 2 clustering finds
    # style differences rather than disguised skill differences.
    #
    # `playstyle_features` (raw) is kept around unmodified for
    # later interpretation/reporting; only the version fed to
    # scaling and clustering is adjusted.
    # ========================================================

    print(
        "\nResidualizing skill-correlated playstyle features..."
    )

    playstyle_with_skill = (
        playstyle_features.merge(
            skill_clustered[
                [
                    "player_id",
                    "skill_group",
                    "skill_score"
                ]
            ],
            on="player_id",
            how="inner",
            validate="one_to_one"
        )
    )

    playstyle_features_adjusted = (
        residualize_playstyle_features(
            playstyle_with_skill
        )
    )

    playstyle_oriented = (
        prepare_playstyle_features(
            playstyle_features_adjusted
        )
    )

    (
        scaled_playstyle_features,
        playstyle_scaler
    ) = scale_playstyle_features(
        playstyle_oriented
    )

    print(
        "\n=== PLAYSTYLE STANDARDIZATION ==="
    )

    print(
        scaled_playstyle_features
        .drop(columns=["player_id"])
        .agg(["mean", "std"])
        .transpose()
        .to_string()
    )

    # ========================================================
    # STEP 8: MERGE PLAYSTYLE FEATURES WITH SKILL GROUPS
    # ========================================================

    playstyle_cluster_data = (
        skill_clustered[
            [
                "player_id",
                "skill_cluster",
                "skill_group",
                "skill_score"
            ]
        ]
        .merge(
            scaled_playstyle_features,
            on="player_id",
            how="inner",
            validate="one_to_one"
        )
    )

    if len(playstyle_cluster_data) != len(
        skill_clustered
    ):
        raise ValueError(
            "Player count changed while combining "
            "skill groups and playstyle features."
        )

    # ========================================================
    # STEP 9: LEVEL 2 - DEVELOPING PLAYSTYLES
    # ========================================================

    developing_group = (
        "Developing / Lower-Performance"
    )

    print(
        "\nTesting playstyles inside:"
    )

    print(
        f"  {developing_group}"
    )

    (
        developing_data,
        best_developing_playstyle_k,
        developing_playstyle_k_results
    ) = test_playstyle_k_values(
        playstyle_cluster_data,
        developing_group,
        k_min=2,
        k_max=5,
        random_state=42
    )

    (
        developing_playstyles,
        developing_playstyle_model,
        developing_playstyle_counts
    ) = cluster_playstyles(
        developing_data,
        best_developing_playstyle_k,
        random_state=42
    )

    developing_metrics = evaluate_playstyle_clustering(
        developing_data,
        developing_playstyles,
        PLAYSTYLE_CLUSTERING_FEATURES,
        developing_group
    )

    # --------------------------------------------------------
    # Developing archetypes
    # --------------------------------------------------------

    (
        developing_archetype_map,
        developing_centroids,
        developing_scaled_centroids
    ) = interpret_playstyle_clusters(
        developing_data,
        developing_playstyles,
        playstyle_features
    )

    developing_playstyles = (
        apply_playstyle_archetypes(
            developing_playstyles,
            developing_archetype_map
        )
    )

    print_playstyle_archetype_summary(
        developing_playstyles,
        developing_archetype_map
    )

    print_playstyle_evidence(
        developing_playstyles,
        developing_archetype_map,
        developing_centroids
    )

    print(
        "\n=== DEVELOPING PLAYSTYLE RESULT ==="
    )

    print(
        f"Selected K: {best_developing_playstyle_k}"
    )

    for cluster_id, count in (
        developing_playstyle_counts.items()
    ):
        print(
            f"Playstyle Cluster {cluster_id}: "
            f"{count} players"
        )

    print_playstyle_cluster_profiles(
        developing_data,
        developing_playstyles,
        playstyle_features
    )

    developing_stability = (
        test_playstyle_stability(
            playstyle_cluster_data,
            developing_group,
            random_states=[
                1,
                7,
                21,
                42,
                100
            ],
            k_min=2,
            k_max=5
        )
    )

    # ========================================================
    # STEP 10: LEVEL 2 - HIGHER-PERFORMANCE PLAYSTYLES
    # ========================================================

    higher_group = (
        "Higher-Performance"
    )

    print(
        "\nTesting playstyles inside:"
    )

    print(
        f"  {higher_group}"
    )

    (
        higher_data,
        best_higher_playstyle_k,
        higher_playstyle_k_results
    ) = test_playstyle_k_values(
        playstyle_cluster_data,
        higher_group,
        k_min=2,
        k_max=5,
        random_state=42
    )

    (
        higher_playstyles,
        higher_playstyle_model,
        higher_playstyle_counts
    ) = cluster_playstyles(
        higher_data,
        best_higher_playstyle_k,
        random_state=42
    )

    higher_metrics = evaluate_playstyle_clustering(
        higher_data,
        higher_playstyles,
        PLAYSTYLE_CLUSTERING_FEATURES,
        higher_group
    )

    # --------------------------------------------------------
    # Higher-performance archetypes
    # --------------------------------------------------------

    (
        higher_archetype_map,
        higher_centroids,
        higher_scaled_centroids
    ) = interpret_playstyle_clusters(
        higher_data,
        higher_playstyles,
        playstyle_features
    )

    higher_playstyles = (
        apply_playstyle_archetypes(
            higher_playstyles,
            higher_archetype_map
        )
    )

    print_playstyle_archetype_summary(
        higher_playstyles,
        higher_archetype_map
    )

    print_playstyle_evidence(
        higher_playstyles,
        higher_archetype_map,
        higher_centroids
    )

    print(
        "\n=== HIGHER-PERFORMANCE PLAYSTYLE RESULT ==="
    )

    print(
        f"Selected K: {best_higher_playstyle_k}"
    )

    for cluster_id, count in (
        higher_playstyle_counts.items()
    ):
        print(
            f"Playstyle Cluster {cluster_id}: "
            f"{count} players"
        )

    print_playstyle_cluster_profiles(
        higher_data,
        higher_playstyles,
        playstyle_features
    )

    higher_stability = (
        test_playstyle_stability(
            playstyle_cluster_data,
            higher_group,
            random_states=[
                1,
                7,
                21,
                42,
                100
            ],
            k_min=2,
            k_max=5
        )
    )

    # ========================================================
    # STEP 11: FINAL PLAYER PROFILES
    # ========================================================

    final_profiles = (
        build_final_player_profiles(
            developing_playstyles,
            higher_playstyles,
            playstyle_features
        )
    )

    print_final_player_profiles(
        final_profiles,
        number_of_players=20
    )

    # ========================================================
    # STEP 12: PCA VISUALIZATION
    # ========================================================

    print(
        "\nGenerating PCA visualizations..."
    )

    plot_skill_clusters(
        clustering_data,
        skill_clustered[
            "skill_cluster"
        ].to_numpy(),
        CLUSTERING_FEATURES,
        title="PaddlePad Level 1 Skill Clusters"
    )

    plot_playstyle_clusters(
        developing_data,
        developing_playstyles[
            "playstyle_cluster"
        ].to_numpy(),
        PLAYSTYLE_CLUSTERING_FEATURES,
        "Developing / Lower-Performance Playstyles"
    )

    plot_playstyle_clusters(
        higher_data,
        higher_playstyles[
            "playstyle_cluster"
        ].to_numpy(),
        PLAYSTYLE_CLUSTERING_FEATURES,
        "Higher-Performance Playstyles"
    )

    # ========================================================
    # CURRENT PIPELINE STATUS
    # ========================================================

    print(
        "\n=== CURRENT PIPELINE ==="
    )

    print(
        "1. Match generation                   [DONE]"
    )

    print(
        "2. Player profile aggregation         [DONE]"
    )

    print(
        "3. Skill feature engineering          [DONE]"
    )

    print(
        "4. Skill feature direction            [DONE]"
    )

    print(
        "5. Skill StandardScaler               [DONE]"
    )

    print(
        "6. Statistical skill score             [DONE]"
    )

    print(
        "7. Level 1 skill K-Means               [DONE]"
    )

    print(
        "8. Skill cluster interpretation        [DONE]"
    )

    print(
        "9. Playstyle feature engineering       [DONE]"
    )

    print(
        "10. Playstyle StandardScaler             [DONE]"
    )

    print(
        "11. Developing playstyle K-Means       [DONE]"
    )

    print(
        "12. Developing playstyle evaluation    [DONE]"
    )

    print(
        "13. Developing playstyle stability     [DONE]"
    )

    print(
        "14. Higher-performance playstyles      [DONE]"
    )

    print(
        "15. Higher-performance evaluation     [DONE]"
    )

    print(
        "16. Higher-performance stability      [DONE]"
    )

    print(
        "17. Final player profiles              [DONE]"
    )

    print(
        "18. PCA visualization                  [DONE]"
    )

    print(
        "\nProcess finished."
    )