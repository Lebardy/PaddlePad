# ============================================================
# PaddlePad ML - Feature Engineering Tests
# ============================================================
#
# Covers feature_engineering.py: the skill and playstyle
# feature construction/direction/scaling functions, and the
# residualization math introduced this session.
#
# Does NOT cover: print/display functions (print_feature_*,
# print_scaling_summary), or K-Means/clustering logic (see
# test_playstyle_naming.py for the archetype-naming layer).
#
# Run with:  pytest test_feature_engineering.py
# ============================================================

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    ALL_FEATURES,
    STD_FEATURES,
    PLAYSTYLE_FEATURES,
    FEATURES_TO_RESIDUALIZE,
    create_ml_features,
    prepare_for_ml,
    scale_ml_features,
    create_playstyle_features,
    prepare_playstyle_features,
    scale_playstyle_features,
    residualize_playstyle_features,
)


# ================================================================
# Helpers
# ================================================================

def _make_player_profiles(n=6, seed=0):
    """
    Minimal player_profiles-shaped DataFrame carrying every
    column create_ml_features / create_playstyle_features
    require. Mirrors what player_profiles.aggregate_player_profiles
    actually produces.
    """

    rng = np.random.default_rng(seed)

    return pd.DataFrame({
        "player_id": [f"PLR_{i:03d}" for i in range(n)],

        "winner_rate_mean": rng.uniform(0.1, 0.9, n),
        "general_error_rate_mean": rng.uniform(0.1, 0.9, n),
        "dink_error_rate_mean": rng.uniform(0.1, 0.9, n),
        "drop_efficiency_mean": rng.uniform(0.1, 0.9, n),
        "aggression_mean": rng.uniform(0.1, 0.9, n),

        "winner_rate_std": rng.uniform(0.01, 0.3, n),
        "general_error_rate_std": rng.uniform(0.01, 0.3, n),
        "dink_error_rate_std": rng.uniform(0.01, 0.3, n),
        "drop_efficiency_std": rng.uniform(0.01, 0.3, n),
        "aggression_std": rng.uniform(0.01, 0.3, n),

        "drop_preference_rate_mean": rng.uniform(0.1, 0.9, n),
        "drop_preference_rate_std": rng.uniform(0.01, 0.3, n),
        "net_game_preference_rate_mean": rng.uniform(0.1, 0.9, n),
        "net_game_preference_rate_std": rng.uniform(0.01, 0.3, n),

        "average_drop_attempts": rng.uniform(5, 20, n),
        "average_winners": rng.uniform(1, 20, n),
        "average_unforced_errors": rng.uniform(1, 15, n),
        "average_match_duration": rng.uniform(10, 30, n),
    })


# ================================================================
# create_ml_features
# ================================================================

def test_create_ml_features_raises_on_missing_columns():
    df = pd.DataFrame({
        "player_id": ["P1"],
        "winner_rate_mean": [0.5]
    })

    with pytest.raises(ValueError):
        create_ml_features(df)


def test_create_ml_features_keeps_only_expected_columns():
    df = _make_player_profiles()

    result = create_ml_features(df)

    assert list(result.columns) == ["player_id"] + ALL_FEATURES


def test_create_ml_features_replaces_inf_and_nan_with_zero():
    df = _make_player_profiles(n=2)
    df.loc[0, "winner_rate_mean"] = np.inf
    df.loc[1, "aggression_std"] = np.nan

    result = create_ml_features(df)

    assert result.loc[0, "winner_rate_mean"] == 0.0
    assert result.loc[1, "aggression_std"] == 0.0


# ================================================================
# prepare_for_ml
# ================================================================

def test_prepare_for_ml_inverts_only_errors_and_stds():
    df = _make_player_profiles()
    ml_features = create_ml_features(df)
    oriented = prepare_for_ml(ml_features)

    assert np.allclose(
        oriented["general_error_rate_mean"],
        -ml_features["general_error_rate_mean"]
    )
    assert np.allclose(
        oriented["dink_error_rate_mean"],
        -ml_features["dink_error_rate_mean"]
    )

    for feature in STD_FEATURES:
        assert np.allclose(
            oriented[feature], -ml_features[feature]
        )

    assert np.allclose(
        oriented["winner_rate_mean"], ml_features["winner_rate_mean"]
    )
    assert np.allclose(
        oriented["drop_efficiency_mean"], ml_features["drop_efficiency_mean"]
    )
    assert np.allclose(
        oriented["aggression_mean"], ml_features["aggression_mean"]
    )


def test_prepare_for_ml_does_not_mutate_input():
    df = _make_player_profiles()
    ml_features = create_ml_features(df)
    before = ml_features.copy()

    prepare_for_ml(ml_features)

    pd.testing.assert_frame_equal(ml_features, before)


# ================================================================
# scale_ml_features
# ================================================================

def test_scale_ml_features_produces_standardized_output():
    df = _make_player_profiles(n=30)
    ml_features = create_ml_features(df)
    oriented = prepare_for_ml(ml_features)
    scaled, scaler = scale_ml_features(oriented)

    assert list(scaled["player_id"]) == list(df["player_id"])

    for feature in ALL_FEATURES:
        assert abs(scaled[feature].mean()) < 1e-8
        assert abs(scaled[feature].std(ddof=0) - 1.0) < 1e-8


# ================================================================
# create_playstyle_features
# ================================================================

def test_create_playstyle_features_computes_drop_usage_rate():
    df = _make_player_profiles(n=1)
    df.loc[0, "average_drop_attempts"] = 10.0
    df.loc[0, "average_match_duration"] = 20.0

    result = create_playstyle_features(df)

    assert result.loc[0, "drop_usage_rate"] == pytest.approx(0.5)


def test_create_playstyle_features_floors_duration_at_one_minute():
    df = _make_player_profiles(n=1)
    df.loc[0, "average_drop_attempts"] = 5.0
    df.loc[0, "average_match_duration"] = 0.2

    result = create_playstyle_features(df)

    assert result.loc[0, "drop_usage_rate"] == pytest.approx(5.0)


def test_create_playstyle_features_computes_error_to_winner_ratio():
    df = _make_player_profiles(n=1)
    df.loc[0, "average_unforced_errors"] = 6.0
    df.loc[0, "average_winners"] = 3.0

    result = create_playstyle_features(df)

    assert result.loc[0, "error_to_winner_ratio"] == pytest.approx(2.0)


def test_create_playstyle_features_floors_winners_at_one():
    df = _make_player_profiles(n=1)
    df.loc[0, "average_unforced_errors"] = 4.0
    df.loc[0, "average_winners"] = 0.3

    result = create_playstyle_features(df)

    assert result.loc[0, "error_to_winner_ratio"] == pytest.approx(4.0)


def test_create_playstyle_features_keeps_only_expected_columns():
    df = _make_player_profiles()

    result = create_playstyle_features(df)

    assert list(result.columns) == ["player_id"] + PLAYSTYLE_FEATURES


# ================================================================
# prepare_playstyle_features -- must NOT invert anything
# ================================================================

def test_prepare_playstyle_features_is_a_pure_passthrough():
    """
    Unlike prepare_for_ml, playstyle features are never sign-
    flipped -- they aren't "good/bad", just descriptive. This
    locks that design decision in: if someone "fixes" this to
    mirror prepare_for_ml, it would silently break residualization
    and every archetype direction downstream.
    """

    df = _make_player_profiles()
    playstyle_features = create_playstyle_features(df)

    result = prepare_playstyle_features(playstyle_features)

    pd.testing.assert_frame_equal(result, playstyle_features)


# ================================================================
# scale_playstyle_features
# ================================================================

def test_scale_playstyle_features_produces_standardized_output():
    df = _make_player_profiles(n=30)
    playstyle_features = create_playstyle_features(df)
    scaled, scaler = scale_playstyle_features(playstyle_features)

    for feature in PLAYSTYLE_FEATURES:
        assert abs(scaled[feature].mean()) < 1e-8
        assert abs(scaled[feature].std(ddof=0) - 1.0) < 1e-8


# ================================================================
# residualize_playstyle_features
# ================================================================

def test_residualize_removes_exact_linear_relationship():
    n = 10
    skill_scores = np.linspace(20, 90, n)
    aggression = 0.01 * skill_scores + 0.2  # perfectly linear in skill

    df = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "skill_group": ["Group A"] * n,
        "skill_score": skill_scores,
        "aggression_mean": aggression,
    })

    result = residualize_playstyle_features(
        df, features_to_residualize=["aggression_mean"]
    )

    assert np.allclose(result["aggression_mean"], 0.0, atol=1e-8)


def test_residualize_leaves_zero_correlation_with_skill():
    n = 40
    rng = np.random.default_rng(1)
    skill_scores = rng.uniform(0, 100, n)
    aggression = 0.005 * skill_scores + rng.normal(0, 0.05, n)

    df = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "skill_group": ["Group A"] * n,
        "skill_score": skill_scores,
        "aggression_mean": aggression,
    })

    result = residualize_playstyle_features(
        df, features_to_residualize=["aggression_mean"]
    )

    correlation = np.corrcoef(
        result["aggression_mean"], skill_scores
    )[0, 1]

    assert abs(correlation) < 1e-8


def test_residualize_leaves_unlisted_features_untouched():
    n = 10
    skill_scores = np.linspace(20, 90, n)

    df = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(n)],
        "skill_group": ["Group A"] * n,
        "skill_score": skill_scores,
        "aggression_mean": 0.01 * skill_scores,
        "drop_efficiency_std": np.linspace(0.1, 0.2, n),
    })

    result = residualize_playstyle_features(
        df, features_to_residualize=["aggression_mean"]
    )

    assert np.array_equal(
        result["drop_efficiency_std"].to_numpy(),
        df["drop_efficiency_std"].to_numpy()
    )


def test_residualize_fits_separately_per_skill_group():
    """
    Two groups with opposite-sign slopes. If residualization
    correctly fits each group independently, both groups'
    residuals end up uncorrelated with skill_score. If it
    incorrectly pooled both groups into one fit, at least one
    group would retain a real correlation.
    """

    # NOTE: each feature needs SOME real noise around its trend
    # line. A perfectly noiseless linear relationship residualizes
    # to values that are all ~0 (correct!), but correlating an
    # array of pure floating-point rounding noise against
    # anything is numerically undefined and can misleadingly
    # report a large "correlation" that's actually meaningless.

    n = 20
    rng = np.random.default_rng(2)

    skill_a = rng.uniform(0, 50, n)
    skill_b = rng.uniform(50, 100, n)

    feature_a = 0.02 * skill_a + 1.0 + rng.normal(0, 0.05, n)
    feature_b = -0.03 * skill_b + 5.0 + rng.normal(0, 0.05, n)

    df = pd.DataFrame({
        "player_id": [f"P{i}" for i in range(2 * n)],
        "skill_group": ["A"] * n + ["B"] * n,
        "skill_score": np.concatenate([skill_a, skill_b]),
        "aggression_mean": np.concatenate([feature_a, feature_b]),
    })

    result = residualize_playstyle_features(
        df, features_to_residualize=["aggression_mean"]
    )

    for group in ["A", "B"]:
        subset = result[result["skill_group"] == group]

        correlation = np.corrcoef(
            subset["aggression_mean"], subset["skill_score"]
        )[0, 1]

        assert abs(correlation) < 1e-8, (
            f"group {group} still correlated: {correlation}"
        )


def test_residualize_falls_back_to_raw_value_for_tiny_groups():
    """Groups with fewer than 3 players can't support a stable
    linear fit -- the feature should be left unchanged rather
    than crash or produce a nonsensical fit."""

    df = pd.DataFrame({
        "player_id": ["P1", "P2"],
        "skill_group": ["Tiny"] * 2,
        "skill_score": [40.0, 80.0],
        "aggression_mean": [0.3, 0.7],
    })

    result = residualize_playstyle_features(
        df, features_to_residualize=["aggression_mean"]
    )

    assert np.array_equal(
        result["aggression_mean"].to_numpy(),
        df["aggression_mean"].to_numpy()
    )


def test_residualize_falls_back_when_skill_score_has_no_variance():
    """If every player in a group has the identical skill_score,
    the regression slope is undefined -- should leave the
    feature unchanged instead of crashing on a divide by zero."""

    df = pd.DataFrame({
        "player_id": ["P1", "P2", "P3"],
        "skill_group": ["Flat"] * 3,
        "skill_score": [50.0, 50.0, 50.0],
        "aggression_mean": [0.1, 0.5, 0.9],
    })

    result = residualize_playstyle_features(
        df, features_to_residualize=["aggression_mean"]
    )

    assert np.array_equal(
        result["aggression_mean"].to_numpy(),
        df["aggression_mean"].to_numpy()
    )


def test_residualize_default_feature_list_matches_module_constant():
    n = 12
    rng = np.random.default_rng(3)
    skill_scores = rng.uniform(0, 100, n)

    data = {
        "player_id": [f"P{i}" for i in range(n)],
        "skill_group": ["Group A"] * n,
        "skill_score": skill_scores,
    }

    for feature in FEATURES_TO_RESIDUALIZE:
        data[feature] = rng.uniform(0, 1, n)

    df = pd.DataFrame(data)

    result = residualize_playstyle_features(df)

    for feature in FEATURES_TO_RESIDUALIZE:
        correlation = np.corrcoef(
            result[feature], skill_scores
        )[0, 1]

        assert abs(correlation) < 1e-8
