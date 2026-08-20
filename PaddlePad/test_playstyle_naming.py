# ============================================================
# PaddlePad ML - Playstyle Archetype Naming Tests
# ============================================================
#
# Covers only the archetype-naming layer in clustering.py:
# generate_playstyle_archetype_name, _skill_group_prefix, and
# the collision-escalation logic in
# _generate_unique_playstyle_names.
#
# It does NOT cover the rest of the pipeline (feature
# engineering, residualization, K-Means selection, the
# synthetic generator, or the pklmart converter) -- none of
# that has automated tests yet.
#
# Run with:  pytest test_playstyle_naming.py
# ============================================================

import pandas as pd
import pytest

from clustering import (
    generate_playstyle_archetype_name,
    _generate_unique_playstyle_names,
    _skill_group_prefix,
    DEFAULT_IDENTITY_NOUN,
    MEANINGFUL_EFFECT_SIZE,
    PLAYSTYLE_CLUSTERING_FEATURES,
)


def _profile(**overrides):
    """
    A neutral scaled-centroid profile (every feature at exactly
    the skill group's average, z=0) with the given features
    overridden. Mirrors one row of the scaled centroids that
    interpret_playstyle_clusters actually passes in.
    """

    values = {
        feature: 0.0
        for feature in PLAYSTYLE_CLUSTERING_FEATURES
    }

    values.update(overrides)

    return pd.Series(values)


# ================================================================
# Grammar: name is always "prefix [adjectives] noun"
# ================================================================

def test_flat_profile_produces_just_prefix_and_default_noun():
    profile = _profile()

    name = generate_playstyle_archetype_name(
        profile, "Developing / Lower-Performance"
    )

    assert name == f"Developing {DEFAULT_IDENTITY_NOUN}"


def test_name_always_starts_with_the_skill_prefix():
    profile = _profile(aggression_mean=0.9, drop_efficiency_mean=0.8)

    name = generate_playstyle_archetype_name(
        profile, "Higher-Performance"
    )

    assert name.startswith("Advanced ")


def test_name_never_ends_with_a_bare_adjective():
    """
    Every name must end in a noun (an identity label, or the
    default), never trail off on an adjective -- that's what
    made the old naming read as an incoherent word list.
    """

    identity_nouns = {
        "Driver", "Dropper", "Power Player", "Net Player",
        DEFAULT_IDENTITY_NOUN
    }

    profile = _profile(
        aggression_mean=0.9,
        error_to_winner_ratio=-0.7,
        aggression_std=0.5
    )

    name = generate_playstyle_archetype_name(
        profile, "Higher-Performance"
    )

    assert any(
        name.endswith(noun) for noun in identity_nouns
    )


# ================================================================
# Direction: high z -> high-side label, low z -> low-side label
# ================================================================

def test_high_and_low_trait_values_pick_opposite_labels():
    high = _profile(aggression_mean=0.9)
    low = _profile(aggression_mean=-0.9)

    high_name = generate_playstyle_archetype_name(
        high, "Higher-Performance"
    )
    low_name = generate_playstyle_archetype_name(
        low, "Higher-Performance"
    )

    assert "Aggressive" in high_name
    assert "Patient" in low_name


@pytest.mark.parametrize(
    "feature,low_label,high_label",
    [
        ("drop_preference_rate_mean", "Driver", "Dropper"),
        ("net_game_preference_rate_mean", "Power Player", "Net Player"),
    ]
)
def test_identity_noun_reflects_strongest_identity_feature(
    feature, low_label, high_label
):
    high = _profile(**{feature: 0.9})
    low = _profile(**{feature: -0.9})

    assert generate_playstyle_archetype_name(
        high, "Higher-Performance"
    ).endswith(high_label)

    assert generate_playstyle_archetype_name(
        low, "Higher-Performance"
    ).endswith(low_label)


# ================================================================
# Effect-size cutoff
# ================================================================

def test_traits_below_effect_size_cutoff_are_ignored():
    barely_below = _profile(
        aggression_mean=MEANINGFUL_EFFECT_SIZE - 0.01
    )

    name = generate_playstyle_archetype_name(
        barely_below, "Higher-Performance"
    )

    assert name == f"Advanced {DEFAULT_IDENTITY_NOUN}"


def test_traits_at_or_above_effect_size_cutoff_are_included():
    barely_above = _profile(
        aggression_mean=MEANINGFUL_EFFECT_SIZE + 0.01
    )

    name = generate_playstyle_archetype_name(
        barely_above, "Higher-Performance"
    )

    assert "Aggressive" in name


# ================================================================
# max_traits parameter (used by collision escalation)
# ================================================================

def test_max_traits_limits_adjective_count():
    profile = _profile(
        aggression_mean=0.9,
        drop_efficiency_mean=0.8,
        error_to_winner_ratio=0.7
    )

    name_two = generate_playstyle_archetype_name(
        profile, "Higher-Performance", max_traits=2
    )

    name_three = generate_playstyle_archetype_name(
        profile, "Higher-Performance", max_traits=3
    )

    assert len(name_three.split()) == len(name_two.split()) + 1


# ================================================================
# Skill group prefix, including groups not explicitly known
# ================================================================

@pytest.mark.parametrize(
    "skill_group,expected_prefix",
    [
        ("Developing / Lower-Performance", "Developing"),
        ("Higher-Performance", "Advanced"),
        ("Intermediate-Performance", "Intermediate"),
        ("Performance Group 3", "Group 3"),
    ]
)
def test_skill_group_prefix_mapping(skill_group, expected_prefix):
    assert _skill_group_prefix(skill_group) == expected_prefix


# ================================================================
# Collision handling across a whole skill group's clusters
# ================================================================

def test_clearly_distinct_clusters_get_distinct_names():
    scaled_centroids = pd.DataFrame(
        [
            _profile(aggression_mean=0.9),
            _profile(aggression_mean=-0.9)
        ]
    )

    names = _generate_unique_playstyle_names(
        scaled_centroids, "Higher-Performance"
    )

    assert len(set(names.values())) == 2
    assert not any("Type" in name for name in names.values())


def test_collision_at_default_detail_is_resolved_by_escalating():
    """
    Both clusters tie on the two strongest traits and only
    differ on a third, weaker one -- max_traits=2 alone would
    produce the same name for both. Naming should notice the
    collision and pull in the third trait automatically,
    without ever falling back to a numbered suffix.
    """

    shared = dict(
        aggression_mean=0.9,
        drop_efficiency_mean=0.85
    )

    scaled_centroids = pd.DataFrame(
        [
            _profile(**shared, winner_rate_std=0.21),
            _profile(**shared, winner_rate_std=-0.21)
        ]
    )

    names = _generate_unique_playstyle_names(
        scaled_centroids, "Higher-Performance"
    )

    assert len(set(names.values())) == 2
    assert not any("Type" in name for name in names.values())


def test_genuinely_identical_clusters_fall_back_to_numbered_suffix():
    scaled_centroids = pd.DataFrame(
        [
            _profile(aggression_mean=0.9),
            _profile(aggression_mean=0.9)
        ]
    )

    names = _generate_unique_playstyle_names(
        scaled_centroids, "Developing / Lower-Performance"
    )

    assert len(set(names.values())) == 2
    assert sum("Type" in name for name in names.values()) == 1


def test_more_than_two_clusters_all_get_unique_names():
    scaled_centroids = pd.DataFrame(
        [
            _profile(aggression_mean=0.9),
            _profile(aggression_mean=-0.9),
            _profile(net_game_preference_rate_mean=0.9),
            _profile(net_game_preference_rate_mean=-0.9),
            _profile(drop_efficiency_std=0.6)
        ]
    )

    names = _generate_unique_playstyle_names(
        scaled_centroids, "Higher-Performance"
    )

    assert len(set(names.values())) == 5
