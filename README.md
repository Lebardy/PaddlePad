# PaddlePad ML

The ML pipeline for PaddlePad: turns raw umpire-logged pickleball
match data into a skill score and playstyle archetype for each
player.

## What's here

- `player_profiles.py` - turns multiple raw match rows per player into
  one row of aggregated stats (mean + consistency).
- `skill_model.py` - the 0-100 skill score and Beginner/Intermediate/
  Professional tier, from those aggregated stats.
- `feature_engineering.py` - builds the skill and playstyle feature
  sets, including `residualize_playstyle_features` (removes skill's
  influence from playstyle features so archetypes aren't just skill
  in disguise).
- `clustering.py` - the two-level K-Means (skill groups, then
  playstyle archetypes within each group) and the archetype-naming
  logic.

These four files have zero dependencies on each other or on any
other file -- only on `numpy`, `pandas`, and `scikit-learn`.

## Call order

```
aggregate_player_profiles(raw_match_logs)
    -> build_skill_model(player_profiles)
    -> create_ml_features / prepare_for_ml / scale_ml_features
    -> prepare_clustering_data -> test_skill_k_values -> cluster_skill_groups
    -> interpret_skill_clusters -> apply_skill_cluster_labels
    -> create_playstyle_features
    -> merge in skill_group/skill_score, then residualize_playstyle_features
    -> prepare_playstyle_features -> scale_playstyle_features
    -> test_playstyle_k_values -> cluster_playstyles
    -> interpret_playstyle_clusters -> apply_playstyle_archetypes
```

## Scope of this repo

This repo intentionally contains only the pipeline itself -- no
synthetic data generator, no evaluation/validation tooling, no
visualization, no third-party data importers, and no test suite.
Those are development tools used while building this pipeline, kept
in a separate local working project, not part of what ships.

One thing worth flagging if this pipeline is ever extended: any
third-party match data used to validate it during development (e.g.
real match datasets pulled from Kaggle) may carry a non-commercial
license. Never bundle licensed third-party data into this repo or
any commercial build of the app.
