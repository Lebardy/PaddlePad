# PaddlePad ML - app_core

This folder is the complete, self-contained ML pipeline for PaddlePad,
meant to be lifted directly into the app's backend. It contains only
what a running app needs to turn raw umpire-logged match data into a
skill score and playstyle archetype for each player.

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

These four files have **zero dependencies on each other or on
anything else in this repo** (confirmed via import check + a
standalone smoke test) -- only on `numpy`, `pandas`, and
`scikit-learn`. Copy this folder anywhere and it works.

## Call order

Mirrors `main.py` in the parent folder, minus all the
`print_*` diagnostic calls (those are dev-only and safe to skip in
the app):

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

## What's deliberately NOT here, and why

- `data_generator.py` - only generates fake match data for testing.
  The app has real umpire-logged data instead.
- `evaluation.py` - compares discovered clusters against known
  ground truth. Only meaningful with synthetic data; a live app has
  no ground truth to compare against.
- `visualization.py` - matplotlib plots for developer analysis, not
  something a backend service calls.
- `pklmart_import.py` and any pklmart data - used only to validate
  this pipeline against real third-party match data during
  development. **Do not bundle pklmart data with the app under any
  circumstances** - it's licensed CC BY-NC-SA (non-commercial,
  share-alike); shipping it inside a commercial app would violate
  that license.
- `main.py`, `test_*.py` - the manual demo script and the
  automated test suite. Development tools, not app code.

## Keeping this folder in sync

This is a **copy**, not a symlink or a package reference. If
`player_profiles.py`, `skill_model.py`, `feature_engineering.py`, or
`clustering.py` change in the parent folder, this folder will NOT
update automatically -- re-copy the four files (or diff them) before
shipping a new version to the app.
