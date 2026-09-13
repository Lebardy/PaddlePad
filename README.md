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
    -> merge in rally_points, then interpret_skill_clusters(rank_by="rally_points")
    -> apply_skill_cluster_labels
    -> create_playstyle_features
    -> merge in skill_group/skill_score, then residualize_playstyle_features
    -> prepare_playstyle_features -> scale_playstyle_features
    -> test_playstyle_k_values -> cluster_playstyles
    -> interpret_playstyle_clusters -> apply_playstyle_archetypes
```

## Which number does what

Two numbers describe a player's skill, and they have separate jobs.

- **K-Means sees neither.** The skill groups are found from the
  behavioural features alone, so who is grouped with whom never depends
  on a score.
- **`rally_points` names the skill groups.** It is the app's rally
  rating: every rally moves a player's points by how it ended and who
  ended it, Elo-style. The group whose players average the most points
  is named the higher group. It comes from the app, not from these
  files, and is merged in by player id before
  `interpret_skill_clusters`.
- **`skill_score` takes skill out of the playstyle features.**
  `residualize_playstyle_features` still removes its effect before the
  second K-Means, exactly as before.

Why the names moved to rally points: on a synthetic pool of 50 players
with a hidden true ability, group numbers ranked by rally points
followed that ability closely (Spearman 0.81 over ten K-Means random
starts); ranked by `skill_score` they barely did (0.17). Nobody changed
group, and playstyles were unaffected. That is one synthetic pool, so
it should be checked again on real matches.

Called without `rank_by`, `interpret_skill_clusters` still ranks by
`skill_score`, so these files run on their own.

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
