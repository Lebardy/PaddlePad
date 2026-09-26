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
  in disguise), and `extract_playstyle_components`, the PCA step
  before the second K-Means.
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
    -> extract_playstyle_components (PCA, fitted once on everyone)
    -> test_playstyle_k_values / cluster_playstyles on its scores (features=...)
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

## The boil-down step

Before the second K-Means, `extract_playstyle_components` boils the
thirteen playstyle features down with PCA: fitted once on every player,
keeping the fewest components that together explain at least 80% of the
spread. The playstyle K-Means then clusters on those scores, passed in
through the `features` argument. This is the pipeline's unsupervised
feature extraction step. Archetype names are still read from the
thirteen features, which travel alongside the scores.

It was tested against the same pipeline without it, with the pass rule
written before anything ran. On a simulated pool of 50 players with a
hidden net-play habit and a hidden ability, over ten K-Means random
starts, it found the net-play habit about as well (share of the habit's
spread between styles 0.34, against 0.36 without it; run-to-run spread
0.08) and let about as little ability into the styles (0.16 against
0.13; spread 0.08). On 151 real players from the pklmart dataset, over
the same 100 draws of 80% of the players, pairs that shared a group kept
sharing one just as often (0.68 against 0.68). One side effect on the
simulated pool: in 4 of 10 starts the largest skill group split into
five styles, one of them a single player.

Called without `features`, both functions still cluster on the thirteen
features, so these files run on their own.

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
