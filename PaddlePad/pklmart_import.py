# ============================================================
# PaddlePad ML - pklmart Real-Data Import
# ============================================================
#
# Converts the pklmart Kaggle dataset (game/rally/shot/team CSVs)
# into the same per-player-per-match row shape produced by
# data_generator.generate_unlabeled_match_logs(), so real match
# data can be run through the existing pipeline unchanged.
#
# Source: https://www.kaggle.com/datasets/cakesofspan/pklmarts-competitive-pickleball-extracts
# License: CC BY-NC-SA 4.0 (non-commercial use only, share-alike).
# The raw CSVs are intentionally NOT bundled into this project --
# point `data_dir` at wherever you downloaded and extracted them.
#
# One output row = one player's stats in one pklmart GAME (a
# single game to 11, not a whole best-of-N match). A pklmart
# "game" is the closest match in scale/duration to what our own
# pipeline calls a "match".
#
# KNOWN APPROXIMATIONS (real data doesn't map perfectly onto our
# umpire-tappable schema):
#
#   - match_duration_mins is ESTIMATED from shot/rally counts.
#     pklmart does not record real timestamps in this export, so
#     per-minute rate features derived from it are approximate,
#     not measured.
#
#   - pklmart distinguishes "Error" (forced) from "Unforced
#     Error". Our schema only tracks one error bucket, so both
#     are combined into unforced_errors / dink_errors.
#
#   - uses_stacking is not derivable from this export and is set
#     to 0 for every row. This is harmless: no ML feature in this
#     pipeline uses stacking_rate (see feature_engineering.py).
# ============================================================

import pandas as pd


DINK_SHOT_TYPE = "D"
THIRD_SHOT_DROP = "tsDrp"
THIRD_SHOT_DRIVE = "tsDrv"

ERROR_ENDING_TYPES = ["Error", "Unforced Error"]
WINNER_ENDING_TYPE = "Winner"

# Rough estimate used only to approximate match_duration_mins,
# since pklmart does not record real game timestamps.
SECONDS_PER_SHOT_ESTIMATE = 2.5


# ============================================================
# LOAD RAW TABLES
# ============================================================

def _load_raw_tables(data_dir):

    games = pd.read_csv(
        f"{data_dir}/game.csv"
    )

    teams = pd.read_csv(
        f"{data_dir}/team.csv"
    )

    rallies = pd.read_csv(
        f"{data_dir}/rally.csv"
    )

    shots = pd.read_csv(
        f"{data_dir}/shot.csv"
    )

    return games, teams, rallies, shots


# ============================================================
# TAG EACH RALLY WITH ITS ENDING SHOT TYPE
# ============================================================

def _attach_ending_shot_type(rallies, shots):
    """
    Every rally ends on one shot. We need that shot's type to
    know whether a winner/error happened at the net (Dink) or
    not, matching our own dink_errors/dink_winners split.
    """

    ending_shots = (
        shots
        .sort_values("shot_nbr")
        .groupby("rally_id")
        .tail(1)
        [
            [
                "rally_id",
                "shot_type"
            ]
        ]
        .rename(
            columns={
                "shot_type": "ending_shot_type"
            }
        )
    )

    return rallies.merge(
        ending_shots,
        on="rally_id",
        how="left"
    )


# ============================================================
# COUNT WINNERS / ERRORS, SPLIT BY DINK VS NOT
# ============================================================

def _count_rally_outcomes(rallies):

    is_error = rallies["ending_type"].isin(ERROR_ENDING_TYPES)
    is_winner = rallies["ending_type"] == WINNER_ENDING_TYPE
    is_dink = rallies["ending_shot_type"] == DINK_SHOT_TYPE
    has_player = rallies["ending_player_id"].notna()

    def count_by(mask, column_name):

        subset = rallies[mask]

        return (
            subset
            .groupby(
                [
                    "game_id",
                    "ending_player_id"
                ]
            )
            .size()
            .reset_index(
                name=column_name
            )
            .rename(
                columns={
                    "ending_player_id": "player_id"
                }
            )
        )

    dink_errors_ct = count_by(
        is_error & is_dink & has_player,
        "dink_errors"
    )

    unforced_errors_ct = count_by(
        is_error & ~is_dink & has_player,
        "unforced_errors"
    )

    dink_winners_ct = count_by(
        is_winner & is_dink & has_player,
        "dink_winners"
    )

    clean_winners_ct = count_by(
        is_winner & ~is_dink & has_player,
        "clean_winners"
    )

    return (
        dink_errors_ct,
        unforced_errors_ct,
        dink_winners_ct,
        clean_winners_ct
    )


# ============================================================
# COUNT DROP / DRIVE ATTEMPTS AND DROP SUCCESS
# ============================================================

def _count_drop_and_drive(shots, rallies):

    shots_with_game = shots.merge(
        rallies[
            [
                "rally_id",
                "game_id"
            ]
        ],
        on="rally_id",
        how="left"
    )

    max_shot_nbr = (
        shots_with_game
        .groupby("rally_id")["shot_nbr"]
        .transform("max")
    )

    shots_with_game["is_final_shot"] = (
        shots_with_game["shot_nbr"] == max_shot_nbr
    )

    drop_shots = shots_with_game[
        shots_with_game["shot_type"] == THIRD_SHOT_DROP
    ].copy()

    drive_shots = shots_with_game[
        shots_with_game["shot_type"] == THIRD_SHOT_DRIVE
    ]

    drop_attempts_ct = (
        drop_shots
        .groupby(
            [
                "game_id",
                "player_id"
            ]
        )
        .size()
        .reset_index(
            name="drop_attempts"
        )
    )

    drive_attempts_ct = (
        drive_shots
        .groupby(
            [
                "game_id",
                "player_id"
            ]
        )
        .size()
        .reset_index(
            name="drive_attempts"
        )
    )

    # A drop attempt "fails" only if it is the LAST shot of its
    # rally AND that rally ended in an error attributed to the
    # SAME player who hit the drop. Every other drop attempt
    # (rally continued, or ended in a winner) counts as
    # successful.
    drop_shots = drop_shots.merge(
        rallies[
            [
                "rally_id",
                "ending_type",
                "ending_player_id"
            ]
        ],
        on="rally_id",
        how="left"
    )

    drop_failed = (
        drop_shots["is_final_shot"]
        & drop_shots["ending_type"].isin(ERROR_ENDING_TYPES)
        & (
            drop_shots["ending_player_id"]
            ==
            drop_shots["player_id"]
        )
    )

    drop_failures_ct = (
        drop_shots[drop_failed]
        .groupby(
            [
                "game_id",
                "player_id"
            ]
        )
        .size()
        .reset_index(
            name="drop_failures"
        )
    )

    return (
        drop_attempts_ct,
        drive_attempts_ct,
        drop_failures_ct
    )


# ============================================================
# ESTIMATE MATCH DURATION FROM SHOT VOLUME
# ============================================================

def _estimate_game_duration(shots, rallies):

    shots_with_game = shots.merge(
        rallies[
            [
                "rally_id",
                "game_id"
            ]
        ],
        on="rally_id",
        how="left"
    )

    shot_counts = (
        shots_with_game
        .groupby("game_id")
        .size()
        .reset_index(
            name="total_shots"
        )
    )

    shot_counts["match_duration_mins"] = (
        shot_counts["total_shots"]
        * SECONDS_PER_SHOT_ESTIMATE
        / 60.0
    )

    return shot_counts[
        [
            "game_id",
            "match_duration_mins"
        ]
    ]


# ============================================================
# BUILD THE PER-GAME PLAYER ROSTER
# ============================================================

def _build_roster(games, teams):
    """
    One row per (game_id, player_id) for every player who
    appeared in that game, carrying skill_lvl along for
    optional external validation later (see
    evaluation.build_hidden_player_profiles for the analogous
    pattern used with our own synthetic data).

    pklmart's own `match_id` (a best-of-N event spanning
    multiple games) is intentionally dropped here: our pipeline
    treats each GAME as one "match" row, and carrying pklmart's
    match_id through would collide with the game_id -> match_id
    rename done later in load_pklmart_as_match_logs.
    """

    game_teams = games[
        [
            "game_id",
            "skill_lvl",
            "w_team_id",
            "l_team_id"
        ]
    ]

    winning_side = game_teams.rename(
        columns={
            "w_team_id": "team_id"
        }
    )[
        [
            "game_id",
            "skill_lvl",
            "team_id"
        ]
    ]

    losing_side = game_teams.rename(
        columns={
            "l_team_id": "team_id"
        }
    )[
        [
            "game_id",
            "skill_lvl",
            "team_id"
        ]
    ]

    long_form = pd.concat(
        [
            winning_side,
            losing_side
        ],
        ignore_index=True
    )

    roster = long_form.merge(
        teams[
            [
                "team_id",
                "player_id"
            ]
        ],
        on="team_id",
        how="left"
    )

    return roster[
        [
            "game_id",
            "skill_lvl",
            "player_id"
        ]
    ]


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def load_pklmart_as_match_logs(data_dir):
    """
    Read the extracted pklmart CSVs from `data_dir` and return a
    DataFrame with the same columns as
    data_generator.generate_unlabeled_match_logs():

        player_id, match_id, match_number, drop_attempts,
        drop_successes, drive_attempts, dink_errors,
        clean_winners, dink_winners, unforced_errors,
        match_duration_mins, uses_stacking, _hidden_skill_lvl

    `_hidden_skill_lvl` mirrors the `_hidden_profile` column in
    our synthetic generator: it must be dropped before the data
    is fed into the ML pipeline, and exists only for external
    validation (comparing discovered skill clusters against
    pklmart's real skill_lvl tags).
    """

    games, teams, rallies, shots = _load_raw_tables(
        data_dir
    )

    rallies = _attach_ending_shot_type(
        rallies,
        shots
    )

    (
        dink_errors_ct,
        unforced_errors_ct,
        dink_winners_ct,
        clean_winners_ct
    ) = _count_rally_outcomes(
        rallies
    )

    (
        drop_attempts_ct,
        drive_attempts_ct,
        drop_failures_ct
    ) = _count_drop_and_drive(
        shots,
        rallies
    )

    duration_by_game = _estimate_game_duration(
        shots,
        rallies
    )

    roster = _build_roster(
        games,
        teams
    )

    # --------------------------------------------------------
    # A small number of team_id references in games.csv don't
    # resolve to a player in team.csv (source data quality
    # gap, not something we can recover). Drop those rows
    # rather than silently keeping a NaN player_id.
    # --------------------------------------------------------

    unresolved = roster["player_id"].isna().sum()

    if unresolved:

        print(
            f"pklmart_import: dropping {unresolved} roster "
            f"row(s) with no resolvable player_id"
        )

    roster = roster[
        roster["player_id"].notna()
    ]

    # --------------------------------------------------------
    # Merge every count table onto the roster. Players who
    # didn't record a given event in a game get 0, not NaN.
    # --------------------------------------------------------

    result = roster.copy()

    for table, column in [
        (dink_errors_ct, "dink_errors"),
        (unforced_errors_ct, "unforced_errors"),
        (dink_winners_ct, "dink_winners"),
        (clean_winners_ct, "clean_winners"),
        (drop_attempts_ct, "drop_attempts"),
        (drive_attempts_ct, "drive_attempts"),
        (drop_failures_ct, "drop_failures")
    ]:

        result = result.merge(
            table,
            on=[
                "game_id",
                "player_id"
            ],
            how="left"
        )

        result[column] = (
            result[column]
            .fillna(0)
            .astype(int)
        )

    result["drop_successes"] = (
        result["drop_attempts"]
        -
        result["drop_failures"]
    )

    result = result.drop(
        columns=[
            "drop_failures"
        ]
    )

    result = result.merge(
        duration_by_game,
        on="game_id",
        how="left"
    )

    # --------------------------------------------------------
    # Final column shape, matching data_generator's output
    # --------------------------------------------------------

    result["match_number"] = (
        result
        .groupby("player_id")
        .cumcount()
        + 1
    ).astype(int)

    result["uses_stacking"] = 0

    result = result.rename(
        columns={
            "game_id": "match_id",
            "skill_lvl": "_hidden_skill_lvl"
        }
    )

    output_columns = [
        "player_id",
        "match_id",
        "match_number",
        "drop_attempts",
        "drop_successes",
        "drive_attempts",
        "dink_errors",
        "clean_winners",
        "dink_winners",
        "unforced_errors",
        "match_duration_mins",
        "uses_stacking",
        "_hidden_skill_lvl"
    ]

    return result[output_columns]
