import numpy as np
import pandas as pd


def generate_unlabeled_match_logs(
    num_players=240,
    matches_per_player=10,
    random_state=42
):
    """
    Generate multiple unlabeled match logs per player.

    Each player receives multiple matches so that we can
    later calculate both average performance and consistency.
    """

    rng = np.random.default_rng(
        random_state
    )

    hidden_profiles = [

        # Beginner
        (
            "Beginner Banger",
            0.30,
            12,
            4,
            22,
            14,
            0.10
        ),

        (
            "Beginner Cautious",
            0.42,
            9,
            3,
            14,
            16,
            0.08
        ),

        # Intermediate
        (
            "Intermediate Dinker",
            0.82,
            2,
            8,
            5,
            21,
            0.25
        ),

        (
            "Intermediate Driver",
            0.55,
            9,
            14,
            15,
            20,
            0.20
        ),

        (
            "Intermediate Grinder",
            0.88,
            2,
            7,
            4,
            24,
            0.70
        ),

        # Professional
        (
            "Professional Finisher",
            0.84,
            3,
            21,
            5,
            28,
            0.75
        ),

        (
            "Professional Defender",
            0.89,
            2,
            14,
            2,
            31,
            0.90
        ),

        (
            "Professional Tactician",
            0.94,
            1,
            17,
            2,
            33,
            0.95
        )
    ]

    rows = []

    for player_num in range(
        1,
        num_players + 1
    ):

        profile = hidden_profiles[
            rng.integers(
                0,
                len(hidden_profiles)
            )
        ]

        (
            hidden_name,
            drop_efficiency,
            dink_errors,
            winners,
            unforced_errors,
            duration,
            stacking_probability
        ) = profile

        # ----------------------------------------------------
        # Per-player style tendencies.
        #
        # Drawn independently of the hidden profile so they
        # carry NO structural correlation with skill: a
        # player's drop-vs-drive and net-vs-power tendencies
        # are personal stylistic choices, not something tied
        # to which skill archetype generated their match
        # stats. (Profile-level constants would inherit
        # whatever skill correlation the profile's OTHER
        # parameters have, since a skill tier only contains
        # 2-3 discrete profiles and per-match noise averages
        # out over many matches -- exactly the leakage this
        # pipeline is designed to avoid.)
        # ----------------------------------------------------

        player_drop_preference = rng.uniform(
            0.20,
            0.80
        )

        player_net_game_preference = rng.uniform(
            0.15,
            0.75
        )

        for match_num in range(
            1,
            matches_per_player + 1
        ):

            actual_duration = max(
                8,
                rng.normal(
                    duration,
                    2.5
                )
            )

            actual_winners = max(
                0,
                int(
                    rng.normal(
                        winners,
                        max(
                            1,
                            winners * 0.15
                        )
                    )
                )
            )

            actual_net_game_preference = np.clip(
                rng.normal(
                    player_net_game_preference,
                    0.12
                ),
                0.05,
                0.95
            )

            dink_winners = int(
                round(
                    actual_winners *
                    actual_net_game_preference
                )
            )

            dink_winners = max(
                0,
                min(
                    dink_winners,
                    actual_winners
                )
            )

            clean_winners_count = (
                actual_winners -
                dink_winners
            )

            actual_unforced = max(
                0,
                int(
                    rng.normal(
                        unforced_errors,
                        max(
                            1,
                            unforced_errors * 0.15
                        )
                    )
                )
            )

            actual_dink_errors = max(
                0,
                int(
                    rng.normal(
                        dink_errors,
                        max(
                            1,
                            dink_errors * 0.15
                        )
                    )
                )
            )

            third_shot_opportunities = max(
                2,
                int(
                    rng.normal(
                        18,
                        4
                    )
                )
            )

            actual_drop_preference = np.clip(
                rng.normal(
                    player_drop_preference,
                    0.12
                ),
                0.05,
                0.95
            )

            drop_attempts = int(
                round(
                    third_shot_opportunities *
                    actual_drop_preference
                )
            )

            drop_attempts = max(
                0,
                min(
                    drop_attempts,
                    third_shot_opportunities
                )
            )

            drive_attempts = (
                third_shot_opportunities -
                drop_attempts
            )

            actual_drop_efficiency = np.clip(
                rng.normal(
                    drop_efficiency,
                    0.06
                ),
                0.05,
                0.99
            )

            drop_successes = int(
                round(
                    drop_attempts *
                    actual_drop_efficiency
                )
            )

            drop_successes = min(
                drop_successes,
                drop_attempts
            )

            uses_stacking = int(
                rng.random()
                <
                stacking_probability
            )

            rows.append({

                "player_id":
                    f"PLR_{player_num:03d}",

                "match_id":
                    f"PLR_{player_num:03d}_M{match_num:02d}",

                "match_number":
                    match_num,

                "drop_attempts":
                    drop_attempts,

                "drop_successes":
                    drop_successes,

                "drive_attempts":
                    drive_attempts,

                "dink_errors":
                    actual_dink_errors,

                "clean_winners":
                    clean_winners_count,

                "dink_winners":
                    dink_winners,

                "unforced_errors":
                    actual_unforced,

                "match_duration_mins":
                    actual_duration,

                "uses_stacking":
                    uses_stacking,

                "_hidden_profile":
                    hidden_name
            })

    return pd.DataFrame(rows)