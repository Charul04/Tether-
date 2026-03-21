"""
Tether — Loneliness Detection Engine
Data Generation Script
Generates realistic synthetic behavioral data for model training
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
N_USERS = 2000
N_WEEKS = 26  # 6 months of data per user

LONELINESS_TYPES = {
    0: "Healthy",
    1: "Situational",   # new city, breakup, job change
    2: "Social",        # no friends, surface relationships
    3: "Existential",   # surrounded by people but unseen
    4: "Chronic"        # severe, long-term isolation
}


# ─────────────────────────────────────────────
# FEATURE GENERATORS
# ─────────────────────────────────────────────

def generate_user_profile(user_id, loneliness_type):
    """Generate base behavioral profile per loneliness type"""

    profiles = {
        0: {  # Healthy
            "msg_response_time_avg": np.random.normal(45, 15),       # minutes
            "conversation_init_ratio": np.random.normal(0.52, 0.1),   # 0-1
            "contact_diversity": np.random.normal(18, 4),             # unique contacts/week
            "social_app_ratio": np.random.normal(0.45, 0.08),         # social vs solo apps
            "late_night_activity": np.random.normal(0.08, 0.03),      # % activity 1am-4am
            "weekend_activity_score": np.random.normal(0.72, 0.1),    # relative to weekday
            "vocab_diversity": np.random.normal(0.78, 0.06),          # TTR score
            "msg_length_avg": np.random.normal(42, 12),               # words
            "reply_rate": np.random.normal(0.88, 0.06),
            "plans_future_ratio": np.random.normal(0.31, 0.06),       # future tense in text
            "first_person_singular": np.random.normal(0.18, 0.04),    # I/me ratio
            "emoji_variety": np.random.normal(8.2, 2.0),
            "checkin_frequency": np.random.normal(6.1, 1.5),          # times/week
            "group_chat_activity": np.random.normal(0.62, 0.1),
            "physical_activity_proxy": np.random.normal(0.71, 0.12),
        },
        1: {  # Situational
            "msg_response_time_avg": np.random.normal(85, 25),
            "conversation_init_ratio": np.random.normal(0.35, 0.1),
            "contact_diversity": np.random.normal(9, 3),
            "social_app_ratio": np.random.normal(0.28, 0.08),
            "late_night_activity": np.random.normal(0.18, 0.05),
            "weekend_activity_score": np.random.normal(0.48, 0.12),
            "vocab_diversity": np.random.normal(0.65, 0.07),
            "msg_length_avg": np.random.normal(28, 10),
            "reply_rate": np.random.normal(0.71, 0.08),
            "plans_future_ratio": np.random.normal(0.18, 0.05),
            "first_person_singular": np.random.normal(0.28, 0.05),
            "emoji_variety": np.random.normal(4.1, 1.5),
            "checkin_frequency": np.random.normal(3.2, 1.2),
            "group_chat_activity": np.random.normal(0.35, 0.1),
            "physical_activity_proxy": np.random.normal(0.48, 0.12),
        },
        2: {  # Social
            "msg_response_time_avg": np.random.normal(120, 35),
            "conversation_init_ratio": np.random.normal(0.22, 0.08),
            "contact_diversity": np.random.normal(5, 2),
            "social_app_ratio": np.random.normal(0.18, 0.06),
            "late_night_activity": np.random.normal(0.24, 0.07),
            "weekend_activity_score": np.random.normal(0.38, 0.1),
            "vocab_diversity": np.random.normal(0.58, 0.08),
            "msg_length_avg": np.random.normal(18, 8),
            "reply_rate": np.random.normal(0.58, 0.1),
            "plans_future_ratio": np.random.normal(0.12, 0.04),
            "first_person_singular": np.random.normal(0.34, 0.06),
            "emoji_variety": np.random.normal(2.4, 1.0),
            "checkin_frequency": np.random.normal(1.8, 0.9),
            "group_chat_activity": np.random.normal(0.18, 0.07),
            "physical_activity_proxy": np.random.normal(0.38, 0.1),
        },
        3: {  # Existential
            "msg_response_time_avg": np.random.normal(95, 30),
            "conversation_init_ratio": np.random.normal(0.41, 0.1),
            "contact_diversity": np.random.normal(12, 4),
            "social_app_ratio": np.random.normal(0.38, 0.08),
            "late_night_activity": np.random.normal(0.29, 0.08),
            "weekend_activity_score": np.random.normal(0.55, 0.12),
            "vocab_diversity": np.random.normal(0.71, 0.06),
            "msg_length_avg": np.random.normal(55, 15),
            "reply_rate": np.random.normal(0.79, 0.07),
            "plans_future_ratio": np.random.normal(0.09, 0.03),
            "first_person_singular": np.random.normal(0.31, 0.05),
            "emoji_variety": np.random.normal(3.2, 1.2),
            "checkin_frequency": np.random.normal(4.8, 1.4),
            "group_chat_activity": np.random.normal(0.51, 0.1),
            "physical_activity_proxy": np.random.normal(0.55, 0.12),
        },
        4: {  # Chronic
            "msg_response_time_avg": np.random.normal(240, 60),
            "conversation_init_ratio": np.random.normal(0.12, 0.06),
            "contact_diversity": np.random.normal(2, 1),
            "social_app_ratio": np.random.normal(0.08, 0.04),
            "late_night_activity": np.random.normal(0.41, 0.1),
            "weekend_activity_score": np.random.normal(0.22, 0.08),
            "vocab_diversity": np.random.normal(0.48, 0.08),
            "msg_length_avg": np.random.normal(8, 4),
            "reply_rate": np.random.normal(0.38, 0.1),
            "plans_future_ratio": np.random.normal(0.05, 0.02),
            "first_person_singular": np.random.normal(0.42, 0.07),
            "emoji_variety": np.random.normal(0.8, 0.5),
            "checkin_frequency": np.random.normal(0.4, 0.3),
            "group_chat_activity": np.random.normal(0.06, 0.04),
            "physical_activity_proxy": np.random.normal(0.21, 0.08),
        }
    }
    return profiles[loneliness_type]


def add_temporal_drift(value, week, loneliness_type, feature_name, noise_scale=0.05):
    """Add realistic temporal drift patterns"""
    # Healthy users are stable
    if loneliness_type == 0:
        drift = np.random.normal(0, noise_scale * abs(value))
        return max(0, value + drift)

    # Others drift progressively worse over time
    drift_rate = {1: 0.008, 2: 0.015, 3: 0.010, 4: 0.020}[loneliness_type]

    # Negative features get worse over time
    negative_features = ["msg_response_time_avg", "late_night_activity", "first_person_singular"]
    direction = 1 if feature_name in negative_features else -1

    temporal_drift = direction * drift_rate * week * abs(value)
    noise = np.random.normal(0, noise_scale * abs(value))

    return max(0, value + temporal_drift + noise)


def generate_dataset():
    """Generate full training dataset"""
    records = []

    for user_id in range(N_USERS):
        # Assign loneliness type with realistic distribution
        loneliness_type = np.random.choice(
            [0, 1, 2, 3, 4],
            p=[0.35, 0.25, 0.18, 0.14, 0.08]
        )

        profile = generate_user_profile(user_id, loneliness_type)

        # Demographic features
        age = np.random.randint(18, 65)
        city_size = np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])  # small/medium/large
        recently_moved = np.random.choice([0, 1], p=[0.8, 0.2])
        lives_alone = np.random.choice([0, 1], p=[0.6, 0.4])
        work_from_home = np.random.choice([0, 1], p=[0.55, 0.45])

        for week in range(N_WEEKS):
            record = {
                "user_id": user_id,
                "week": week,
                "loneliness_type": loneliness_type,
                "loneliness_label": LONELINESS_TYPES[loneliness_type],
                "age": age,
                "city_size": city_size,
                "recently_moved": recently_moved,
                "lives_alone": lives_alone,
                "work_from_home": work_from_home,
            }

            # Add behavioral features with temporal drift
            for feature, base_value in profile.items():
                record[feature] = round(
                    add_temporal_drift(base_value, week, loneliness_type, feature), 4
                )

            # Derived features
            record["social_health_score"] = compute_health_score(record)
            record["drift_velocity"] = compute_drift_velocity(record, week)
            record["crisis_risk"] = 1 if (
                record["social_health_score"] < 30 and
                loneliness_type >= 3
            ) else 0

            records.append(record)

    df = pd.DataFrame(records)
    return df


def compute_health_score(record):
    """Compute 0-100 social health score from behavioral features"""
    score = 100

    # Penalize each risk signal
    if record["msg_response_time_avg"] > 120: score -= 12
    elif record["msg_response_time_avg"] > 60: score -= 6

    if record["conversation_init_ratio"] < 0.25: score -= 15
    elif record["conversation_init_ratio"] < 0.4: score -= 7

    if record["contact_diversity"] < 5: score -= 18
    elif record["contact_diversity"] < 10: score -= 8

    if record["social_app_ratio"] < 0.15: score -= 12
    elif record["social_app_ratio"] < 0.3: score -= 5

    if record["late_night_activity"] > 0.3: score -= 14
    elif record["late_night_activity"] > 0.18: score -= 6

    if record["weekend_activity_score"] < 0.3: score -= 10
    elif record["weekend_activity_score"] < 0.5: score -= 4

    if record["vocab_diversity"] < 0.5: score -= 8
    elif record["vocab_diversity"] < 0.65: score -= 3

    if record["reply_rate"] < 0.5: score -= 8
    elif record["reply_rate"] < 0.7: score -= 3

    if record["plans_future_ratio"] < 0.08: score -= 6
    if record["first_person_singular"] > 0.38: score -= 7
    if record["emoji_variety"] < 1.5: score -= 4
    if record["checkin_frequency"] < 1.0: score -= 8
    if record["group_chat_activity"] < 0.1: score -= 5
    if record["physical_activity_proxy"] < 0.3: score -= 6

    return max(0, min(100, score + np.random.normal(0, 3)))


def compute_drift_velocity(record, week):
    """Estimate how fast the score is declining"""
    if week == 0:
        return 0.0
    # Proxy: higher for declining loneliness types over time
    base_velocity = {0: 0.0, 1: -0.8, 2: -1.4, 3: -1.0, 4: -2.1}
    lt = record["loneliness_type"]
    return round(base_velocity[lt] + np.random.normal(0, 0.3), 3)


if __name__ == "__main__":
    print("🌱 Generating Tether training dataset...")
    df = generate_dataset()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/tether_behavioral_data.csv", index=False)

    print(f"✅ Dataset generated: {len(df):,} records | {df['user_id'].nunique()} users")
    print(f"\nLoneliness type distribution:")
    dist = df.groupby("loneliness_label")["user_id"].nunique()
    for label, count in dist.items():
        pct = count / df['user_id'].nunique() * 100
        print(f"  {label:15s}: {count:4d} users ({pct:.1f}%)")
    print(f"\nFeatures: {len(df.columns)} columns")
    print(df.describe().round(2))
