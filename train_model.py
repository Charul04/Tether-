"""
Tether — Loneliness Detection Engine
Model Training & Prediction Module
Gradient Boosting multi-class classifier + regression score predictor
(sklearn-based, drop-in compatible with XGBoost API)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "msg_response_time_avg","conversation_init_ratio","contact_diversity",
    "social_app_ratio","late_night_activity","weekend_activity_score",
    "vocab_diversity","msg_length_avg","reply_rate","plans_future_ratio",
    "first_person_singular","emoji_variety","checkin_frequency",
    "group_chat_activity","physical_activity_proxy",
    "age","city_size","recently_moved","lives_alone","work_from_home",
]

LONELINESS_LABELS = {0:"Healthy",1:"Situational",2:"Social",3:"Existential",4:"Chronic"}
TYPE_COLORS = {"Healthy":"#a8e63d","Situational":"#f5a623","Social":"#e67e22","Existential":"#8e44ad","Chronic":"#c0392b"}


def load_data(path="data/tether_behavioral_data.csv"):
    df = pd.read_csv(path)
    latest = df.sort_values("week").groupby("user_id").last().reset_index()
    return df, latest


def train_all_models(save_path="models/"):
    print("📊 Loading data...")
    df_full, df_latest = load_data()

    X = df_latest[FEATURE_COLS]
    y_class = df_latest["loneliness_type"]
    y_score = df_latest["social_health_score"]
    y_crisis = df_latest["crisis_risk"]

    X_tr, X_te, yc_tr, yc_te, ys_tr, ys_te, ycr_tr, ycr_te = train_test_split(
        X, y_class, y_score, y_crisis, test_size=0.2, random_state=42, stratify=y_class
    )

    os.makedirs(save_path, exist_ok=True)

    print("🌲 Training loneliness type classifier...")
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.08,
                                      subsample=0.8, random_state=42)
    clf.fit(X_tr, yc_tr)
    yc_pred = clf.predict(X_te)
    print(f"   Accuracy: {accuracy_score(yc_te, yc_pred):.3f} | F1: {f1_score(yc_te, yc_pred, average='weighted'):.3f}")
    print(classification_report(yc_te, yc_pred, target_names=list(LONELINESS_LABELS.values())))

    print("📈 Training social health score regressor...")
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08,
                                     subsample=0.8, random_state=42)
    reg.fit(X_tr, ys_tr)
    ys_pred = reg.predict(X_te)
    print(f"   MAE: {mean_absolute_error(ys_te, ys_pred):.2f} points")

    print("🚨 Training crisis risk predictor...")
    crisis = RandomForestClassifier(n_estimators=150, max_depth=6,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    crisis.fit(X_tr, ycr_tr)
    ycr_pred = crisis.predict(X_te)
    print(f"   Crisis F1: {f1_score(ycr_te, ycr_pred, zero_division=0):.3f}")

    joblib.dump(clf,    f"{save_path}/classifier.pkl")
    joblib.dump(reg,    f"{save_path}/regressor.pkl")
    joblib.dump(crisis, f"{save_path}/crisis_predictor.pkl")
    joblib.dump(FEATURE_COLS, f"{save_path}/feature_cols.pkl")
    print(f"\n✅ All models saved to {save_path}/")

    return clf, reg, crisis


class TetherPredictor:
    def __init__(self, model_path="models/"):
        self.clf    = joblib.load(f"{model_path}/classifier.pkl")
        self.reg    = joblib.load(f"{model_path}/regressor.pkl")
        self.crisis = joblib.load(f"{model_path}/crisis_predictor.pkl")
        self.feature_cols = joblib.load(f"{model_path}/feature_cols.pkl")

    def predict(self, features: dict) -> dict:
        df = pd.DataFrame([features])[self.feature_cols]
        loneliness_type  = int(self.clf.predict(df)[0])
        type_probs       = self.clf.predict_proba(df)[0]
        health_score     = float(np.clip(self.reg.predict(df)[0], 0, 100))
        crisis_prob      = float(self.crisis.predict_proba(df)[0][1])

        return {
            "loneliness_type":  loneliness_type,
            "loneliness_label": LONELINESS_LABELS[loneliness_type],
            "type_probabilities": {LONELINESS_LABELS[i]: round(float(p),3) for i,p in enumerate(type_probs)},
            "health_score":     round(health_score, 1),
            "crisis_probability": round(crisis_prob, 3),
            "crisis_risk_level": self._crisis_level(crisis_prob),
            "fingerprint":      self._fingerprint(features),
            "interventions":    self._interventions(loneliness_type),
            "drift_status":     self._drift(features),
            "key_signals":      self._signals(features),
        }

    def _fingerprint(self, f):
        return {
            "Social Isolation":     round(max(0,min(100, 100 - f.get("contact_diversity",10)*4 - f.get("group_chat_activity",0.5)*20)),1),
            "Emotional Distance":   round(max(0,min(100, f.get("msg_response_time_avg",60)/3 + (1-f.get("reply_rate",0.8))*40)),1),
            "Existential Void":     round(max(0,min(100, (1-f.get("plans_future_ratio",0.2))*60 + f.get("first_person_singular",0.2)*80)),1),
            "Behavioral Withdrawal":round(max(0,min(100, (1-f.get("weekend_activity_score",0.7))*50 + f.get("late_night_activity",0.1)*120)),1),
            "Linguistic Decay":     round(max(0,min(100, (1-f.get("vocab_diversity",0.7))*70 + (1-f.get("plans_future_ratio",0.2))*30)),1),
        }

    def _interventions(self, lt):
        all_inv = {
            0:[{"action":"Maintain connection rituals","priority":"low","detail":"Schedule one meaningful conversation per week"},
               {"action":"Expand weak ties","priority":"low","detail":"Attend one new community event this month"}],
            1:[{"action":"Reconnect with one old friend","priority":"high","detail":"Reach out to someone you haven't spoken to in 3+ weeks"},
               {"action":"Join a local group","priority":"medium","detail":"Find one recurring meetup matching your interests"},
               {"action":"Establish a daily anchor","priority":"medium","detail":"One fixed social interaction per day, even brief"}],
            2:[{"action":"Depth over breadth","priority":"high","detail":"Invest in 2-3 relationships rather than many surface ones"},
               {"action":"Shared activity connection","priority":"high","detail":"Join a class or club where connection happens through doing"},
               {"action":"Reduce solo consumption","priority":"medium","detail":"Replace one solo streaming session with a shared one"}],
            3:[{"action":"Authentic expression","priority":"high","detail":"Share something real with one person this week, not small talk"},
               {"action":"Find your tribe","priority":"high","detail":"Seek communities centered on values not just activities"},
               {"action":"Reduce performance","priority":"medium","detail":"Practice one conversation where you drop the social mask"}],
            4:[{"action":"Professional support","priority":"critical","detail":"Connect with a mental health professional this week"},
               {"action":"One micro-connection daily","priority":"high","detail":"A smile, a thank you, a 2-minute chat — start impossibly small"},
               {"action":"Crisis line available","priority":"high","detail":"iCall India: 9152987821 | Mon-Sat 8am-10pm"}],
        }
        return all_inv.get(lt, all_inv[1])

    def _crisis_level(self, p):
        if p < 0.15: return "Low"
        elif p < 0.35: return "Moderate"
        elif p < 0.60: return "High"
        else: return "Critical"

    def _drift(self, f):
        avg = np.mean([f.get("conversation_init_ratio",0.5),
                       f.get("contact_diversity",10)/20,
                       f.get("reply_rate",0.8),
                       1-f.get("late_night_activity",0.1),
                       f.get("weekend_activity_score",0.7)])
        if avg > 0.65: return {"status":"Stable","direction":"→","color":"#a8e63d"}
        elif avg > 0.45: return {"status":"Drifting","direction":"↘","color":"#f5a623"}
        else: return {"status":"Declining","direction":"↓","color":"#c0392b"}

    def _signals(self, f):
        s = []
        if f.get("late_night_activity",0) > 0.25:
            s.append({"signal":"Late night activity spike","severity":"medium","detail":f"{f['late_night_activity']*100:.0f}% activity after 1am"})
        if f.get("contact_diversity",10) < 6:
            s.append({"signal":"Shrinking social circle","severity":"high","detail":f"Only {f['contact_diversity']:.0f} unique contacts this week"})
        if f.get("conversation_init_ratio",0.5) < 0.25:
            s.append({"signal":"Stopped reaching out","severity":"high","detail":"Rarely initiating conversations anymore"})
        if f.get("plans_future_ratio",0.2) < 0.08:
            s.append({"signal":"Future thinking collapsed","severity":"medium","detail":"Almost no future-oriented language detected"})
        if f.get("weekend_activity_score",0.7) < 0.35:
            s.append({"signal":"Weekend isolation pattern","severity":"high","detail":"Activity collapses on weekends"})
        return s[:4]


if __name__ == "__main__":
    clf, reg, crisis = train_all_models()

    print("\n🧪 Test Prediction:")
    p = TetherPredictor()
    r = p.predict({
        "msg_response_time_avg":180,"conversation_init_ratio":0.18,
        "contact_diversity":3,"social_app_ratio":0.12,
        "late_night_activity":0.35,"weekend_activity_score":0.28,
        "vocab_diversity":0.52,"msg_length_avg":12,"reply_rate":0.45,
        "plans_future_ratio":0.06,"first_person_singular":0.40,
        "emoji_variety":1.2,"checkin_frequency":0.8,
        "group_chat_activity":0.08,"physical_activity_proxy":0.25,
        "age":24,"city_size":2,"recently_moved":1,"lives_alone":1,"work_from_home":1,
    })
    print(f"  Type: {r['loneliness_label']}")
    print(f"  Score: {r['health_score']}/100")
    print(f"  Crisis: {r['crisis_risk_level']} ({r['crisis_probability']:.1%})")
    print(f"  Drift: {r['drift_status']['status']} {r['drift_status']['direction']}")
