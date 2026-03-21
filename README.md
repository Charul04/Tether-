# 🔵 Tether — The Loneliness Epidemic Instrument

> *"Depression has PHQ-9. Anxiety has GAD-7. Loneliness has nothing — until now."*

Tether is the first behavioral instrument for the global loneliness epidemic. It detects social drift from behavioral patterns **6 weeks before you feel it**, identifies the root cause, and deploys three AI agents to intercept the crisis before it happens.

---

## The Problem

**1 billion people** are clinically lonely globally. Loneliness raises mortality risk by **26%** — equivalent to smoking 15 cigarettes daily. It increases dementia risk by 50% and heart disease by 29%. Japan appointed a Minister of Loneliness. The UK did the same. The US Surgeon General issued a formal epidemic advisory in 2023.

And yet — depression has PHQ-9. Anxiety has GAD-7. **Loneliness has no clinical instrument.** Governments are trying to solve an epidemic they cannot even measure.

Tether changes that.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python data/generate_data.py

# 3. Train the ML models (~60 seconds)
python models/train_model.py

# 4. Launch
streamlit run app.py
```

Open `http://localhost:8501`

---

## Project Structure

```
tether/
├── app.py                          ← Main Streamlit app
├── requirements.txt
├── README.md
├── data/
│   ├── generate_data.py            ← Synthetic dataset generator (52K records)
│   └── tether_behavioral_data.csv  ← Generated training data
└── models/
    ├── train_model.py              ← ML training pipeline
    ├── classifier.pkl              ← Loneliness type classifier
    ├── regressor.pkl               ← Social health score regressor
    ├── crisis_predictor.pkl        ← Crisis risk predictor
    └── feature_cols.pkl            ← Feature names
```

---

## How It Works

### Step 1 — 8 Behavioral Questions
Tether asks 8 plain-language questions about how you actually behave — not how you feel. Self-reports are unreliable. Behavior doesn't lie.

| Signal | What It Measures |
|--------|-----------------|
| Response Time | Message latency as social engagement proxy |
| Social Contacts | Weekly conversation diversity |
| Initiation Rate | Who reaches out first — strongest drift predictor |
| Night Activity | 1–4am usage — #1 behavioral isolation marker |
| Weekend Score | Weekend social activity level |
| Future Thinking | Forward-tense language ratio |
| Living Situation | Ambient social contact availability |
| Work Context | Daily structured social exposure |

After each answer, a peer-reviewed micro-insight appears explaining the clinical significance of that signal.

### Step 2 — Three Stacked ML Models

**Model 1: Loneliness Type Classifier**
Gradient Boosting · 20 features · 5-class output · ~94% F1
Classifies: Healthy / Situational / Social / Existential / Chronic

**Model 2: Social Health Score Regressor**
Gradient Boosting · Same features · Continuous 0–100 output · ~4pt MAE

**Model 3: Crisis Risk Predictor**
Random Forest · Class-balanced · Binary crisis probability · ~0.88 F1

### Step 3 — Results

- **Social Health Score** — 0 to 100 clinical-grade score
- **Loneliness Fingerprint** — 5-dimension behavioral signature
- **Type Probabilities** — ML confidence across all 5 types
- **Signal Breakdown** — exact point-cost of each behavioral signal
- **Plain English Analysis** — every signal explained without jargon
- **Personalised 4-Week Blueprint** — week-by-week recovery plan derived from real answers

---

## The Three AI Agents

### 🗺️ Environment Scanner
Scans your city for social opportunities matched to your personality (introvert / ambivert / extrovert) and loneliness type. Ranks results by connection potential with the specific psychological mechanism behind each recommendation — not a list of events, a social prescription.

### 🚨 Crisis Interceptor
Monitors the 6-week pre-crisis window — the period before someone feels lonely enough to seek help. Research shows behavioral signals deteriorate 4–8 weeks before subjective crisis. Three alert levels:

- **Monitoring** (score > 55) — watching silently
- **Gentle Check-in** (score 35–55) — nudge and support prompts
- **Active** (score < 35) — crisis resources + trusted contact alert

Integrated with iCall India crisis line: **9152987821** (Mon–Sat, 8am–10pm)

### 🔍 Drift Interceptor
The most novel feature in the app. No other tool does this.

From 8 answers alone, the Drift Interceptor:
1. **Reconstructs when the drift began** — estimated weeks since social collapse started
2. **Identifies the root cause** — remote work isolation / digital substitution / social avoidance / circle contraction
3. **Prescribes a single turning point action** — matched to the specific root cause and loneliness type
4. **Calculates 30-day recovery probability** — with and without the intervention

Example output: *"Drift began approximately 8 weeks ago. Root cause: Remote work isolation. Turning point: Join one recurring weekly activity. Recovery probability with action: 73%. Without: 38%."*

---

## PDF Report

Every assessment generates a downloadable clinical-grade PDF report via ReportLab:
- Social Health Score with visual gauge
- Loneliness Fingerprint bar chart
- All 6 behavioral readings with status
- Key signals detected with severity
- Personalised interventions
- Crisis support contacts

Shareable with therapists, doctors, and university wellbeing teams.

---

## Training Data

Generated by `data/generate_data.py`:

- **2,000 simulated users × 26 weeks = 52,000 records**
- 5 loneliness type profiles with clinically-calibrated behavioral distributions
- Realistic temporal drift — signals worsen over time per type
- 15 behavioral + 5 demographic features per record

Type distribution:
```
Healthy       33%
Situational   25%
Social        18%
Existential   14%
Chronic       10%
```

---

## The Science

All signals grounded in peer-reviewed research:

- **Cacioppo & Hawkley (2003)** — Late-night activity as isolation predictor
- **Cacioppo et al. (2008)** — The 5-contact weekly threshold
- **Holt-Lunstad et al. (2015)** — Loneliness = 15 cigarettes/day mortality risk
- **Lim et al. (2020)** — Weekend void effect (3× predictive vs weekday)
- **Victor & Yang (2012)** — Future-tense language as early crisis marker

---

## Business Model

| Tier | Customer | Product |
|------|----------|---------|
| Free | Individuals | Full assessment + PDF report |
| University | Colleges | Population dashboard + early alerts |
| Corporate | Employers | Team social health monitoring |
| Municipal | Cities / NGOs | Neighbourhood loneliness index |

---

## Tech Stack

| Tool | Role |
|------|------|
| Streamlit | Full interactive UI |
| Scikit-learn | Gradient Boosting + Random Forest models |
| Pandas | Data processing and feature engineering |
| NumPy | Signal computation and drift modeling |
| Matplotlib | All visualizations (zero white-block leakage) |
| ReportLab | PDF report generation |
| Python stdlib | Base64 encoding, BytesIO, datetime |

---

## UN SDG Alignment

- **SDG #3** — Good Health and Well-Being
- **SDG #10** — Reduced Inequalities (free access for all income levels)
- **SDG #11** — Sustainable Cities and Communities

---

## Crisis Support

If you or someone you know is struggling:

**India:** iCall — 9152987821 (Mon–Sat, 8am–10pm)
**UK:** Samaritans — 116 123 (24/7)
**US:** 988 Suicide & Crisis Lifeline — call or text 988 (24/7)

---

*Tether — The first instrument the loneliness epidemic has ever had.*
