# PL Transfer Value

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-exploratory%20analysis-7C3AED)

A machine-learning application for estimating professional football players' market values from performance, age, position, competition, club strength, and contract context.

The project uses Transfermarkt data from the Kaggle `davidcariboo/player-scores` dataset. Its product direction is a scouting-oriented valuation system with interpretable predictions, bargain detection, comparable players, prediction intervals, and squad-building tools.

## Project status

**Phase 2 of 8 complete: data acquisition and exploratory analysis.**

Current capabilities:

- Reproducible Python 3.12 environment
- Kaggle dataset ingestion and validation
- Historical player-valuation data
- Exploratory notebook with five interactive Plotly analyses
- Leakage-aware modelling plan
- Professional project structure

The feature-engineering pipeline, trained models, Streamlit interface, screenshots, and deployment link will be added in subsequent phases.

## Why this project?

Football market value is influenced by more than raw goals and assists. A useful valuation model must consider:

- Playing time and rate-based performance
- Age and career stage
- Position-specific expectations
- League and club context
- Contract duration
- Historical timing of both performance and valuation data

The project treats player valuation as a time-dependent machine-learning problem rather than a static leaderboard.

## Key exploratory findings

### Market value is heavily right-skewed

Most players have relatively low values, while a small number of elite players are valued above €100M. Training directly on raw euros would cause large-value players to dominate squared-error loss.

The modelling target will therefore be:

```python
y = np.log1p(market_value_in_eur)
```

Predictions will be returned to euros using:

```python
market_value_in_eur = np.expm1(predicted_y)
```

### Age has a non-linear effect

Median value remains comparatively stable through the 23–29 prime-age period and declines noticeably from age 30. The model will include both `age` and `age_squared` so the baseline regression can represent a curved age profile.

### Position changes how performance should be interpreted

In the complete player snapshot, attackers, midfielders, and defenders share a €0.30M median value, while goalkeepers have a €0.15M median.

Among players with at least 300 minutes in the 2025/26 analysis window:

| Position | Players | Median goals | Median value |
|---|---:|---:|---:|
| Attack | 1,640 | 3 | €2.5M |
| Midfield | 1,742 | 1 | €2.2M |
| Defender | 2,092 | 0 | €2.0M |
| Goalkeeper | 494 | 0 | €1.2M |

Zero goals has a different meaning for a goalkeeper, defender, and attacker. Position will therefore be one-hot encoded, with position-aware interactions considered later.

### Contract duration has a strong association with value

| Contract remaining | Median value |
|---|---:|
| Less than 1 year | €0.3M |
| 1–2 years | €0.5M |
| 2–3 years | €0.9M |
| 3–4 years | €1.8M |
| 4+ years | €5.5M |

Players with at least four years remaining have a median value approximately 18 times higher than players entering their final year.

This is an association, not proof of causation. Clubs are also more likely to give long contracts to younger and already valuable players.

## Dataset

Source: [Transfermarkt player scores dataset on Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores)

| File | Rows | Purpose |
|---|---:|---|
| `players.csv` | 50,149 | Player identity, birth date, position, club, contract, and current value |
| `appearances.csv` | 1,894,350 | Match-level goals, assists, minutes, dates, and competitions |
| `clubs.csv` | 796 | Club and domestic-competition context |
| `player_valuations.csv` | 656,301 | Historical market values used for time-aligned targets |

Target coverage:

- 41,528 players have a current market value
- 41,528 players have historical valuation records
- 26,874 valued players have a known contract date
- 17,306 valued players have a known, non-expired contract at the analysis reference date

Raw dataset files are intentionally excluded from Git.

## Methodology

### Historical alignment

A player's current market value must not be attached to an old season. Historical targets are taken from `player_valuations.csv` and aligned with the relevant performance window.

### Time-based evaluation

The model will train on earlier seasons and test on the most recent season. A random split would mix future and past observations and produce an unrealistically optimistic evaluation.

### Evaluation in real euros

Although the model trains on `log1p(value)`, predictions will be converted back to euros before reporting:

- Mean Absolute Error
- Root Mean Squared Error
- R²

### Planned model comparison

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor
4. Hyperparameter tuning of the winning model

## Project structure

```text
pl-transfer-value/
├── data/
│   ├── raw/                 # Kaggle CSV files (gitignored)
│   └── processed/           # Engineered features (gitignored)
├── models/                  # Deployed model artifacts
├── notebooks/
│   └── 01_explore.ipynb     # Exploratory analysis
├── src/
│   ├── __init__.py
│   └── load.py              # Data loading and validation
├── .streamlit/              # Streamlit configuration
├── app.py                   # Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

## Windows setup

### 1. Create and enter the project folder

```powershell
New-Item -ItemType Directory -Force "$HOME\pl-transfer-value"
Set-Location "$HOME\pl-transfer-value"
```

### 2. Create the Python 3.12 environment

```powershell
uv python install 3.12
uv venv --python 3.12 --seed venv
```

### 3. Activate the environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
python --version
```

Expected output:

```text
Python 3.12.x
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Kaggle authentication

Authenticate through the Kaggle CLI:

```powershell
kaggle auth login
```

Never commit API tokens, `kaggle.json`, or other credentials.

Download the required files:

```powershell
$files = @(
    "players.csv",
    "appearances.csv",
    "clubs.csv",
    "player_valuations.csv"
)

foreach ($file in $files) {
    kaggle datasets download davidcariboo/player-scores `
        -f $file `
        -p "data/raw" `
        --unzip
}
```

## Run the data audit

```powershell
python src/load.py
```

## Open the exploratory notebook

```powershell
code notebooks/01_explore.ipynb
```

Select the `Python 3.12 (pl-transfer-value)` kernel and run the cells in order.

## Roadmap

- [x] Python 3.12 environment and project skeleton
- [x] Kaggle data acquisition
- [x] Data-quality audit
- [x] Exploratory analysis
- [ ] Player-season feature engineering
- [ ] Time-based model comparison
- [ ] Hyperparameter tuning and model persistence
- [ ] Streamlit valuation interface
- [ ] SHAP explanations
- [ ] Over/under-valued player detector
- [ ] Prediction intervals
- [ ] Comparable-player search
- [ ] What-if analysis and age projection
- [ ] Squad builder
- [ ] Streamlit Community Cloud deployment

## Limitations

- Transfermarkt market values are estimates, not confirmed transfer fees.
- Market value can be influenced by injuries, reputation, nationality, commercial appeal, and negotiation conditions not fully represented in match data.
- Dataset coverage is stronger for well-documented leagues and players.
- The simple July-to-June season window is most appropriate for European competitions.
- Current contract-expiration dates are not reliable historical contract records.
- Missing contract information reduces the usable contract-analysis sample.
- The contract relationship is observational and may be confounded by age and existing player quality.
- The final system should support scouting decisions rather than replace expert judgment.

## Application

The Streamlit screenshot and live deployment link will be added after the application is implemented and deployed.
