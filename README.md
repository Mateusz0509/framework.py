# Football Data Analysis Framework

Modular Python framework for football match data analysis, feature engineering, and baseline outcome prediction.

## Features
- `DataLoader` for reading match CSV files with pandas.
- `FeatureEngineer` for rolling averages over the last 5 games.
- `ModelTrainer` for baseline scikit-learn outcome prediction.
- Educational structure that is easy to extend.

## Project Structure
```text
football_repo/
├── data/
│   └── sample_matches.csv
├── src/
│   └── football_framework.py
├── requirements.txt
└── README.md
```

## Installation
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/football_framework.py
```

## Data Format
Expected columns in the CSV:
- `date`
- `home_team`
- `away_team`
- `home_xg`
- `away_xg`
- `home_shots`
- `away_shots`
- `home_possession`
- `away_possession`
- `result` (`H`, `D`, `A`)

## Next Extensions
- Add away/home-specific rolling features.
- Add form, points, goal difference, and ELO.
- Replace baseline classifier with RandomForest or XGBoost.
- Add model persistence with `joblib`.
