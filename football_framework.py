import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load_matches(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, parse_dates=['date'])
        return df.sort_values(['date', 'home_team', 'away_team']).reset_index(drop=True)

class FeatureEngineer:
    def __init__(self, window: int = 5):
        self.window = window

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out['home_team_xg_roll5'] = out.groupby('home_team')['home_xg'].transform(lambda s: s.shift(1).rolling(self.window, min_periods=1).mean())
        out['home_team_shots_roll5'] = out.groupby('home_team')['home_shots'].transform(lambda s: s.shift(1).rolling(self.window, min_periods=1).mean())
        out['home_team_pos_roll5'] = out.groupby('home_team')['home_possession'].transform(lambda s: s.shift(1).rolling(self.window, min_periods=1).mean())
        out['away_team_xg_roll5'] = out.groupby('away_team')['away_xg'].transform(lambda s: s.shift(1).rolling(self.window, min_periods=1).mean())
        out['away_team_shots_roll5'] = out.groupby('away_team')['away_shots'].transform(lambda s: s.shift(1).rolling(self.window, min_periods=1).mean())
        out['away_team_pos_roll5'] = out.groupby('away_team')['away_possession'].transform(lambda s: s.shift(1).rolling(self.window, min_periods=1).mean())
        return out

class ModelTrainer:
    def __init__(self):
        self.pipeline = None

    def build_model(self):
        numeric_features = [
            'home_team_xg_roll5', 'home_team_shots_roll5', 'home_team_pos_roll5',
            'away_team_xg_roll5', 'away_team_shots_roll5', 'away_team_pos_roll5'
        ]
        categorical_features = ['home_team', 'away_team']

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numeric_features)
        ])

        self.pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        return self.pipeline

    def train(self, X_train, y_train):
        if self.pipeline is None:
            self.build_model()
        self.pipeline.fit(X_train, y_train)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        preds = self.pipeline.predict(X_test)
        return accuracy_score(y_test, preds)

if __name__ == '__main__':
    loader = DataLoader('data/sample_matches.csv')
    df = loader.load_matches()

    fe = FeatureEngineer(window=5)
    df_feat = fe.add_rolling_features(df).dropna().reset_index(drop=True)

    feature_cols = [
        'home_team', 'away_team',
        'home_team_xg_roll5', 'home_team_shots_roll5', 'home_team_pos_roll5',
        'away_team_xg_roll5', 'away_team_shots_roll5', 'away_team_pos_roll5'
    ]
    X = df_feat[feature_cols]
    y = df_feat['result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    trainer = ModelTrainer()
    trainer.build_model()
    trainer.train(X_train, y_train)
    acc = trainer.evaluate(X_test, y_test)

    print(f'Baseline accuracy: {acc:.3f}')
