from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit
from sklearn.multioutput import RegressorChain
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.feature_engineer import FeatureEngineer


def create_pipeline_and_search(target, lags, window_size):
    """
    Cria uma pipeline e configura uma busca de hiperparâmetros usando HalvingGridSearchCV.

    Args:
        target (str): Nome da coluna alvo.
        lags (int): Número de lags a serem criados.
        window_size (int): Janela para criar features baseadas em janelas.

    Returns:
        HalvingGridSearchCV: Objeto configurado para busca de hiperparâmetros.
    """  # noqa 401
    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer(target, lags, window_size)),  # noqa 401
        ("scaler", StandardScaler()),
        ("model", RegressorChain(base_estimator=GradientBoostingRegressor(random_state=123), random_state=123))  # noqa
    ])

    param_grid = {
        "model__base_estimator__n_estimators": [100, 200, 300],
        "model__base_estimator__learning_rate": [0.01, 0.05, 0.1],
        "model__base_estimator__max_depth": [3, 5, 8],
        "model__base_estimator__min_samples_split": [2, 5, 10],
        "model__base_estimator__min_samples_leaf": [1, 2, 4]
    }

    tscv = TimeSeriesSplit(n_splits=6)

    search = HalvingGridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        factor=3,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
        random_state=132
    )

    return search
