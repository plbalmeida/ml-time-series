from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineer para séries temporais.

    Esta classe cria várias características baseadas em uma série temporal,
    como lags, médias móveis, diferenças etc.

    Args:
        target (str): O nome da coluna alvo na série temporal.
        lags (int): O número de defasagens a serem criadas.
        window_size (list): A lista com tamanhos da janela para calcular variáveis móveis.

    Attributes:
        target (str): O nome da coluna alvo na série temporal.
        lags (int): O número de defasagens a serem criadas.
        window_size (list): A lista com tamanhos da janela para calcular variáveis móveis.
    """  # noqa

    def __init__(self, target, lags, window_size):
        """
        Inicializa o FeatureEngineer com os parâmetros fornecidos.

        Args:
            target (str): O nome da coluna alvo na série temporal.
            lags (int): O número de defasagens a serem criadas.
            window_size (list): A lista com tamanhos da janela para calcular variáveis móveis.
        """  # noqa
        self.target = target
        self.lags = lags
        self.window_size = window_size

    def fit(self, X, y=None):
        """
        Método de ajuste necessário para conformidade com o scikit-learn,
        não realiza nenhuma operação.

        Args:
            X (pd.DataFrame): O dataframe de entrada.
            y (pd.Series, opcional): A série alvo (não utilizada).

        Returns:
            self: Retorna a instância do próprio objeto.
        """
        return self

    def transform(self, X):
        """
        Transforma a série temporal adicionando características engenheiradas.

        Args:
            X (pd.DataFrame): O dataframe de entrada contendo a série temporal.

        Returns:
            pd.DataFrame: Um novo dataframe com as características adicionadas.
        """
        X = X.copy()

        for lag in range(0, self.lags):
            X[f"lag_{lag+1}"] = X[self.target].shift(lag)

        for window in self.window_size:
            X[f"rolling_mean_{window}"] = X[self.target].rolling(window=window).mean()  # noqa
            X[f"rolling_std_{window}"] = X[self.target].rolling(window=window).std()  # noqa
            X[f"ewm_mean_{window}"] = X[self.target].ewm(span=window).mean()
            X[f"ewm_std_{window}"] = X[self.target].ewm(span=window).std()

        X["diff"] = X[self.target].diff()
        X["year"] = X.index.year
        X["quarter"] = X.index.quarter
        X["month"] = X.index.month
        X["day"] = X.index.day
        X["day_of_week"] = X.index.dayofweek
        X = X.drop(columns=[self.target])
        X.fillna(0, inplace=True)
        return X
