import sys
from dotenv import load_dotenv
import os

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o diretório definido em PYTHONPATH ao sys.path
sys.path.append(os.getenv("PYTHONPATH"))

import unittest  # noqa
import pandas as pd  # noqa
from src.feature_engineer import FeatureEngineer  # noqa


class TestFeatureEngineer(unittest.TestCase):

    def setUp(self):
        # cria um DataFrame de exemplo para os testes
        data = {
            "date": pd.date_range(start="2022-01-01", periods=10, freq="D"),
            "target": range(10)
        }
        self.df = pd.DataFrame(data)
        self.df.set_index("date", inplace=True)

    def test_lags(self):
        fe = FeatureEngineer(target="target", lags=3, window_size=[2, 3])
        transformed_df = fe.fit_transform(self.df)

        # imprime os valores das colunas de defasagem para inspeção
        print(transformed_df[["lag_1", "lag_2", "lag_3"]])

        # verifica se as colunas de defasagem foram criadas
        self.assertIn("lag_1", transformed_df.columns)
        self.assertIn("lag_2", transformed_df.columns)
        self.assertIn("lag_3", transformed_df.columns)

        # verifica se os valores estão corretos após preenchimento de NaN com 0
        self.assertEqual(transformed_df["lag_1"].iloc[0], 0)
        self.assertEqual(transformed_df["lag_1"].iloc[1], 1)
        self.assertEqual(transformed_df["lag_1"].iloc[2], 2)

        self.assertEqual(transformed_df["lag_2"].iloc[0], 0)
        self.assertEqual(transformed_df["lag_2"].iloc[1], 0)
        self.assertEqual(transformed_df["lag_2"].iloc[2], 1)
        self.assertEqual(transformed_df["lag_2"].iloc[3], 2)

        self.assertEqual(transformed_df["lag_3"].iloc[0], 0)
        self.assertEqual(transformed_df["lag_3"].iloc[1], 0)
        self.assertEqual(transformed_df["lag_3"].iloc[2], 0)
        self.assertEqual(transformed_df["lag_3"].iloc[3], 1)
        self.assertEqual(transformed_df["lag_3"].iloc[4], 2)

    def test_rolling_features(self):
        fe = FeatureEngineer(target="target", lags=0, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se as colunas de médias móveis foram criadas
        self.assertIn("rolling_mean_2", transformed_df.columns)
        self.assertIn("rolling_std_2", transformed_df.columns)
        self.assertIn("ewm_mean_2", transformed_df.columns)
        self.assertIn("ewm_std_2", transformed_df.columns)

    def test_diff(self):
        fe = FeatureEngineer(target="target", lags=0, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se a coluna de diferença foi criada
        self.assertIn("diff", transformed_df.columns)
        # verifica se os valores estão corretos
        self.assertEqual(transformed_df["diff"].iloc[1], 1)

    def test_date_features(self):
        fe = FeatureEngineer(target="target", lags=0, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se as colunas de data foram criadas
        self.assertIn("year", transformed_df.columns)
        self.assertIn("quarter", transformed_df.columns)
        self.assertIn("month", transformed_df.columns)
        self.assertIn("day", transformed_df.columns)
        self.assertIn("day_of_week", transformed_df.columns)

    def test_no_target_column(self):
        fe = FeatureEngineer(target="target", lags=1, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se a coluna alvo foi removida
        self.assertNotIn("target", transformed_df.columns)

    def test_fill_na(self):
        fe = FeatureEngineer(target="target", lags=3, window_size=[2])
        transformed_df = fe.fit_transform(self.df)

        # verifica se os valores NaN foram preenchidos com 0
        self.assertFalse(transformed_df.isna().any().any())


if __name__ == "__main__":
    unittest.main()
