import sys
from dotenv import load_dotenv
import os

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o diretório definido em PYTHONPATH ao sys.path
sys.path.append(os.getenv("PYTHONPATH"))

import unittest  # noqa
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit  # noqa
from src.model_train import create_pipeline_and_search  # noqa


class TestPipelineAndSearch(unittest.TestCase):

    def setUp(self):
        # parâmetros para a função
        self.target = "target"
        self.lags = 3
        self.window_size = [2, 3]

    def test_create_pipeline_and_search(self):
        # cria o objeto de busca de hiperparâmetros
        search = create_pipeline_and_search(self.target, self.lags, self.window_size)  # noqa

        # verifica se o objeto retornado é uma instância de HalvingGridSearchCV
        self.assertIsInstance(search, HalvingGridSearchCV)

        # verifica se o pipeline está configurado corretamente
        self.assertIn("feature_engineering", search.estimator.named_steps)
        self.assertIn("scaler", search.estimator.named_steps)
        self.assertIn("model", search.estimator.named_steps)

        # verifica se os parâmetros do grid de busca estão corretos
        param_grid = search.param_grid
        self.assertIn("model__base_estimator__n_estimators", param_grid)
        self.assertIn("model__base_estimator__learning_rate", param_grid)
        self.assertIn("model__base_estimator__max_depth", param_grid)
        self.assertIn("model__base_estimator__min_samples_split", param_grid)
        self.assertIn("model__base_estimator__min_samples_leaf", param_grid)

        # verifica os valores dos parâmetros do grid
        self.assertEqual(param_grid["model__base_estimator__n_estimators"], [100, 200, 300])  # noqa
        self.assertEqual(param_grid["model__base_estimator__learning_rate"], [0.01, 0.05, 0.1])  # noqa
        self.assertEqual(param_grid["model__base_estimator__max_depth"], [3, 5, 8])  # noqa
        self.assertEqual(param_grid["model__base_estimator__min_samples_split"], [2, 5, 10])  # noqa
        self.assertEqual(param_grid["model__base_estimator__min_samples_leaf"], [1, 2, 4])  # noqa

    def test_pipeline_structure(self):
        # cria o objeto de busca de hiperparâmetros
        search = create_pipeline_and_search(self.target, self.lags, self.window_size)  # noqa

        # verifica se a pipeline tem os passos corretos
        steps = search.estimator.steps
        self.assertEqual(steps[0][0], "feature_engineering")
        self.assertEqual(steps[1][0], "scaler")
        self.assertEqual(steps[2][0], "model")

    def test_time_series_split(self):
        # cria o objeto de busca de hiperparâmetros
        search = create_pipeline_and_search(self.target, self.lags, self.window_size)  # noqa

        # Verifica se o cross-validator é um TimeSeriesSplit
        self.assertIsInstance(search.cv, TimeSeriesSplit)
        self.assertEqual(search.cv.n_splits, 6)


if __name__ == "__main__":
    unittest.main()
