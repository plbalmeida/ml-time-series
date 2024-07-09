import sys
from dotenv import load_dotenv
import os

# carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# adiciona o diretório definido em PYTHONPATH ao sys.path
sys.path.append(os.getenv("PYTHONPATH"))

import unittest  # noqa
import pandas as pd  # noqa
from src.utils import target_transform  # noqa


class TestTargetTransform(unittest.TestCase):

    def setUp(self):
        # cria um DataFrame de exemplo para os testes
        self.train = pd.DataFrame({'target': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        self.target = 'target'
        self.horizon = 3

    def test_target_transform_columns(self):
        # transforma o alvo para múltiplos passos à frente
        transformed = target_transform(self.train, self.target, self.horizon)

        # verifica se as colunas estão nomeadas corretamente
        expected_columns = ['target_t1', 'target_t2', 'target_t3']
        self.assertListEqual(list(transformed.columns), expected_columns)

    def test_target_transform_horizon_one(self):
        # transforma o alvo com horizonte de um passo à frente
        horizon_one = 1
        transformed = target_transform(self.train, self.target, horizon_one)

        # verifica se a coluna está nomeada corretamente
        expected_columns = ['target_t1']
        self.assertListEqual(list(transformed.columns), expected_columns)


if __name__ == "__main__":
    unittest.main()
