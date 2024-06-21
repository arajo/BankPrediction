from Jobs.DataFormatter import DataFormatter
from Loaders.ModelLoader import ModelLoader


class TestModel:
    def __init__(self, model_config, logger):
        self.MODEL_CONFIG = model_config
        self.logger = logger

    def test(self, model, test_dataset=None, test_account=None):
        self.logger.info("Test Model : " + model)
        model_loader = ModelLoader(model, self.logger)

        if test_dataset:
            scores = model_loader.model.evaluate(test_dataset)
            self.logger.info("Accuracy: %.2f%%" % (scores[1] * 100))

        if test_account:
            self.logger.info(f"Test Account: {test_account}")
            test_data = DataFormatter(self.MODEL_CONFIG).generate_test_data(test_account)
            results = model_loader.predict_top_k(test_data)
            for i, v in results.items():
                self.logger.info(f"예측 은행은: '{i}' 확률은: {v} % 입니다!")
