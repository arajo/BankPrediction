import tensorflow as tf

from Loaders.DataLoader import DataLoader
from Loaders.ModelArchitecture import ModelArchitecture


class ModelLoader:
    def __init__(self, model_name, MODEL_CONFIG):
        self.MODEL_CONFIG = MODEL_CONFIG
        self.model = self.build_model(model_name)
        self.model.load_weights(MODEL_CONFIG.MODEL_PATH + model_name)
        self.inverse_label_map, self.bank_dic = self.load_dictionary()

    def predict_top_k(self, test_data, k=5):
        pred = self.model.predict(test_data)
        values, indices = tf.math.top_k(pred, k)

        results = {}
        for i, v in zip(indices.numpy()[0], values.numpy()[0]):
            results[self.bank_dic[self.inverse_label_map[i]]] = round(v * 100, 2)
        return results

    def load_dictionary(self, ):
        dataloader = DataLoader(self.MODEL_CONFIG)
        dataloader.run('pred')
        if self.MODEL_CONFIG.VERSION == "v2":
            return dataloader.inverse_label_map, dataloader.inverse_bank_dic
        else:
            return dataloader.inverse_label_map, dataloader.bank_dic

    def build_model(self, model_name):
        if model_name.split('.')[0][-3:] in ['001', '002', '003', '004'] and 'v2' not in model_name:
            return ModelArchitecture(self.MODEL_CONFIG).architecture_model_v1()
        else:
            return ModelArchitecture(self.MODEL_CONFIG).architecture_model_v2()
