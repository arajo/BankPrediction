import json
import tensorflow as tf
from tensorflow.keras.models import Sequential

from config import ModelConfig


class ModelLoader:
    def __init__(self):
        self.model = self.build_model()
        self.model.load_weights(ModelConfig.MODEL_PATH)
        self.inverse_label_map, self.bank_dic = self.load_dictionary()

    def predict_top_k(self, test_data, k=5):
        pred = self.model.predict(test_data)
        values, indices = tf.math.top_k(pred, k)

        results = {}
        for i, v in zip(indices.numpy()[0], values.numpy()[0]):
            results[self.bank_dic[self.inverse_label_map[i]]] = round(v * 100, 2)
        return results

    @staticmethod
    def build_model():
        model = Sequential(
            [
                tf.keras.layers.Embedding(11, ModelConfig.EMBEDDING_DIM, input_length=ModelConfig.MAX_LENGTH),
                tf.keras.layers.LSTM(ModelConfig.EMBEDDING_DIM * 2),
                tf.keras.layers.Dense(ModelConfig.NUM_TARGET, activation='softmax')
            ]
        )
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def load_dictionary():
        with open(ModelConfig.LABEL_FILE, 'r') as f:
            label_map = json.load(f)

        inverse_label_map = {v: k for k, v in label_map.items()}

        with open(ModelConfig.BANK_NAME_FILE, 'r') as f:
            bank_dic = json.load(f)
        return inverse_label_map, bank_dic
