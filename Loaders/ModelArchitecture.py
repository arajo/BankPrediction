import tensorflow as tf

from tensorflow.keras.models import Sequential


class ModelArchitecture:
    def __init__(self, model_config):
        self.MODEL_CONFIG = model_config

    def architecture_model_v1(self, ):
        model = Sequential(
            [
                tf.keras.layers.Embedding(self.MODEL_CONFIG.INPUT_DIM, self.MODEL_CONFIG.EMBEDDING_DIM,
                                          input_length=self.MODEL_CONFIG.MAX_LENGTH),
                tf.keras.layers.LSTM(self.MODEL_CONFIG.EMBEDDING_DIM * 2),
                tf.keras.layers.Dense(self.MODEL_CONFIG.NUM_TARGET, activation='softmax')
            ]
        )
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def architecture_model_v2(self, ):
        model = Sequential(
            [
                tf.keras.layers.Embedding(self.MODEL_CONFIG.INPUT_DIM, self.MODEL_CONFIG.EMBEDDING_DIM,
                                          input_length=self.MODEL_CONFIG.MAX_LENGTH),
                tf.keras.layers.LSTM(self.MODEL_CONFIG.EMBEDDING_DIM * 2, return_sequences=True),
                tf.keras.layers.LSTM(self.MODEL_CONFIG.EMBEDDING_DIM),
                tf.keras.layers.Dense(self.MODEL_CONFIG.NUM_TARGET, activation='softmax')
            ]
        )
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
