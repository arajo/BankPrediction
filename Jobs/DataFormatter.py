import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence


class DataFormatter:
    LABEL_COL = 'label'
    VARIABLE_COL = 'encoded'

    def __init__(self, model_config):
        self.encoding_value = 0
        self.X = None
        self.y = None
        self.MODEL_CONFIG = model_config

    def preprocess(self, test_data: list):
        test_data = [x.replace('-', '') for x in test_data]
        test_data = [self.encoding(x) for x in test_data]
        return tf.data.Dataset.from_tensor_slices(test_data).padded_batch(self.MODEL_CONFIG.BATCH_SIZE,
                                                                          padded_shapes=self.MODEL_CONFIG.MAX_LENGTH)

    def generate(self, data):
        self.X = data[self.VARIABLE_COL].tolist()
        self.y = np.array(data[self.LABEL_COL])
        self.y = to_categorical(self.y)

        x_pad = sequence.pad_sequences(self.X, maxlen=self.MODEL_CONFIG.MAX_LENGTH, padding='post', truncating='post')
        dataset = tf.data.Dataset.from_tensor_slices((x_pad, self.y)).shuffle(
            self.MODEL_CONFIG.SHUFFLE_BUFFER_SIZE).batch(self.MODEL_CONFIG.BATCH_SIZE)
        return dataset

    def encoding(self, account):
        _encoded = []
        for w in account:
            try:
                _encoded.append(int(w) + self.MODEL_CONFIG.ADD_ENCODING)
            except:
                pass
        return _encoded

    def generate_test_data(self, test_data: list):
        test_data = [x.replace('-', '') for x in test_data]
        test_data = [self.encoding(x) for x in test_data]
        return sequence.pad_sequences(test_data, maxlen=self.MODEL_CONFIG.MAX_LENGTH, padding='post')
