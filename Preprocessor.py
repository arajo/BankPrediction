import tensorflow as tf
from config import ModelConfig


class Preprocessor:
    def __init__(self, ):
        self.batch_size = ModelConfig.BATCH_SIZE
        self.max_length = ModelConfig.MAX_LENGTH

    def preprocess(self, test_data: list):
        test_data = [x.replace('-', '') for x in test_data]
        test_data = [self.add_one_encoding(x) for x in test_data]
        return tf.data.Dataset.from_tensor_slices(test_data).padded_batch(self.batch_size,
                                                                          padded_shapes=self.max_length)

    @staticmethod
    def add_one_encoding(account):
        _encoded = []
        for w in account:
            _encoded.append(int(w) + 1)
        return _encoded
