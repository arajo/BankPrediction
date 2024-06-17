import tensorflow as tf

from config import ModelConfig


class DataFormatter(ModelConfig):
    def __init__(self, ):
        super().__init__()
        n_trial = 1
        MODEL_PATH = f'../models/checkpoint-epoch-{self.EPOCH}-batch-{self.BATCH_SIZE}-trial-v2-00{n_trial}.keras'

    def preprocess(self, test_data: list):
        test_data = [x.replace('-', '') for x in test_data]
        test_data = [self.add_one_encoding(x) for x in test_data]
        return tf.data.Dataset.from_tensor_slices(test_data).padded_batch(self.BATCH_SIZE,
                                                                          padded_shapes=self.MAX_LENGTH)

    @staticmethod
    def add_one_encoding(account):
        _encoded = []
        for w in account:
            _encoded.append(int(w) + 1)
        return _encoded
