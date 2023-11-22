from tensorflow.keras.preprocessing import sequence
from config import ModelConfig


class Preprocessor:
    def preprocess(self, test_data: list):
        test_data = [x.replace('-', '') for x in test_data]
        test_data = [self.add_one_encoding(x) for x in test_data]
        return sequence.pad_sequences(test_data, maxlen=ModelConfig.MAX_LENGTH, padding='post')

    @staticmethod
    def add_one_encoding(account):
        _encoded = []
        for w in account:
            _encoded.append(int(w) + 1)
        return _encoded
