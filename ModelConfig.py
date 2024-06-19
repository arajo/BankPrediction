class BaseConfig:
    def __init__(self):
        self.BASE_MODEL_PATH = './Models/'


class ModelConfigV1(BaseConfig):
    def __init__(self):
        super().__init__()
        self.VERSION = "v1"
        self.DICT_VERSION = "v1"
        self.MAX_LENGTH = 14
        self.EPOCH = 20
        self.BATCH_SIZE = 32
        self.EMBEDDING_DIM = 32
        self.NUM_TARGET = 52
        self.ADD_ENCODING = 1


class ModelConfigV2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.VERSION = "v2"
        self.MAX_LENGTH = 20
        self.EPOCH = 500
        self.BATCH_SIZE = 1024
        self.EMBEDDING_DIM = 32
        self.NUM_TARGET = 54
        self.n_trial = 2
        self.TRAIN_MODEL_NAME = f'checkpoint-epoch-{self.EPOCH}-batch-{self.BATCH_SIZE}-trial-v2-00{self.n_trial}.keras'
        self.TRAIN_MODEL_PATH = self.BASE_MODEL_PATH + self.TRAIN_MODEL_NAME
        self.ADD_ENCODING = 0


class ModelConfigV2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.VERSION = "v2"
        self.DICT_VERSION = "v2"
        self.MAX_LENGTH = 20
        self.EPOCH = 500
        self.BATCH_SIZE = 1024
        self.EMBEDDING_DIM = 32
        self.NUM_TARGET = 54
        self.n_trial = 2
        self.TRAIN_MODEL_NAME = f'checkpoint-epoch-{self.EPOCH}-batch-{self.BATCH_SIZE}-trial-v2-00{self.n_trial}.keras'
        self.TRAIN_MODEL_PATH = self.BASE_MODEL_PATH + self.TRAIN_MODEL_NAME
        self.ADD_ENCODING = 0


class ModelConfigV3(BaseConfig):
    def __init__(self):
        super().__init__()
        self.VERSION = "v3"
        self.DICT_VERSION = "v2"
        self.MAX_LENGTH = 20
        self.EPOCH = 500
        self.BATCH_SIZE = 1024
        self.EMBEDDING_DIM = 64
        self.NUM_TARGET = 54
        self.n_trial = str(2).zfill(3)
        self.TRAIN_MODEL_NAME = f'checkpoint-ebd-{self.EMBEDDING_DIM}-num-{self.NUM_TARGET}-trial-{self.VERSION}-{self.n_trial}.keras'
        self.TRAIN_MODEL_PATH = self.BASE_MODEL_PATH + self.TRAIN_MODEL_NAME
        self.ADD_ENCODING = 1   # because of padding 0
