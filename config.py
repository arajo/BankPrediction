class BaseConfig:
    def __init__(self):
        self.BASE_MODEL_PATH = './Models/'


class ModelConfigV1(BaseConfig):
    def __init__(self):
        super().__init__()
        self.VERSION = "v1"
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


class Config:
    def __init__(self, model_name):
        self.config = ModelConfigV2() if 'v2' in model_name else ModelConfigV1()
        self.config.SHUFFLE_BUFFER_SIZE = self.config.BATCH_SIZE * 2
        self.config.LABEL_DICT_PATH = f"./data/code/{self.config.VERSION}/label_map.json"
        self.config.BANK_DICT_PATH = f"./data/code/{self.config.VERSION}/bank_dic.json"
        self.config.INPUT_DIM = self.config.ADD_ENCODING + 10
