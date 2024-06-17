class ModelConfig:
    VERSION = "v1"
    MAX_LENGTH = 14
    EPOCH = 20
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 100
    EMBEDDING_DIM = 32
    NUM_TARGET = 52
    MODEL_PATH = 'Models/'
    LABEL_DICT_PATH = f'./data/code/{VERSION}/label_map.json'
    BANK_DICT_PATH = f'./data/code/{VERSION}/bank_dic.json'


class ModelConfig_v2:
    VERSION = "v2"
    MAX_LENGTH = 20
    EPOCH = 500
    BATCH_SIZE = 256
    SHUFFLE_BUFFER_SIZE = 100
    EMBEDDING_DIM = 32
    NUM_TARGET = 54
    MODEL_PATH = 'Models/'
    LABEL_DICT_PATH = f"../data/code/{VERSION}/bank_dic.json"
    BANK_DICT_PATH = f"../data/code/{VERSION}/label_map.json"
