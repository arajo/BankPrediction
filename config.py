class ModelConfigV1:
    VERSION = "v1"
    MAX_LENGTH = 14
    EPOCH = 20
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 100
    EMBEDDING_DIM = 32
    NUM_TARGET = 52
    MODEL_PATH = './Models/'
    LABEL_DICT_PATH = f'./data/code/{VERSION}/label_map.json'
    BANK_DICT_PATH = f'./data/code/{VERSION}/bank_dic.json'
    ADD_ENCODING = 1
    INPUT_DIM = ADD_ENCODING + 10


class ModelConfigV2:
    VERSION = "v2"
    MAX_LENGTH = 20
    EPOCH = 500
    BATCH_SIZE = 512
    SHUFFLE_BUFFER_SIZE = 1024
    EMBEDDING_DIM = 32
    NUM_TARGET = 54
    MODEL_PATH = './Models/'
    LABEL_DICT_PATH = f"./data/code/{VERSION}/label_map.json"
    BANK_DICT_PATH = f"./data/code/{VERSION}/bank_dic.json"
    n_trial = 2
    TRAIN_MODEL_NAME = f'checkpoint-epoch-{EPOCH}-batch-{BATCH_SIZE}-trial-v2-00{n_trial}.keras'
    TRAIN_MODEL_PATH = MODEL_PATH + TRAIN_MODEL_NAME
    ADD_ENCODING = 0
    INPUT_DIM = ADD_ENCODING + 10


def get_model_config(model_name):
    if 'v2' in model_name:
        return ModelConfigV2
    else:
        return ModelConfigV1
