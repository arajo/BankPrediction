class ModelConfig:
    MAX_LENGTH = 14
    EPOCH = 20
    BATCH_SIZE = 64
    SHUFFLE_BUFFER_SIZE = 100
    EMBEDDING_DIM = 32
    NUM_TARGET = 52
    MODEL_PATH = './models/checkpoint-epoch-20-batch-64-trial-002.h5'
    LABEL_FILE = 'data/label_map.json'
    BANK_NAME_FILE = 'data/bank_dic.json'

