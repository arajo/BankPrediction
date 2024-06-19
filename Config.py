from ModelConfig import ModelConfigV1, ModelConfigV2, ModelConfigV3


class Config:
    def __init__(self, model_name):
        _name = model_name.split("-")
        _version = [x for x in _name if 'v' in x]
        if not _version:
            self.config = ModelConfigV1()
        else:
            _version = _version[0][-1]
            exec(f"self.config = ModelConfigV{_version}()")

        self.config.SHUFFLE_BUFFER_SIZE = self.config.BATCH_SIZE * 2
        self.config.LABEL_DICT_PATH = f"./data/code/{self.config.VERSION}/label_map.json"
        self.config.BANK_DICT_PATH = f"./data/code/{self.config.VERSION}/bank_dic.json"
        self.config.INPUT_DIM = self.config.ADD_ENCODING + 10
