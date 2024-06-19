from Loaders.DataLoader import DataLoader
from Jobs.DataFormatter import DataFormatter
from Jobs.TrainModel import TrainModel
from Jobs.TestModel import TestModel
from config import Config

if __name__ == '__main__':
    MODEL_CONFIG = Config('v2').config
    MODE = 'pred'  # init, train, valid, pred

    dataloader = DataLoader(MODEL_CONFIG)
    train_dataformatter, test_dataformatter = DataFormatter(MODEL_CONFIG), DataFormatter(MODEL_CONFIG)
    data = dataloader.run(MODE)

    if MODE == 'train':
        data['encoded'] = data.account.map(lambda x: train_dataformatter.encoding(x.strip()))
        data['account_len'] = data.account.map(lambda x: len(x))
        data['encoding_len'] = data.encoded.map(lambda x: len(x))
        data = data[data.account_len == data.encoding_len]
        train, test = dataloader.dataset_split(data, test_size=0.1)
        train_dataset = train_dataformatter.generate(train)
        test_dataset = test_dataformatter.generate(test)
        TrainModel(MODEL_CONFIG).train(train_dataset, test_dataset)

    test_account = ['77130201395276']
    test_model = MODEL_CONFIG.TRAIN_MODEL_NAME
    TestModel(MODEL_CONFIG).test(test_model, test_account=test_account)

