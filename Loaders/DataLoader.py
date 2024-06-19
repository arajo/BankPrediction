import pandas as pd
import json

from sklearn.model_selection import train_test_split


class DataLoader:
    DATASET_PATH = "./data/bank_accounts/merged.csv"
    TRAIN_DATA_PATH = "./data/bank_accounts/train.json"
    VALID_DATA_PATH = "./data/bank_accounts/valid.json"
    STRATIFY_COL = 'bank'

    def __init__(self, model_config):
        self.label_map, self.inverse_label_map = {}, {}
        self.bank_dic, self.inverse_bank_dic = {}, {}
        self.MODEL_CONFIG = model_config

    def generate_dict(self, data, save_dict=False):
        label_map = {l: i for i, l in enumerate(data.bank.unique())}

        bank_dic = {}
        _tmp_unique_df = data[['bank_name', 'bank']].drop_duplicates()
        for row in _tmp_unique_df.iterrows():
            bank_dic[row[1]['bank_name']] = row[1]['bank']

        if save_dict:
            with open(self.MODEL_CONFIG.BANK_DICT_PATH, "w+") as json_file:
                json.dump(bank_dic, json_file)

            with open(self.MODEL_CONFIG.LABEL_DICT_PATH, "w+") as json_file:
                json.dump(label_map, json_file)

    def dataset_split(self, data, save_data=False, test_size=0.1):

        train, valid = train_test_split(data, random_state=2024, shuffle=True, stratify=data[self.STRATIFY_COL],
                                        test_size=test_size)

        if save_data:
            train.to_json(self.TRAIN_DATA_PATH, orient='records')
            valid.to_json(self.VALID_DATA_PATH, orient='records')
        return train, valid

    @staticmethod
    def load_dataset(path, return_type=None):
        with open(path, "r") as json_file:
            data = json.load(json_file)
        if return_type == 'df':
            return pd.DataFrame(data)
        return data

    def run(self, mode):
        """
        :param mode: {init (Create dataset), train, valid, pred}
        :return: dataset (pd.DataFrame)
        """
        if mode == 'init':
            data = self.load_dataset(self.DATASET_PATH, return_type='df')
            self.generate_dict(data, False)
            data['label'] = data.bank.map(lambda x: self.label_map[x])
            self.dataset_split(data, False)
        elif mode == 'train':
            data = self.load_dataset(self.TRAIN_DATA_PATH, return_type='df')
        elif mode == 'valid':
            data = self.load_dataset(self.VALID_DATA_PATH, return_type='df')
        elif mode == 'pred':
            data = None

        self.label_map = self.load_dataset(self.MODEL_CONFIG.LABEL_DICT_PATH)
        self.bank_dic = self.load_dataset(self.MODEL_CONFIG.BANK_DICT_PATH)
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.inverse_bank_dic = {v: k for k, v in self.bank_dic.items()}
        return data
