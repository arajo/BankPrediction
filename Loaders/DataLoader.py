import pandas as pd
import json


class DataLoader:
    def __init__(self):
        self.DASTSET_PATH = "../data/bank_accounts/merged.csv"
        self.VERSION = "v2"
        self.BANK_DICT_PATH = f"../data/code/{self.VERSION}/bank_dic.json"
        self.LABEL_DICT_PATH = f"../data/code/{self.VERSION}/label_map.json"
        self.TRAIN_DATA_PATH = "../data/bank_accounts/train.json"
        self.VALID_DATA_PATH = "../data/bank_accounts/valid.json"
        self.NUM_TARGET = 54
        self.label_map, self.inverse_label_map, self.bank_dic = {}, {}, {}

    def load_raw_data(self):
        with open(self.DATASET_PATH, "r") as file:
            data = json.load(file)
        return pd.DataFrame(data)

    def generate_dict(self, data, save_dict=False):
        label_map = {l: i for i, l in enumerate(data.bank.unique())}

        bank_dic = {}
        _tmp_unique_df = data[['bank_name', 'bank']].drop_duplicates()
        for row in _tmp_unique_df.iterrows():
            bank_dic[row[1]['bank_name']] = row[1]['bank']

        if save_dict:
            with open(self.BANK_DICT_PATH, "w+") as json_file:
                json.dump(bank_dic, json_file)

            with open(self.LABEL_DICT_PATH, "w+") as json_file:
                json.dump(label_map, json_file)

    def dataset_split(self, data, save_data=False):
        from sklearn.model_selection import train_test_split

        data['label'] = data.bank.map(lambda x: self.label_map[x])
        train, valid = train_test_split(data, stratify=data['bank'], test_size=0.1)

        if save_data:
            train.to_json(self.TRAIN_DATA_PATH, orient='records')
            valid.to_json(self.VALID_DATA_PATH, orient='records')

    def load_dict(self):
        with open(self.LABEL_DICT_PATH) as json_file:
            self.label_map = json.load(json_file)

        with open(self.BANK_DICT_PATH) as json_file:
            self.bank_dic = json.load(json_file)

        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    @staticmethod
    def load_dataset(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data

    def run(self, mode):
        """
        :param mode: {0: initial(Create dataset), 1: train data, 2: valid data}
        :return: dataset
        """
        if mode == 0:
            data = self.load_raw_data()
            self.generate_dict(data, False)
            self.dataset_split(data, False)
        elif mode == 1:
            data = self.load_dataset(self.TRAIN_DATA_PATH)
        elif mode == 2:
            data = self.load_dataset(self.VALID_DATA_PATH)
        self.load_dict()
        return data
