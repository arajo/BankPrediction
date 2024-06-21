import unittest

from Config import Config
from Loaders.DataLoader import DataLoader
from logger import Logger


class Tests(unittest.TestCase):
    def setUp(self):
        self.logger = Logger().get_logger()
        self.model_config_v2 = Config('v2').config
        self.model_config_v1 = Config('v1').config

    def testDictionaryCount_v2(self):
        data_loader = DataLoader(self.model_config_v2, self.logger)
        data_loader.load_dictionary()
        self.assertEqual(len(data_loader.bank_dic), 54)

    def testDictionaryCount_v1(self):
        data_loader = DataLoader(self.model_config_v1, self.logger)
        data_loader.load_dictionary()
        self.assertEqual(len(data_loader.bank_dic), 52)
