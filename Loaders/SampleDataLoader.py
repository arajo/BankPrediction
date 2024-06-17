import pandas as pd
import xlwings as xw

from config import ModelConfig


class SampleDataLoader:
    def __init__(self):
        self.DATA_PATH = "../data/입금기관 계좌번호 학습 데이터(랜덤)_reformat.xlsx"

    def load_data(self):
        workbook = xw.Book(self.DATA_PATH)
        sheet1 = self.read_excel(workbook, 1)
        sheet2 = self.read_excel(workbook, 2)
        data = pd.concat([sheet1, sheet2], axis=0)
        workbook.close()

        data = data.iloc[:, :-1]  # '끝' 컬럼 제거
        data = data.dropna().reset_index(drop=True)
        data['은행명'] = data['은행명'].map(lambda x: x.strip())
        assert data.shape[0] == ModelConfig.NUM_TARGET
        return data

    @staticmethod
    def read_excel(workbook, i):
        worksheet = workbook.sheets(i)
        content = worksheet.used_range.value
        return pd.DataFrame(content[1:], columns=content[0])
