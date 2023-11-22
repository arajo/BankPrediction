import os
import streamlit as st

from PIL import Image
from ModelLoader import ModelLoader
from Preprocessor import Preprocessor

st.title('🏦 입금기관 예측 모델 테스트 💰')


@st.cache_resource
def load_model():
    return ModelLoader()


def show_image(top_pred_bank, c2):
    default_image_path = "./data/bank_images/"
    if '국민' in top_pred_bank:
        bank_image = Image.open(default_image_path + "kb.jpeg")
    elif '신한' in top_pred_bank:
        bank_image = Image.open(default_image_path + "shinhan.jpeg")
    elif '농협' in top_pred_bank:
        bank_image = Image.open(default_image_path + "nh.png")
    else:
        bank_image = None
    if bank_image:
        c2.image(bank_image)


model_load_state = st.text('Loading model...')
model = load_model()
preprocessor = Preprocessor()
image = Image.open("./data/Picture1.png")
model_load_state.text("All Loaded! (using st.cache)")

st.title('')
st.subheader('🤖 모델 개념도')
st.image(
    image,
    caption='LSTM model'
)

st.title('')
st.subheader('\n😎 입금기관 예측 테스트')
query = st.text_input(
    '아래에 계좌번호를 입력해주세요. 👇',
)
if query:
    data = preprocessor.preprocess([query])
    predictions = model.predict_top_k(data)

    if predictions:
        c1, c2 = st.columns(2)
        top_pred_bank = list(predictions.keys())[0]
        c1.write(predictions)
        show_image(top_pred_bank, c2)
