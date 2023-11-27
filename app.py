import streamlit as st
from PIL import Image
from st_keyup import st_keyup
from os import listdir

import config
from ModelLoader import ModelLoader
from Preprocessor import Preprocessor
from logger import app_logger

st.title('🏦 입금기관 예측 모델 테스트 💰')


def get_model_list():
    return [x for x in listdir(config.ModelConfig.MODEL_PATH) if x.endswith('keras')]


@st.cache_resource
def load_model(model_name):
    return ModelLoader(model_name)


if "query" not in st.session_state:
    st.session_state.query = ""


def show_image(top_pred_bank, c2):
    default_image_path = "./data/bank_images/"

    if '국민' in top_pred_bank:
        bank_image_path = "kb.jpeg"
    elif '신한' in top_pred_bank:
        bank_image_path = "shinhan.jpeg"
    elif '농협' in top_pred_bank:
        bank_image_path = "nh.png"
    elif '현대' in top_pred_bank:
        bank_image_path = "hyundai.jpeg"
    elif '키움' in top_pred_bank:
        bank_image_path = "kiwoom.png"
    elif '제주' in top_pred_bank:
        bank_image_path = "jeju.jpeg"
    else:
        bank_image_path = None

    if bank_image_path:
        bank_image = Image.open(default_image_path + bank_image_path)
        c2.image(bank_image)


model_names = sorted(get_model_list(), reverse=True)
model_name = model_names[0]
model_name = st.selectbox(
    'Choose model to use.',
    model_names
)

model_load_state = st.text('Loading model...')
model = load_model(model_name)
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
st.text('아래에 계좌번호를 입력해주세요. 👇 (숫자, 최대 14자)')
log_query = ''
query = st_keyup(
    '',
    key="0",
    label_visibility='collapsed',
    max_chars=14
)

if query:
    if not query.isdigit():
        app_logger.error('input query: ' + query)
        st.error('Please enter an bank account number!')
    data = preprocessor.preprocess([query])
    predictions = model.predict_top_k(data)

    if predictions:
        c1, c2 = st.columns(2)
        top_pred_bank = list(predictions.keys())[0]
        c1.write(predictions)
        show_image(top_pred_bank, c2)
        app_logger.info('input query: ' + query + ' predictions: ' + str(predictions))
