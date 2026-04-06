# 🏦 입금기관 추천 모델

계좌번호 입력을 기반으로 입금기관(은행/증권사)을 예측하는  
**Multi-class Classification 모델**입니다.

- 총 대상 기관: **52개**
  - 은행: 25
  - 증권사: 27

---

## 📊 Model Overview

- Architecture: **LSTM (Multi-class Classification)**
- Performance:
  - **v5**: 91.35% (Test Accuracy)
  - **v3**: 89.81% (Test Accuracy)

---

## ⚙️ Environment

- Python: **3.10.10**

---

## 🚀 Demo

👉 https://bankprediction-rpmximxeacjclgdn45hbq8.streamlit.app/

![App Demo](./data/app.png)

---

## 📌 Model Release History

| Phase | Freezing Date | Deployment Date |
|------|--------------|----------------|
| 1st  | 2024-10-06   | 2024-11-30     |
| 2nd  | 2025-12-02   | 2026-04-03     |

---

## 📝 Notes

- 계좌번호 패턴 기반으로 금융기관을 분류하는 문제로, 시계열/문자열 패턴 학습을 위해 LSTM을 적용
- 지속적인 데이터 업데이트 및 모델 성능 개선 진행 중
