"""
streamlit_app.py
================
Aplikasi web untuk inferencing model credit scoring.

Aplikasi ini memuat pipeline model terbaik yang telah dilatih, menerima data
nasabah melalui form, lalu menampilkan prediksi kelas credit score beserta
probabilitasnya. Karena seluruh pembersihan dan preprocessing sudah berada di
dalam pipeline, aplikasi cukup mengirim data mentah ke model.

Terdapat juga tiga tombol test case yang mewakili tiap kelas, yaitu Good,
Standard, dan Poor, agar pengujian hasil deployment menjadi mudah.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Import class cleaner agar pipeline dapat di unpickle dengan benar.
from preprocess import CreditCleaner  # noqa: F401

# imblearn hanya diperlukan jika model dilatih ulang melalui pipeline.py yang
# memakai SMOTE. Import dibungkus try agar aplikasi tetap jalan dengan model
# cadangan berbasis scikit-learn biasa.
try:
    import imblearn  # noqa: F401
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_credit_score_model.joblib"
METADATA_PATH = BASE_DIR / "model_metadata.json"

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician",
    "Scientist", "Teacher", "Writer",
]
CREDIT_MIX = ["Bad", "Standard", "Good"]
PAY_MIN = ["No", "Yes"]
PAYMENT_BEHAVIOUR = [
    "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
    "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
    "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
]

# Tiga test case yang mewakili masing masing kelas target.
TEST_CASES = {
    "Good": {
        "Age": 49, "Annual_Income": 19462.93, "Monthly_Inhand_Salary": 1496.15,
        "Num_Bank_Accounts": 1, "Num_Credit_Card": 1, "Interest_Rate": 8.0,
        "Num_of_Loan": 4, "Delay_from_due_date": 9, "Num_of_Delayed_Payment": 13,
        "Changed_Credit_Limit": 5.51, "Num_Credit_Inquiries": 7,
        "Outstanding_Debt": 160.3, "Credit_Utilization_Ratio": 26.49,
        "Total_EMI_per_month": 160.82, "Amount_invested_monthly": 80.0,
        "Monthly_Balance": 337.09, "Credit_History_Years": 32, "Credit_History_Months": 4,
        "Occupation": "Engineer", "Credit_Mix": "Good", "Payment_of_Min_Amount": "No",
        "Payment_Behaviour": "Low_spent_Small_value_payments",
    },
    "Standard": {
        "Age": 34, "Annual_Income": 99425.4, "Monthly_Inhand_Salary": 8157.45,
        "Num_Bank_Accounts": 8, "Num_Credit_Card": 5, "Interest_Rate": 18.0,
        "Num_of_Loan": 4, "Delay_from_due_date": 13, "Num_of_Delayed_Payment": 12,
        "Changed_Credit_Limit": 10.6, "Num_Credit_Inquiries": 8,
        "Outstanding_Debt": 892.49, "Credit_Utilization_Ratio": 44.09,
        "Total_EMI_per_month": 172.53, "Amount_invested_monthly": 89.87,
        "Monthly_Balance": 793.34, "Credit_History_Years": 16, "Credit_History_Months": 5,
        "Occupation": "Media_Manager", "Credit_Mix": "Standard", "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "High_spent_Large_value_payments",
    },
    "Poor": {
        "Age": 39, "Annual_Income": 78443.48, "Monthly_Inhand_Salary": 6358.96,
        "Num_Bank_Accounts": 7, "Num_Credit_Card": 5, "Interest_Rate": 23.0,
        "Num_of_Loan": 4, "Delay_from_due_date": 39, "Num_of_Delayed_Payment": 19,
        "Changed_Credit_Limit": 6.37, "Num_Credit_Inquiries": 6,
        "Outstanding_Debt": 1527.77, "Credit_Utilization_Ratio": 29.24,
        "Total_EMI_per_month": 177.39, "Amount_invested_monthly": 463.74,
        "Monthly_Balance": 274.77, "Credit_History_Years": 15, "Credit_History_Months": 6,
        "Occupation": "Entrepreneur", "Credit_Mix": "Bad", "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "Low_spent_Medium_value_payments",
    },
}

DEFAULTS = TEST_CASES["Standard"]
SCORE_COLOR = {"Poor": "#e74c3c", "Standard": "#f39c12", "Good": "#27ae60"}
SCORE_NOTE = {
    "Poor": "Skor kredit tergolong buruk. Nasabah berisiko tinggi sehingga perlu verifikasi tambahan atau jaminan.",
    "Standard": "Skor kredit tergolong standar. Nasabah berisiko menengah dan dapat diproses dengan evaluasi biasa.",
    "Good": "Skor kredit tergolong baik. Nasabah berisiko rendah dan layak direkomendasikan untuk disetujui.",
}


st.set_page_config(page_title="Credit Score Prediction", page_icon="💳", layout="wide")


@st.cache_resource
def load_model_and_metadata():
    if not MODEL_PATH.exists():
        st.error("Model belum ditemukan. Jalankan pipeline.py terlebih dahulu.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as file:
            metadata = json.load(file)
    return model, metadata


def init_state():
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)


def apply_test_case(case_name):
    for key, value in TEST_CASES[case_name].items():
        st.session_state[key] = value


def build_input_row():
    """Menyusun satu baris data mentah dari nilai form sesuai skema dataset."""
    years = st.session_state["Credit_History_Years"]
    months = st.session_state["Credit_History_Months"]
    return {
        "Age": st.session_state["Age"],
        "Annual_Income": st.session_state["Annual_Income"],
        "Monthly_Inhand_Salary": st.session_state["Monthly_Inhand_Salary"],
        "Num_Bank_Accounts": st.session_state["Num_Bank_Accounts"],
        "Num_Credit_Card": st.session_state["Num_Credit_Card"],
        "Interest_Rate": st.session_state["Interest_Rate"],
        "Num_of_Loan": st.session_state["Num_of_Loan"],
        "Delay_from_due_date": st.session_state["Delay_from_due_date"],
        "Num_of_Delayed_Payment": st.session_state["Num_of_Delayed_Payment"],
        "Changed_Credit_Limit": st.session_state["Changed_Credit_Limit"],
        "Num_Credit_Inquiries": st.session_state["Num_Credit_Inquiries"],
        "Outstanding_Debt": st.session_state["Outstanding_Debt"],
        "Credit_Utilization_Ratio": st.session_state["Credit_Utilization_Ratio"],
        "Total_EMI_per_month": st.session_state["Total_EMI_per_month"],
        "Amount_invested_monthly": st.session_state["Amount_invested_monthly"],
        "Monthly_Balance": st.session_state["Monthly_Balance"],
        "Credit_History_Age": f"{years} Years and {months} Months",
        "Occupation": st.session_state["Occupation"],
        "Credit_Mix": st.session_state["Credit_Mix"],
        "Payment_of_Min_Amount": st.session_state["Payment_of_Min_Amount"],
        "Payment_Behaviour": st.session_state["Payment_Behaviour"],
    }


def render_sidebar(metadata):
    with st.sidebar:
        st.markdown("### Informasi Model")
        if metadata:
            st.write(f"Model terbaik: {metadata.get('best_experiment', '-')}")
            st.write(f"Metrik utama: {metadata.get('primary_metric', '-')}")
            results = metadata.get("results", [])
            if results:
                best = results[0]
                st.markdown("### Performa pada Data Test")
                st.metric("F1 Macro", f"{best.get('test_f1_macro', 0):.4f}")
                st.metric("Accuracy", f"{best.get('test_accuracy', 0):.4f}")
                st.metric("Balanced Accuracy", f"{best.get('test_balanced_accuracy', 0):.4f}")
        st.markdown("---")
        st.markdown("### Test Case per Kelas")
        st.caption("Tekan salah satu untuk mengisi form otomatis.")
        if st.button("Contoh kelas Good"):
            apply_test_case("Good")
        if st.button("Contoh kelas Standard"):
            apply_test_case("Standard")
        if st.button("Contoh kelas Poor"):
            apply_test_case("Poor")


def render_form():
    st.markdown("#### Data Keuangan Nasabah")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Age", 14, 100, key="Age")
        st.number_input("Annual Income", 0.0, 1e7, key="Annual_Income", step=1000.0)
        st.number_input("Monthly Inhand Salary", 0.0, 1e6, key="Monthly_Inhand_Salary", step=100.0)
        st.number_input("Num Bank Accounts", 0, 30, key="Num_Bank_Accounts")
        st.number_input("Num Credit Card", 0, 30, key="Num_Credit_Card")
        st.number_input("Interest Rate", 0.0, 100.0, key="Interest_Rate", step=1.0)
    with c2:
        st.number_input("Num of Loan", 0, 20, key="Num_of_Loan")
        st.number_input("Delay from Due Date", -10, 200, key="Delay_from_due_date")
        st.number_input("Num of Delayed Payment", 0, 100, key="Num_of_Delayed_Payment")
        st.number_input("Changed Credit Limit", 0.0, 100.0, key="Changed_Credit_Limit", step=0.5)
        st.number_input("Num Credit Inquiries", 0, 100, key="Num_Credit_Inquiries")
        st.number_input("Outstanding Debt", 0.0, 1e6, key="Outstanding_Debt", step=100.0)
    with c3:
        st.number_input("Credit Utilization Ratio", 0.0, 100.0, key="Credit_Utilization_Ratio", step=1.0)
        st.number_input("Total EMI per Month", 0.0, 1e5, key="Total_EMI_per_month", step=50.0)
        st.number_input("Amount Invested Monthly", 0.0, 1e5, key="Amount_invested_monthly", step=10.0)
        st.number_input("Monthly Balance", 0.0, 1e6, key="Monthly_Balance", step=50.0)
        st.number_input("Credit History (Years)", 0, 60, key="Credit_History_Years")
        st.number_input("Credit History (Months)", 0, 11, key="Credit_History_Months")

    st.markdown("#### Informasi Kategorik")
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        st.selectbox("Occupation", OCCUPATIONS, key="Occupation")
    with c5:
        st.selectbox("Credit Mix", CREDIT_MIX, key="Credit_Mix")
    with c6:
        st.selectbox("Payment of Min Amount", PAY_MIN, key="Payment_of_Min_Amount")
    with c7:
        st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR, key="Payment_Behaviour")


def render_result(model, input_row):
    df = pd.DataFrame([input_row])
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    classes = list(model.classes_)

    color = SCORE_COLOR.get(prediction, "#95a5a6")
    st.markdown(
        f"<div style='padding:18px;border-radius:10px;background:{color};color:white;"
        f"text-align:center;font-size:28px;font-weight:bold;'>"
        f"Prediksi Credit Score: {prediction}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Probabilitas per Kelas")
    cols = st.columns(len(classes))
    for col, cls in zip(cols, classes):
        prob = probabilities[classes.index(cls)]
        col.metric(cls, f"{prob * 100:.1f}%")

    st.info(SCORE_NOTE.get(prediction, ""))
    with st.expander("Lihat data input"):
        st.dataframe(df.T.rename(columns={0: "value"}))


def main():
    init_state()
    model, metadata = load_model_and_metadata()

    st.markdown(
        "<h1 style='text-align:center;'>Credit Score Prediction System</h1>"
        "<p style='text-align:center;color:#7f8c8d;'>Menilai performa kredit nasabah "
        "menggunakan machine learning</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    render_sidebar(metadata)
    render_form()

    st.markdown("---")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        predict = st.button("Prediksi Credit Score", type="primary", use_container_width=True)

    if predict:
        with st.spinner("Menghitung prediksi ..."):
            render_result(model, build_input_row())


if __name__ == "__main__":
    main()
