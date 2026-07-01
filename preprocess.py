"""
preprocess.py
=============
Tahap preprocessing untuk pipeline credit scoring.

Modul ini berisi dua class berbasis OOP.
1. CreditCleaner, sebuah transformer scikit-learn yang membersihkan kolom mentah
   yang berantakan menjadi fitur numerik dan kategorik yang bersih.
2. Preprocessor, class yang memisahkan data menjadi train dan test serta
   membangun ColumnTransformer untuk imputasi, scaling, dan encoding.

CreditCleaner sengaja dibuat sebagai transformer agar bisa ikut tersimpan di
dalam pipeline model. Dengan begitu, proses inferencing pada Streamlit cukup
memberikan data mentah dan seluruh pembersihan dilakukan otomatis.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# ----------------------------------------------------------------------------- 
# Definisi fitur. Dijaga sederhana sesuai kebutuhan modelling.
# ----------------------------------------------------------------------------- 
TARGET_COL = "Credit_Score"

# Kolom numerik mentah yang nilainya perlu dibersihkan dari karakter sampah.
RAW_NUMERIC_FEATURES = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

# Fitur numerik turunan dari teks Credit_History_Age.
DERIVED_NUMERIC_FEATURES = ["Credit_History_Age_Months"]

NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + DERIVED_NUMERIC_FEATURES

CATEGORICAL_FEATURES = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

# Urutan kolom akhir yang dihasilkan CreditCleaner dan diterima ColumnTransformer.
MODEL_FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Penanda nilai sampah yang sering muncul di dataset dan harus dianggap kosong.
GARBAGE_MARKERS = {
    "", "_", "__", "___", "____", "_______", "!@9#%8",
    "NM", "nan", "NaN", "None", "#F%$D@*&8", "__10000__",
}


class CreditCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer untuk membersihkan data mentah credit scoring.

    Tugas utama.
    1. Mengubah kolom numerik yang tersimpan sebagai teks kotor menjadi angka.
    2. Mengubah teks Credit_History_Age menjadi total bulan.
    3. Mengubah nilai yang mustahil, misalnya umur negatif, menjadi kosong.
    4. Membersihkan kolom kategorik dari penanda sampah.
    Hasil akhirnya adalah DataFrame dengan kolom sesuai MODEL_FEATURE_ORDER.
    """

    def fit(self, X, y=None):
        return self

    def _to_number(self, series):
        cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        cleaned = cleaned.replace("", np.nan)
        return pd.to_numeric(cleaned, errors="coerce")

    def _parse_history_age(self, value):
        if pd.isna(value):
            return np.nan
        text = str(value)
        year = re.search(r"(\d+)\s*Year", text)
        month = re.search(r"(\d+)\s*Month", text)
        if not year and not month:
            return np.nan
        total_years = int(year.group(1)) if year else 0
        total_months = int(month.group(1)) if month else 0
        return total_years * 12 + total_months

    def transform(self, X):
        data = X.copy()

        # 1. Bersihkan kolom numerik mentah.
        for col in RAW_NUMERIC_FEATURES:
            if col in data.columns:
                data[col] = self._to_number(data[col])

        # 2. Buat fitur lama riwayat kredit dalam satuan bulan.
        if "Credit_History_Age" in data.columns:
            data["Credit_History_Age_Months"] = data["Credit_History_Age"].apply(
                self._parse_history_age
            )
        else:
            data["Credit_History_Age_Months"] = np.nan

        # 3. Ganti nilai yang mustahil menjadi kosong agar diimputasi nanti.
        if "Age" in data.columns:
            data.loc[(data["Age"] < 14) | (data["Age"] > 100), "Age"] = np.nan
        non_negative_cols = [
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Num_Credit_Inquiries",
            "Interest_Rate",
        ]
        for col in non_negative_cols:
            if col in data.columns:
                data.loc[data[col] < 0, col] = np.nan

        # 4. Bersihkan kolom kategorik dari penanda sampah.
        for col in CATEGORICAL_FEATURES:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip()
                data[col] = data[col].where(~data[col].isin(GARBAGE_MARKERS), np.nan)

        return data.reindex(columns=MODEL_FEATURE_ORDER)


class Preprocessor:
    """
    Class yang mengatur pemisahan data dan pembangunan transformer scikit-learn.

    Pembagian data memakai stratifikasi agar proporsi kelas pada train dan test
    tetap sama. Transformer numerik memakai imputasi median dan RobustScaler
    karena banyak outlier pada data finansial. Transformer kategorik memakai
    imputasi modus dan OneHotEncoder dengan handle_unknown ignore.
    """

    def __init__(
        self,
        target_col=TARGET_COL,
        output_dir="artifacts/processed",
        test_size=0.2,
        random_state=42,
    ):
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.train_path = self.output_dir / "train.csv"
        self.test_path = self.output_dir / "test.csv"

    def split_data(self, input_path):
        print("--- Tahap 2: Split Data ---")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(input_path)
        data = data.dropna(subset=[self.target_col]).copy()

        train_df, test_df = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=data[self.target_col],
        )

        train_df.to_csv(self.train_path, index=False)
        test_df.to_csv(self.test_path, index=False)
        print(f"Train disimpan ke {self.train_path}, shape {train_df.shape}")
        print(f"Test disimpan ke {self.test_path}, shape {test_df.shape}")
        return self.train_path, self.test_path

    def load_split(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        x_train = train_df.drop(columns=[self.target_col])
        y_train = train_df[self.target_col]
        x_test = test_df.drop(columns=[self.target_col])
        y_test = test_df[self.target_col]
        return x_train, x_test, y_train, y_test

    def build_transformer(self):
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )

        try:
            one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", one_hot),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, NUMERIC_FEATURES),
                ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ]
        )


if __name__ == "__main__":
    pre = Preprocessor()
    pre.split_data("artifacts/ingested/credit_score_data.csv")
