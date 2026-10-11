"""Streamlit dashboard for the Student Dropout and Academic Success dataset.

Run from this directory with: streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "studentDOA_prepared.csv"
MODEL_FILE = BASE_DIR / "studentDOA_model.joblib"
TARGET = "Target"

st.set_page_config(page_title="Student Dropout & Academic Success", layout="wide")
st.title("Student Dropout & Academic Success")
st.caption(
    "Dashboard pembelajaran untuk memahami data mahasiswa, melihat pola akademik, "
    "dan memprediksi status akhir dengan machine learning."
)


@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error("Data belum disiapkan. Jalankan `python Data_Prev.py` terlebih dahulu.")
        st.stop()
    data = pd.read_csv(DATA_FILE)
    data[TARGET] = data[TARGET].astype(str)
    return data


data = load_data()
if not MODEL_FILE.exists():
    st.error("Model belum dibuat. Jalankan `python Ml_model.py` terlebih dahulu.")
    st.stop()
artifacts = joblib.load(MODEL_FILE)
model = artifacts["model"]
X_test = artifacts["X_test"]
y_test = artifacts["y_test"]
test_prediction = artifacts["prediction"]

total, dropout_rate, graduate_rate, enrolled_rate = st.columns(4)
total.metric("Jumlah mahasiswa", f"{len(data):,}")
dropout_rate.metric("Dropout", f"{data[TARGET].eq('Dropout').mean():.1%}")
graduate_rate.metric("Graduate", f"{data[TARGET].eq('Graduate').mean():.1%}")
enrolled_rate.metric("Masih enrolled", f"{data[TARGET].eq('Enrolled').mean():.1%}")

tabs = st.tabs(["1 · Data", "2 · Ringkasan akademik", "3 · Machine Learning"])

with tabs[0]:
    st.subheader("Dataset mahasiswa")
    st.write(f"Struktur: **{data.shape[0]:,} baris × {data.shape[1]} kolom**")
    st.info(
        "Banyak atribut latar belakang disimpan sebagai kode angka. Pada model, "
        "atribut kategori tersebut diperlakukan sebagai kategori, bukan sebagai nilai berurutan."
    )
    st.markdown(
        "**Kelompok kolom:** pendaftaran dan program studi; latar belakang mahasiswa "
        "dan orang tua; kondisi biaya/beasiswa; capaian mata kuliah semester pertama "
        "dan kedua; indikator ekonomi; serta `Target` sebagai status akhir."
    )
    st.dataframe(data.head(100), use_container_width=True)
    with st.expander("Lihat nama kolom"):
        st.write(list(data.columns))

with tabs[1]:
    st.subheader("Pola status dan capaian akademik")
    left, right = st.columns(2)
    with left:
        st.markdown("**Jumlah mahasiswa per status**")
        st.bar_chart(data[TARGET].value_counts().rename("Mahasiswa"))
    with right:
        st.markdown("**Rata-rata mata kuliah yang lulus per semester**")
        pass_means = data.groupby(TARGET)[
            ["Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)"]
        ].mean()
        pass_means.columns = ["Semester 1", "Semester 2"]
        st.bar_chart(pass_means)

    age_status = data.groupby(TARGET)["Age at enrollment"].agg(["mean", "median"]).round(1)
    age_status.columns = ["Rata-rata usia masuk", "Median usia masuk"]
    st.markdown("**Usia saat pendaftaran menurut status**")
    st.dataframe(age_status, use_container_width=True)

with tabs[2]:
    st.subheader("Prediksi status mahasiswa")
    st.caption(
        "Model Random Forest dilatih oleh `Ml_model.py`. Evaluasi menggunakan "
        "pemisahan train/test 80/20 dengan stratifikasi status."
    )
    identifiers = data.index.tolist()
    selected_index = st.selectbox(
        "Pilih baris mahasiswa",
        identifiers,
        format_func=lambda index: f"Baris {index + 1} · status aktual: {data.loc[index, TARGET]}",
    )
    student = data.loc[[selected_index]].drop(columns=[TARGET])
    actual = data.loc[selected_index, TARGET]
    predicted = model.predict(student)[0]
    probabilities = pd.Series(
        model.predict_proba(student)[0], index=model.classes_, name="Probabilitas"
    ).sort_values(ascending=False)

    actual_col, predicted_col = st.columns(2)
    actual_col.metric("Status aktual", actual)
    predicted_col.metric("Prediksi model", predicted)
    st.markdown("**Skor probabilitas per status**")
    st.bar_chart(probabilities)
    st.dataframe(data.loc[[selected_index]], use_container_width=True)

    st.markdown("**Evaluasi pada data test**")
    accuracy, macro_f1 = st.columns(2)
    accuracy.metric("Accuracy", f"{artifacts['accuracy']:.3f}")
    report = artifacts["report"]
    macro_f1.metric("Macro F1", f"{report['macro avg']['f1-score']:.3f}")
    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, test_prediction, labels=labels)
    st.markdown("**Confusion matrix** (baris = status aktual, kolom = prediksi)")
    st.dataframe(
        pd.DataFrame(matrix, index=labels, columns=labels),
        use_container_width=True,
    )
    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True,
    )
    st.warning(
        "Prediksi adalah perkiraan dari pola pada data, bukan kepastian tentang "
        "hasil seorang mahasiswa. Performa dapat berbeda jika model atau pembagian data berubah."
    )

st.markdown("---")
st.caption("Data: studentDOA.csv · Target klasifikasi: Dropout, Enrolled, atau Graduate")
