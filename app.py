import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Konfigurasi Halaman & Judul Dashboard
st.set_page_config(page_title="Dashboard Prediksi Kelulusan Mahasiswa", layout="wide")

st.title("🎓 Dashboard Prediksi Kelulusan Mahasiswa")
st.markdown("""
Aplikasi ini memprediksi status kelulusan mahasiswa berdasarkan data akademik dan kehadiran
menggunakan algoritma **Random Forest Classifier**.
""")

# 2. Memuat Dataset & Melatih Model
@st.cache_data
def load_data():
    df = pd.read_excel("dataset_kelulusan_mahasiswa_kelompok 2.xlsx")
    return df

df = load_data()

# Mengambil kolom terakhir sebagai target (sesuaikan jika kolom target berada di posisi lain)
target_col = df.columns[-1]
X = df.drop(columns=[target_col]).select_dtypes(include=["number"])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
akurasi = accuracy_score(y_test, model.predict(X_test))

# 3. Input di Sidebar Sesuai Fitur Dataset
st.sidebar.header("📝 Input Data Mahasiswa")
st.sidebar.write("Masukkan nilai fitur di bawah ini untuk melakukan prediksi:")

input_data = {}
for col in X.columns:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    input_data[col] = st.sidebar.number_input(
        label=f"{col}",
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

# 4. Tombol Prediksi & Hasil Prediksi Model
st.sidebar.markdown("---")
if st.sidebar.button("🔍 Prediksi Kelulusan", use_container_width=True):
    input_df = pd.DataFrame([input_data])
    prediksi = model.predict(input_df)[0]
    probabilitas = model.predict_proba(input_df)[0]

    st.subheader("🎯 Hasil Prediksi Model")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if str(prediksi).lower() in ["1", "lulus", "tepat waktu", "graduated"]:
            st.success(f"**Status Prediksi:** {prediksi}")
        else:
            st.error(f"**Status Prediksi:** {prediksi}")
    with col_res2:
        st.info(f"**Tingkat Keyakinan Model:** {max(probabilitas)*100:.2f}% | **Akurasi Uji:** {akurasi*100:.2f}%")
    st.markdown("---")

# 5. Minimal 2 Visualisasi Utama
st.subheader("📊 Visualisasi Data Mahasiswa")
col1, col2 = st.columns(2)

# Visualisasi 1: Distribusi Target Kelulusan
with col1:
    st.markdown("**1. Distribusi Status Kelulusan**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, palette="Blues", ax=ax1)
    ax1.set_xlabel("Status Kelulusan")
    ax1.set_ylabel("Jumlah Mahasiswa")
    st.pyplot(fig1)

# Visualisasi 2: Tingkat Kepentingan Fitur (Feature Importance)
with col2:
    st.markdown("**2. Faktor Paling Berpengaruh (Feature Importance)**")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    importances.plot(kind="barh", color="#2b7bba", ax=ax2)
    ax2.set_xlabel("Tingkat Pengaruh")
    st.pyplot(fig2)

# Pratinjau Dataset
with st.expander("Lihat Sampel Dataset"):
    st.dataframe(df.head(20), use_container_width=True)
