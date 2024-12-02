import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Asisten Kesehatan Firedito",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Mendapatkan direktori kerja dari main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Memuat model yang telah disimpan
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.pkl', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.pkl', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.pkl', 'rb'))

# Sidebar untuk navigasi
with st.sidebar:
    selected = option_menu('Sistem Prediksi Multi Penyakit',

                           ['Prediksi Diabetes',
                            'Prediksi Penyakit Jantung',
                            'Prediksi Parkinson'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Halaman Prediksi Diabetes
if selected == 'Prediksi Diabetes':

    st.title('Prediksi Diabetes menggunakan ML')

    # Input data pengguna
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Jumlah Kehamilan', value='0')
    with col2:
        Glucose = st.text_input('Tingkat Glukosa', value='0')
    with col3:
        BloodPressure = st.text_input('Nilai Tekanan Darah', value='0')
    with col1:
        SkinThickness = st.text_input('Nilai Ketebalan Kulit', value='0')
    with col2:
        Insulin = st.text_input('Tingkat Insulin', value='0')
    with col3:
        BMI = st.text_input('Nilai BMI', value='0')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Nilai Fungsi Diabetes Pedigree', value='0')
    with col2:
        Age = st.text_input('Usia Orang', value='0')

    # Default nilai untuk user_input
    user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                  float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]

    diab_diagnosis = ''

    # Tombol untuk melakukan prediksi
    if st.button('Hasil Tes Diabetes'):
        diab_prediction = diabetes_model.predict([user_input])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'Orang tersebut mengidap diabetes'
        else:
            diab_diagnosis = 'Orang tersebut tidak mengidap diabetes'
        st.success(diab_diagnosis)

    # Grafik interaktif
    if st.button("Tampilkan Grafik Diabetes"):
        df = pd.DataFrame({'Fitur': ['Kehamilan', 'Glukosa', 'Tekanan Darah', 'Ketebalan Kulit', 'Insulin', 'BMI', 
                                     'Fungsi Diabetes Pedigree', 'Usia'],
                           'Nilai': user_input})
        fig = px.bar(df, x='Fitur', y='Nilai', title="Visualisasi Data Input Diabetes",
                     labels={'Nilai': 'Skor', 'Fitur': 'Kategori'})
        st.plotly_chart(fig)


# Halaman Prediksi Penyakit Jantung
if selected == 'Prediksi Penyakit Jantung':

    # Judul halaman
    st.title('Prediksi Penyakit Jantung menggunakan ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Usia')

    with col2:
        sex = st.selectbox('Jenis Kelamin', 
                           options=[0, 1], 
                           format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')

    with col3:
        cp = st.selectbox('Jenis Nyeri Dada', 
                          options=[0, 1, 2, 3],
                          format_func=lambda x: {
                              0: 'Typical Angina',
                              1: 'Atypical Angina',
                              2: 'Non-Anginal Pain',
                              3: 'Asymptomatic'
                          }[x])

    with col1:
        trestbps = st.text_input('Tekanan Darah Istirahat')

    with col2:
        chol = st.text_input('Kadar Kolesterol Serum dalam mg/dl')

    with col3:
        fbs = st.text_input('Gula Darah Puasa > 120 mg/dl')

    with col1:
        restecg = st.selectbox('Hasil Elektrokardiogram Istirahat',
                               options=[0, 1, 2],
                               format_func=lambda x: {
                                   0: 'Normal',
                                   1: 'Kelainan gelombang ST-T',
                                   2: 'Kemungkinan atau pasti hipertrofi ventrikel kiri'
                               }[x])

    with col2:
        thalach = st.text_input('Detak Jantung Maksimum yang Dicapai')

    with col3:
        exang = st.selectbox('Angina yang Diinduksi Olahraga', 
                             options=[0, 1],
                             format_func=lambda x: 'Ya' if x == 1 else 'Tidak')

    with col1:
        oldpeak = st.text_input('Depresi ST yang Diinduksi Olahraga')

    with col2:
        slope = st.selectbox('Kemiringan Segmen ST Puncak Olahraga', 
                             options=[0, 1, 2],
                             format_func=lambda x: {
                                 0: 'Menurun',
                                 1: 'Datar',
                                 2: 'Meningkat'
                             }[x])

    with col3:
        ca = st.text_input('Jumlah Pembuluh Utama yang Diwarnai dengan Fluoroskopi')

    with col1:
        thal = st.selectbox('Hasil Tes Thalium', 
                            options=[0, 1, 2],
                            format_func=lambda x: {
                                0: 'Normal',
                                1: 'Cacat Tetap',
                                2: 'Cacat Reversibel'
                            }[x])

    # Kode untuk Prediksi
    heart_diagnosis = ''

    # Membuat tombol untuk Prediksi
    if st.button('Hasil Tes Penyakit Jantung'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) if isinstance(x, str) else x for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'Orang tersebut mengidap penyakit jantung'
        else:
            heart_diagnosis = 'Orang tersebut tidak mengidap penyakit jantung'

    st.success(heart_diagnosis)

    

# Halaman Prediksi Parkinson
if selected == 'Prediksi Parkinson':

    # Judul halaman
    st.title("Prediksi Penyakit Parkinson menggunakan ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # Kode untuk Prediksi
    parkinsons_diagnosis = ''

    # Membuat tombol untuk Prediksi    
    if st.button('Hasil Tes Parkinson'):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = 'Orang tersebut mengidap penyakit Parkinson'
        else:
            parkinsons_diagnosis = 'Orang tersebut tidak mengidap penyakit Parkinson'

    st.success(parkinsons_diagnosis)
