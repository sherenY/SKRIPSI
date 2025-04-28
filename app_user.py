import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="🎵 Prediksi & Rekomendasi Genre Musik", layout="centered")
st.title("🎵 Prediksi Genre & Rekomendasi Lagu Favorit 🎵")

model = load_model("fcnn_genre_classifier90.h5", compile=False)
scaler = joblib.load("scaler90.pkl")
encoder = joblib.load("encoder90.pkl")
data = joblib.load("7000SONGS_NEW2.pkl")

st.subheader("👆 Pilih 3 Lagu yang Kamu Suka")
all_filenames = data["filename"].tolist()

lagu1 = st.selectbox("Lagu ke-1️⃣", [""] + all_filenames, key="lagu1")
lagu2 = st.selectbox("Lagu ke-2️⃣", [""] + [f for f in all_filenames if f != lagu1], key="lagu2")
lagu3 = st.selectbox("Lagu ke-3️⃣", [""] + [f for f in all_filenames if f != lagu1 and f != lagu2], key="lagu3")

selected_filenames = [lagu for lagu in [lagu1, lagu2, lagu3] if lagu != ""]

jumlah_rekomendasi = st.slider("✅ Pilih Jumlah Lagu yang Ingin Direkomendasikan", min_value=5, max_value=30, value=15)

def get_song_recommendations(selected_songs, song_data, max_recommendations=15):
    feature_columns = song_data.columns[2:]
    song_features = song_data[feature_columns].values
    knn = NearestNeighbors(n_neighbors=min(len(song_data), max_recommendations + 10), metric='cosine')
    knn.fit(song_features)

    all_recommended_songs = []

    for song in selected_songs:
        song_index = song_data[song_data['filename'] == song].index[0]
        song_feature = song_features[song_index].reshape(1, -1)

        distances, indices = knn.kneighbors(song_feature)

        for i in range(len(indices[0])):
            song_name = song_data['filename'][indices[0][i]]
            if song_name not in selected_songs:
                all_recommended_songs.append(song_name)

    all_recommended_songs = list(dict.fromkeys(all_recommended_songs))

    return all_recommended_songs[:max_recommendations]


tombol_predict = st.button("🔍 PREDICT 🔍", use_container_width=True)

if tombol_predict:
    if len(selected_filenames) < 3:
        st.error("🚫 Maaf, Kamu perlu memilih 3 lagu!")
    else:
        sample = data[data["filename"].isin(selected_filenames)].reset_index(drop=True)
        features = sample.drop(columns=["filename", "genre"])
        features_numeric = features.select_dtypes(include=[np.number])
        X_scaled = scaler.transform(features_numeric)

        pred_probs = model.predict(X_scaled)
        pred_indices = np.argmax(pred_probs, axis=1)
        pred_labels = encoder.inverse_transform(pred_indices)

        #st.subheader("📈 Hasil Prediksi Genre per Lagu")
        #hasil = pd.DataFrame({
            #"Judul Lagu & Penyanyi": sample["filename"],
            #"Genre Sebenarnya": sample["genre"],
            #"Genre Prediksi": pred_labels
        #})
        #st.dataframe(hasil)

        st.subheader("🎯 Probabilitas Setiap Genre per Lagu")
        probs_df = pd.DataFrame(pred_probs, columns=encoder.classes_)
        probs_df.insert(0, "Judul Lagu dan Penyanyi", sample["filename"])
        probs_df.index = probs_df.index + 1

        st.dataframe(probs_df.style.format(precision=4))

        total_probs = np.sum(pred_probs, axis=0)
        genre_index_favorit = np.argmax(total_probs)
        genre_favorit = encoder.inverse_transform([genre_index_favorit])[0]

        st.subheader("🎧 Genre Favorit Kamu :")
        st.success(f"🥰 Berdasarkan dari 3 lagu yang kamu pilih, Genre favorit kamu adalah **{genre_favorit.upper()}** 🎶")

        st.subheader("🎶 Rekomendasi Lagu Buat Kamu")
        rekomendasi_knn = get_song_recommendations(selected_filenames, data, max_recommendations=jumlah_rekomendasi)
        
        rekomendasi_df = pd.DataFrame(rekomendasi_knn, columns=["Judul Lagu dan Penyanyi"])
        rekomendasi_df.index = rekomendasi_df.index + 1

        st.dataframe(rekomendasi_df)
