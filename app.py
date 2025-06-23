import streamlit as st
import numpy as np
import os
import pickle
from scipy.io import wavfile
from scipy.fftpack import dct
from collections import Counter
import math

# ==== Load model ====
@st.cache_data
def load_model(path='knn_model.pkl'):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

# ==== Ekstraksi fitur ====
def extract_time_domain_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        np.percentile(signal, 25),
        np.percentile(signal, 75),
        np.sum(np.abs(np.diff(signal)))
    ]

def extract_frequency_domain_features(signal, sample_rate):
    fft = np.fft.fft(signal)
    fft = np.abs(fft[:len(fft)//2])
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(fft)]
    return [
        np.mean(fft),
        np.std(fft),
        np.max(fft),
        np.sum(fft),
        np.argmax(fft),
        freqs[np.argmax(fft)]
    ]

def extract_mfcc_manual(signal, sample_rate, num_coeffs=12):
    signal = signal[:int(0.025 * sample_rate)]
    if len(signal) < 1:
        return [0] * num_coeffs
    spectrum = np.fft.fft(signal)
    power_spectrum = np.abs(spectrum[:len(spectrum)//2]) ** 2
    power_spectrum = np.log10(power_spectrum + 1e-10)
    mfcc = dct(power_spectrum, type=2, norm='ortho')[:num_coeffs]
    return mfcc.tolist()

# ==== Normalisasi ====
def normalize_single(sample, min_vals, max_vals):
    sample = np.array(sample)
    denom = max_vals - min_vals
    denom[denom == 0] = 1e-12
    return ((sample - min_vals) / denom).tolist()

# ==== KNN manual ====
def euclidean_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def knn_predict(X_train, y_train, test_sample, k=3):
    distances = [(euclidean_distance(test_sample, train_sample), label)
                 for train_sample, label in zip(X_train, y_train)]
    distances.sort(key=lambda x: x[0])
    top_k = [label for _, label in distances[:k]]
    return Counter(top_k).most_common(1)[0][0]

# ==== GUI Start ====
st.set_page_config(page_title="Gunshot Classifier", layout="centered")
st.title("🔫 Gunshot Audio Classifier (KNN Manual)")
st.markdown("Upload file `.wav` suara tembakan. Sistem akan memproses dan memprediksi jenis senjatanya.")

uploaded_file = st.file_uploader("🎵 Upload file suara (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("🔍 Proses dan Prediksi"):
        st.info("🚀 Memproses file...")

        model = load_model()
        X_train = model["X_train"]
        y_train = model["y_train"]
        min_vals = model["min_vals"]
        max_vals = model["max_vals"]
        k = model.get("k", 3)
        accuracy = model.get("accuracy", None)

        try:
            with open("temp_uploaded.wav", "wb") as f:
                f.write(uploaded_file.read())

            sample_rate, signal = wavfile.read("temp_uploaded.wav")

            if signal.dtype == np.int16:
                signal = signal.astype(np.float32) / 32768.0
            if len(signal.shape) == 2:
                signal = signal[:, 0]
            if len(signal) < 100:
                st.error("❌ File terlalu pendek atau kosong!")
                st.stop()

            # Ekstraksi fitur manual
            time_feat = extract_time_domain_features(signal)
            freq_feat = extract_frequency_domain_features(signal, sample_rate)
            mfcc_feat = extract_mfcc_manual(signal, sample_rate, num_coeffs=len(max_vals) - len(time_feat) - len(freq_feat))
            combined_feat = time_feat + freq_feat + mfcc_feat

            # Debug panjang fitur
            if len(combined_feat) != len(min_vals):
                st.error(f"❌ Panjang fitur input ({len(combined_feat)}) tidak cocok dengan model ({len(min_vals)})")
                st.stop()

            norm_feat = normalize_single(combined_feat, min_vals, max_vals)
            pred_label = knn_predict(X_train, y_train, norm_feat, k=k)

            st.success(f"🎯 Prediksi Senjata: **{pred_label}**")
            if accuracy:
                st.info(f"📊 Akurasi model training: **{accuracy:.2f}%**")

            # Debug preview fitur
            with st.expander("📈 Lihat fitur input"):
                st.write("Fitur Time Domain:", time_feat)
                st.write("Fitur Frequency Domain:", freq_feat)
                st.write("Fitur MFCC:", mfcc_feat)

        except Exception as e:
            st.error(f"❌ Gagal memproses file: {e}")
