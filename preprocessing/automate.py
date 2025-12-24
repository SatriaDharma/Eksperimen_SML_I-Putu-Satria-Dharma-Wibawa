import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing():
    print("Memulai Otomatisasi Preprocessing")
    
    # 1. Tentukan Path 
    input_path = 'heart_raw/heart.csv'
    output_dir = 'preprocessing/heart_preprocessing'
    
    # Cek apakah file input ada
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return

    # 2. Memuat Dataset
    df = pd.read_csv(input_path)
    print(f"Data asli dimuat. Ukuran: {df.shape}")

    # Salin data untuk preprocessing
    df_clean = df.copy()

    # 1. Menghapus atau Menangani Data Kosong (Missing Values)
    # Dalam dataset ini, tidak ada missing values, jadi kita lewati langkah ini.

    # 2. Menghapus Duplikat
    df_clean = df_clean.drop_duplicates()

    # 3. Normalisasi atau Standarisasi Fitur Numerik
    scaler = StandardScaler()
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])

    # 4. Penanganan Outlier
    # Tidak dilakukan, karena data outlier sedikit, dan mungkin penting untuk analisis medis.

    # 5. Encoding Data Kategorikal
    # Semua fitur sudah dalam format numerik, jadi tidak perlu encoding tambahan.

    # 6. Binning (Pengelompokan Data)
    # Tidak dilakukan, karena tidak diperlukan untuk analisis ini.

    # 7. Menyimpan Data Hasil Preprocessing
    os.makedirs(output_dir, exist_ok=True)
    df_clean.to_csv(f"{output_dir}/heart_cleaned.csv", index=False)

    print(f"Preprocessing selesai. Data disimpan di {output_dir}/heart_cleaned.csv")
    print(f"Ukuran Dataset Setelah Preprocessing: {df_clean.shape}")
    print("Otomatisasi Selesai")

if __name__ == "__main__":
    run_preprocessing()