import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def run_preprocessing():
    print("Memulai Otomatisasi Preprocessing")
    
    # 1. Tentukan Path 
    input_path = 'heart_raw/heart.csv'
    output_dir = 'preprocessing/heart_preprocessing'
    output_file = f'{output_dir}/heart_cleaned.csv'
    
    # Cek apakah file input ada
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return

    # 2. Memuat Dataset
    df = pd.read_csv(input_path)
    print(f"Data asli dimuat. Ukuran: {df.shape}")

    # 3. Salin Data untuk Preprocessing
    df_clean = df.copy()

    # 4. Menghapus Duplikat
    df_clean = df.drop_duplicates()
    print(f"Duplikat dihapus. Ukuran sekarang: {df_clean.shape}")

    # 5. Standardisasi Fitur Numerik
    scaler = StandardScaler()
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    try:
        df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
        print("Standardisasi fitur numerik berhasil.")
    except Exception as e:
        print(f"Error saat standardisasi: {e}")
        return

    # 6. Menyimpan Hasil
    os.makedirs(output_dir, exist_ok=True)
    df_clean.to_csv(output_file, index=False)
    
    print(f"Berhasil! Data bersih disimpan di: {output_file}")
    print(f"Ukuran final: {df_clean.shape}")
    print("Otomatisasi Selesai")

if __name__ == "__main__":
    run_preprocessing()