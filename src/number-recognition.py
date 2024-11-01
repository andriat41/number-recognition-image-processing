from PIL import Image, ImageOps
import numpy as np
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import cv2

# Menggunakan morphological operations
def apply_morphology(img_array):
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(img_array, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded

# Menggunakan threshold
def apply_threshold(img):
    return img.point(lambda p: 0 if p < 128 else 1, '1')

# Menggunakan median blur
def apply_noise_reduction(img):
    # Konversi gambar PIL ke array numpy
    img_array = np.array(img, dtype=np.uint8) * 255  # Pastikan dalam format grayscale 0-255
    # Terapkan median blur untuk menghilangkan noise
    img_array = cv2.medianBlur(img_array, 3)
    return img_array

# Menggunakan centering
def center_digit(img_array):
    # Ambil bounding box angka
    rows = np.any(img_array, axis=1)
    cols = np.any(img_array, axis=0)
    
    # Cek apakah gambar memiliki konten (tidak kosong)
    if not np.any(rows) or not np.any(cols):
        # Jika gambar kosong, kembalikan gambar kosong dengan ukuran target
        return np.zeros((6, 9), dtype=np.uint8)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped_img = img_array[ymin:ymax+1, xmin:xmax+1]
    
    # Letakkan angka di tengah dengan padding
    padded_img = np.pad(cropped_img, ((2, 2), (2, 2)), mode='constant', constant_values=0)
    return cv2.resize(padded_img, (6, 9))  # Ubah ukuran ke target

# Menggunakan deskewing
def deskew(img_array):
    coords = np.column_stack(np.where(img_array > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# Menggunakan invert
def invert(img):
    return ImageOps.invert(img.convert("L"))

# Memuat, mengubah ukuran, dan menghitung jumlah piksel baris/kolom
def process_image(img_path, size=(6, 9)):
    img = Image.open(img_path)
    img = invert(img)
    img = img.convert("1")

    # Terapkan threshold
    # thresholding tidak membantu untuk meningkatkan akurasi
    # img = apply_threshold(img)
    img_array = np.array(img, dtype=np.uint8) * 255  # Pastikan format grayscale 0-255
    
    # Terapkan deskewing
    # deskew sangat membantu untuk meningkatkan akurasi
    img_array = deskew(img_array)

    # Terapkan noise reduction dan operasi morfologi
    # noise reduction tidak membantu untuk meningkatkan akurasi
    # img_array = apply_noise_reduction(img_array)
    img_array = apply_morphology(img_array)
    
    # Pusatkan angka dan ubah ukuran
    # centering tidak membantu untuk meningkatkan akurasi
    # img_array = center_digit(img_array)
    
    # Pastikan ukuran output img_array adalah (6, 9)
    img_array = cv2.resize(img_array, size)
    
    row_sums = np.sum(img_array, axis=1)  # Harus berukuran 10
    col_sums = np.sum(img_array, axis=0)  # Harus berukuran 15
    
    return row_sums, col_sums

# Pelatihan: Membuat pola rata-rata untuk setiap digit
def train_patterns(train_folder, size=(6, 9), samples=5):
    patterns = {}
    for digit in [0, 1, 2, 5, 6]:
        digit_folder = os.path.join(train_folder, str(digit))
        row_patterns, col_patterns = [], []
        for img_name in os.listdir(digit_folder)[:samples]:  # Mengambil 5 gambar per digit
            img_path = os.path.join(digit_folder, img_name)
            row_sums, col_sums = process_image(img_path, size)
            row_patterns.append(row_sums)
            col_patterns.append(col_sums)
        avg_row = np.mean(row_patterns, axis=0)
        avg_col = np.mean(col_patterns, axis=0)
        patterns[digit] = (avg_row, avg_col)
        # Menampilkan hasil pelatihan untuk setiap digit
        print(f"Pola pelatihan untuk digit {digit}:")
        print(f"  Rata-rata baris: {avg_row}")
        print(f"  Rata-rata kolom: {avg_col}\n")
    return patterns

# Pengujian: Mengklasifikasikan gambar berdasarkan pola pelatihan
def classify_image(img_path, patterns, size=(6, 9)):
    row_sums, col_sums = process_image(img_path, size)
    min_distance = float('inf')
    best_match = None
    for digit, (avg_row, avg_col) in patterns.items():
        row_diff = np.mean(np.abs(row_sums - avg_row))
        col_diff = np.mean(np.abs(col_sums - avg_col))
        total_diff = row_diff + col_diff
        if total_diff < min_distance:
            min_distance = total_diff
            best_match = digit
    return best_match

# Verifikasi: Menghitung akurasi pada set pengujian
def evaluate_accuracy(test_folder, patterns, size=(6, 9)):
    actual_labels = []
    predicted_labels = []
    confusion_matrix = {digit: {pred: 0 for pred in [0, 1, 2, 5, 6]} for digit in [0, 1, 2, 5, 6]}
    
    for digit in [0, 1, 2, 5, 6]:
        digit_folder = os.path.join(test_folder, str(digit))
        for img_name in os.listdir(digit_folder):
            img_path = os.path.join(digit_folder, img_name)
            prediction = classify_image(img_path, patterns, size)
            actual_labels.append(digit)
            predicted_labels.append(prediction)
            
            # Update confusion matrix
            confusion_matrix[digit][prediction] += 1
    
    accuracy = accuracy_score(actual_labels, predicted_labels)
    
    # Convert confusion matrix to a DataFrame for easier display
    confusion_df = pd.DataFrame(confusion_matrix).T
    confusion_df.index.name = "Angka yang seharusnya"
    confusion_df.columns.name = "Hasil prediksi"
    print("\nConfusion Matrix:")
    print(confusion_df)
    
    return accuracy, confusion_df

# Fungsi Utama
if __name__ == "__main__":
    train_folder = "data/training"  # Folder dengan gambar pelatihan yang diorganisir per digit
    test_folder = "data/testing"    # Folder dengan gambar pengujian yang diorganisir per digit
    image_size = (6, 9)
    
    # Pelatihan
    print("Melakukan pelatihan model...\n")
    patterns = train_patterns(train_folder, size=image_size, samples=5)
    
    # Evaluasi
    print("Menguji model...\n")
    accuracy, confusion_df = evaluate_accuracy(test_folder, patterns, size=image_size)
    
    # Akurasi akhir
    print(f"\nAkurasi Model: {accuracy * 100:.2f}%")
