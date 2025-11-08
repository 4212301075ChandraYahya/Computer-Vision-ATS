import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import LeaveOneOut, cross_val_score, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, f1_score
from sklearn.decomposition import PCA
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
import warnings
from tqdm import tqdm
import psutil
import os

try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)  
except Exception:
    pass  

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = r'D:\Campuzzz\5th semester\MK521  -  Machine Vision\ATS\data'
FILE_PATH = os.path.join(DATA_DIR, 'emnist-letters-train.csv')

OUTPUT_DIR = r'D:\Campuzzz\5th semester\MK521  -  Machine Vision\ATS\output'
LOG_FILE = os.path.join(OUTPUT_DIR, 'evaluation_log.txt')

SAMPLES_PER_CLASS = 500
TOTAL_SAMPLES = 26 * SAMPLES_PER_CLASS
IMAGE_SIZE = 28

RUN_TUNING_MODE = False  
RUN_LOOCV_FINAL = True   

HOG_PARAMS_FINAL = {'orientations': 10, 'ppc': (6, 6), 'cpb': (2, 2)} 
SVM_PARAMS_FINAL = {'C': 10.0, 'kernel': 'linear'} 

def write_log(message):
    print(message)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(message + '\n')

def plot_confusion_matrix(y_true, y_pred, kernel_name):
    labels_az = [chr(ord('A') + i) for i in range(26)] 
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels_az, yticklabels=labels_az,
                cbar=True, cbar_kws={'shrink': 0.8})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    title = f'Confusion Matrix (LOOCV + HOG + LinearSVC)'
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    filename = f'confusion_matrix_hog_linearsvc.png'
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300)
    write_log(f"\n[VISUALISASI] Confusion Matrix disimpan ke: {filepath}")
    plt.close()

def load_and_sample_data(file_path):
    write_log("Memuat dan melakukan sampling data seimbang...")
    df = pd.read_csv(file_path, header=None)
    X_full = df.iloc[:, 1:].values.astype('float32') / 255.0
    y_full = df.iloc[:, 0].values - 1

    X_sampled_list = []
    y_sampled_list = []

    for class_label in range(26):
        class_indices = np.where(y_full == class_label)[0]
        selected_indices = np.random.choice(class_indices, SAMPLES_PER_CLASS, replace=False)
        X_sampled_list.append(X_full[selected_indices])
        y_sampled_list.append(y_full[selected_indices])

    X_sampled = np.concatenate(X_sampled_list, axis=0)
    y_sampled = np.concatenate(y_sampled_list, axis=0)
    write_log(f"Total sampel final: {len(X_sampled)} ({len(np.unique(y_sampled))} kelas)")
    return X_sampled, y_sampled

def extract_hog_features(images, orientations, ppc, cpb):
    hog_features = []
    write_log(f"\n[HOG] Ekstraksi fitur: Orient={orientations}, PPC={ppc}, CPB={cpb}...")
    
    for image in tqdm(images, desc="Ekstraksi HOG"):
        image_2d = image.reshape(IMAGE_SIZE, IMAGE_SIZE)
        features = hog(image_2d,
                       orientations=orientations,
                       pixels_per_cell=ppc,
                       cells_per_block=cpb,
                       transform_sqrt=True,
                       feature_vector=True)
        hog_features.append(features)

    X_features = np.array(hog_features)
    write_log(f"Dimensi fitur HOG: {X_features.shape}")
    return X_features

def evaluate_loocv(X_features, y_labels, C_param):
    write_log(f"\n[LOOCV] Memulai LOOCV FINAL (LinearSVC, C={C_param})...")
    
    model_svm = LinearSVC(C=C_param, max_iter=30000, random_state=42)
    loocv = LeaveOneOut()
    
    start_time = time.time()
    write_log(f"PERINGATAN: LOOCV pada {len(X_features)} sampel ini akan memakan waktu cukup lama.")
    write_log(f"")
    write_log(f"- - - MOHON DITUNGGU - - -")
    write_log(f"")
    
    y_pred = cross_val_predict(model_svm, X_features, y_labels, cv=loocv, n_jobs=-1, verbose=1)
    
    end_time = time.time()
    accuracy = accuracy_score(y_labels, y_pred)
    precision = precision_score(y_labels, y_pred, average='macro')
    f1 = f1_score(y_labels, y_pred, average='macro')
    
    write_log("\n=======================================================")
    write_log("(DONE) Seluruh proses selesai tanpa adanya error")
    write_log("")
    write_log("--- Hasil Evaluasi LOOCV FINAL ---")
    write_log(f"Waktu total LOOCV: {(end_time - start_time)/3600:.2f} jam")
    write_log(f"Akurasi: {accuracy*100:.2f}%")
    write_log(f"Presisi: {precision*100:.2f}%")
    write_log(f"F1-score: {f1*100:.2f}%")
    write_log("=======================================================")

    plot_confusion_matrix(y_labels, y_pred, "linearSVC")

    return accuracy, precision, f1

if __name__ == "__main__":
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    write_log("=======================================================")
    write_log(f"Program Klasifikasi EMNIST Dimulai pada {time.ctime()}")
    write_log(f"Log Output: {LOG_FILE}")
    write_log("=======================================================")

    if not os.path.exists(FILE_PATH):
        write_log(f"ERROR: File {FILE_PATH} tidak ditemukan.")
    else:
        X, y = load_and_sample_data(FILE_PATH)

        if RUN_LOOCV_FINAL:
            write_log("\n---------- FASE 2: EVALUASI LOOCV FINAL ----------")
            X_features_final = extract_hog_features(X, **HOG_PARAMS_FINAL)

            write_log("\n[INFO] Mengurangi dimensi fitur dengan PCA (98% variansi)...")
            pca = PCA(n_components=0.98, random_state=42)
            X_features_final = pca.fit_transform(X_features_final)
            write_log(f"Dimensi fitur setelah PCA: {X_features_final.shape}")

            final_accuracy, final_precision, final_f1 = evaluate_loocv(
                X_features_final, y, SVM_PARAMS_FINAL['C']
            )

            write_log("\n--- EKSEKUSI PROGRAM SELESAI ---")
