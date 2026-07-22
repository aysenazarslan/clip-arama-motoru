import numpy as np
import os

OUTPUT_DIR = "outputs"

# Dosyaları okuyoruz
embeddings = np.load(os.path.join(OUTPUT_DIR, "image_embeddings.npy"))
paths = np.load(os.path.join(OUTPUT_DIR, "image_paths.npy"))

# Genel boyut bilgisi
print("=== GENEL BİLGİLER ===")
print(f"Toplam vektör matrisinin boyutu: {embeddings.shape}")
print(f"Toplam kaydedilmiş dosya yolu sayısı: {len(paths)}\n")

# İlk resmin vektörünü inceleyelim
ilk_resim_yolu = paths[0]
ilk_resim_vektoru = embeddings[0]

print("=== İLK RESMİN VERİLERİ ===")
print(f"Ait olduğu resim: {ilk_resim_yolu}")
print(f"Bu resmin 512 boyutlu vektörü (ilk 10 değeri):\n{ilk_resim_vektoru[:10]}")
print("...\n")

# Vektör içindeki en büyük ve en küçük değerleri (istatistik) görelim
print("=== MATEMATİKSEL İSTATİSTİKLER ===")
print(f"Vektördeki en yüksek değer: {np.max(ilk_resim_vektoru):.4f}")
print(f"Vektördeki en düşük değer: {np.min(ilk_resim_vektoru):.4f}")