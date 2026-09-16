import os
import torch
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# --- 1. Model ve Veritabanı Kurulumu ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"

print("Model yükleniyor...")
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

print("FAISS Indeksi ve yollar yükleniyor...")
OUTPUT_DIR = "outputs"
image_embeddings = np.load(os.path.join(OUTPUT_DIR, "image_embeddings.npy"))
image_paths = np.load(os.path.join(OUTPUT_DIR, "image_paths.npy"))

# Yolları sadece dosya adı kalacak şekilde temizle (Örn: "123.jpg")
image_filenames = [os.path.basename(path) for path in image_paths]

faiss.normalize_L2(image_embeddings)
boyut = image_embeddings.shape[1]
index = faiss.IndexFlatIP(boyut)
index.add(image_embeddings)

# --- 2. Flickr8K Açıklamalarını (Captions) Yükleme ---
# Not: Flickr8K'nın standart formatında virgülle ayrılmış "image,caption" formatı kullanılır.
print("Flickr8K açıklamaları okunuyor...")
df = pd.read_csv("captions.txt", sep=",") 

# Performans için ilk 1000 sorguyu test edelim (Tümünü test etmek istersen limiti kaldır)
test_df = df.head(1000)

hits_at_1 = 0
hits_at_5 = 0
hits_at_10 = 0
toplam_sorgu = len(test_df)

print("Recall@k Testi Başlıyor...")

# --- 3. Recall@k Hesaplama Döngüsü ---
with torch.no_grad():
    for _, row in tqdm(test_df.iterrows(), total=toplam_sorgu):
        target_image = row['image']
        query_text = row['caption']
        
        # Eğer hedef resim bizim indeksimizde yoksa atla
        if target_image not in image_filenames:
            toplam_sorgu -= 1
            continue

        # Metni kodla ve FAISS'te ara
        inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
        text_features = model.text_projection(model.text_model(**inputs).pooler_output)
        
        text_vector = text_features.cpu().numpy()
        faiss.normalize_L2(text_vector)
        
        # Recall@10 için en iyi 10 sonucu getir
        skorlar, indeksler = index.search(text_vector, 10)
        
        # Bulunan indeksleri dosya isimlerine çevir
        bulunan_dosyalar = [image_filenames[idx] for idx in indeksler[0]]
        
        # Hedef görselin kaçıncı sırada olduğunu kontrol et
        if target_image in bulunan_dosyalar[:1]:
            hits_at_1 += 1
        if target_image in bulunan_dosyalar[:5]:
            hits_at_5 += 1
        if target_image in bulunan_dosyalar[:10]:
            hits_at_10 += 1

# --- 4. Sonuçları Raporlama ve Dosyaya Kaydetme ---
recall_1 = (hits_at_1 / toplam_sorgu) * 100
recall_5 = (hits_at_5 / toplam_sorgu) * 100
recall_10 = (hits_at_10 / toplam_sorgu) * 100

# Rapor şablonunu Markdown formatında hazırlıyoruz
rapor_metni = f"""# 📊 VisionSearch Değerlendirme Raporu (Baseline)

**Veri Seti:** Flickr8K (Test edilen sorgu sayısı: {toplam_sorgu})
**Kullanılan Model:** {model_name}
**Vektör İndeksi:** FAISS (IndexFlatIP)

## 🎯 Başarı Metrikleri (Recall@k)
* **Recall@1:**  % {recall_1:.2f}
* **Recall@5:**  % {recall_5:.2f}
* **Recall@10:** % {recall_10:.2f}

---
*Not: Bu sonuçlar `evaluate_metrics.py` betiği tarafından otomatik olarak oluşturulmuştur.*
"""

# Sonuçları önce terminale yazdır
print("\n" + "="*45)
print(f"📊 Fickr8K Recall@k Sonuçları (Sorgu: {toplam_sorgu})")
print("="*45)
print(f"Recall@1  : % {recall_1:.2f}")
print(f"Recall@5  : % {recall_5:.2f}")
print(f"Recall@10 : % {recall_10:.2f}")
print("="*45)

# Ardından Markdown dosyası olarak proje klasörüne kaydet
rapor_adi = "evaluation_report.md"
with open(rapor_adi, "w", encoding="utf-8") as f:
    f.write(rapor_metni)

print(f"\n✅ Harika haber! Sonuçlar '{rapor_adi}' adlı dosyaya kaydedildi.")