import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import faiss
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# --- 1. Ayarlar ve Veritabanını Yükleme ---
OUTPUT_DIR = "outputs"

print("Vektörler ve yollar yükleniyor...")
image_embeddings = np.load(os.path.join(OUTPUT_DIR, "image_embeddings.npy"))
image_paths = np.load(os.path.join(OUTPUT_DIR, "image_paths.npy"))

# --- 2. FAISS İndeksini Hazırlama ---
# FAISS'te doğrudan "Kosinüs Benzerliği" fonksiyonu yoktur. 
# Bunun yerine vektörleri L2 ile normalize edip, "İç Çarpım (Inner Product - IP)" 
# indeksine sokarsak, matematiksel olarak Kosinüs Benzerliği elde etmiş oluruz.

print("FAISS indeksi oluşturuluyor...")
faiss.normalize_L2(image_embeddings) # Vektörleri normalize et

boyut = image_embeddings.shape[1] # 512 boyut
index = faiss.IndexFlatIP(boyut)  # Inner Product (Benzerlik) indeksi oluştur
index.add(image_embeddings)       # 1000 resmi FAISS motoruna yükle

print(f"FAISS sistemine {index.ntotal} adet vektör başarıyla yüklendi!")

# --- 3. Model Yükleme ---
print("CLIP modeli yükleniyor...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- 4. FAISS Arama Fonksiyonu ---
def faiss_search(query, top_k=3):
    # 1. Metni tensöre çevir
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    
    # 2. Metnin vektörünü (embedding) çıkar
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        pooled_output = text_outputs.pooler_output
        text_features = model.text_projection(pooled_output)
        
    # 3. FAISS NumPy beklediği için PyTorch'tan NumPy'a geçiriyoruz
    text_vector = text_features.cpu().numpy()
    
    # 4. Arama yapmadan önce metin vektörünü de normalize etmeliyiz
    faiss.normalize_L2(text_vector)
    
    # 5. Işık hızında arama! (FAISS bizden skorları ve indeksleri döner)
    skorlar, indeksler = index.search(text_vector, top_k)
    
    # FAISS sonuçları matris içinde liste olarak döner (Örn: [[skor1, skor2]]), 
    # biz ilk sorguyu [0] alarak dışarı çıkartıyoruz.
    return skorlar[0], indeksler[0]

# --- 5. Sonuçları Gösterme (Görselleştirme) ---
def plot_results(query, scores, indices):
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    fig.suptitle(f"FAISS Arama Sonucu: '{query}'", fontsize=16)
    
    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        score = scores[i]
        
        img = Image.open(img_path)
        
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"FAISS Skoru: {score:.4f}")
        
    plt.tight_layout()
    plt.show()

# --- 6. İnteraktif Kullanım ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" FAISS + CLIP GÖRSEL ARAMA MOTORU ")
    print("="*50)
    
    while True:
        arama_metni = input("\nAramak istediğiniz resmi tarif edin (Çıkmak için 'q' yazın): ")
        
        if arama_metni.lower() == 'q':
            print("FAISS arama motoru kapatılıyor...")
            break
            
        if not arama_metni.strip():
            print("Lütfen geçerli bir metin girin!")
            continue
            
        skorlar, indeksler = faiss_search(arama_metni, top_k=3)
        plot_results(arama_metni, skorlar, indeksler)