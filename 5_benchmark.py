import os
import time
import torch
import numpy as np
import faiss
from transformers import CLIPModel, CLIPProcessor

# --- 1. Verileri Yükleme ---
OUTPUT_DIR = "outputs"
print("Vektörler diskin üzerinden okunuyor...")
image_embeddings = np.load(os.path.join(OUTPUT_DIR, "image_embeddings.npy"))

# PyTorch için hazırlık (Tensöre çevirip cihaza atma)
device = "cuda" if torch.cuda.is_available() else "cpu"
image_embeddings_tensor = torch.tensor(image_embeddings).to(device)

# FAISS için hazırlık (Normalize edip indekse ekleme)
faiss.normalize_L2(image_embeddings)
boyut = image_embeddings.shape[1]
index = faiss.IndexFlatIP(boyut)
index.add(image_embeddings)

# --- 2. Modeli Yükleme ---
print("CLIP modeli yükleniyor...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

def run_benchmark(query):
    print(f"\n--- ARAMA TESTİ: '{query}' ---")
    
    # Metni vektöre çevirme (Kronometre DIŞINDA tutuyoruz çünkü ikisi için de ortak)
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        text_features = model.text_projection(text_outputs.pooler_output)
    
    # --- YARIŞMACI 1: PYTORCH (Kosinüs Benzerliği) ---
    # Kronometreyi başlat
    start_time_pt = time.perf_counter()
    
    similarities = torch.nn.functional.cosine_similarity(text_features, image_embeddings_tensor)
    top_scores_pt, top_indices_pt = torch.topk(similarities, k=3)
    
    # Kronometreyi durdur
    end_time_pt = time.perf_counter()
    pytorch_suresi = (end_time_pt - start_time_pt) * 1000 # Milisaniyeye çevir
    
    # --- YARIŞMACI 2: FAISS (L2 + Inner Product) ---
    text_vector_faiss = text_features.cpu().numpy()
    faiss.normalize_L2(text_vector_faiss)
    
    # Kronometreyi başlat
    start_time_faiss = time.perf_counter()
    
    skorlar_faiss, indeksler_faiss = index.search(text_vector_faiss, 3)
    
    # Kronometreyi durdur
    end_time_faiss = time.perf_counter()
    faiss_suresi = (end_time_faiss - start_time_faiss) * 1000 # Milisaniyeye çevir
    
    # --- SONUÇLARI YAZDIRMA ---
    print(f"PyTorch Arama Süresi : {pytorch_suresi:.4f} milisaniye")
    print(f"FAISS Arama Süresi   : {faiss_suresi:.4f} milisaniye")
    
    if faiss_suresi < pytorch_suresi:
        fark = pytorch_suresi / faiss_suresi
        print(f"Sonuç: FAISS, PyTorch'tan {fark:.1f} kat daha hızlı!")
    else:
        fark = faiss_suresi / pytorch_suresi
        print(f"Sonuç: PyTorch, FAISS'ten {fark:.1f} kat daha hızlı!")

# --- 3. Testi Başlat ---
if __name__ == "__main__":
    while True:
        arama = input("\nHız testi için bir kelime girin (Çıkmak için 'q'): ")
        if arama.lower() == 'q':
            break
        if not arama.strip():
            continue
            
        run_benchmark(arama)