import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# --- 1. Ayarlar ve Yüklemeler ---
OUTPUT_DIR = "outputs"

print("Vektörler ve yollar yükleniyor...")
image_embeddings = np.load(os.path.join(OUTPUT_DIR, "image_embeddings.npy")) #Daha önce 1. adımda kaydettiğimiz 1000 adet resmin 512 boyutlu NumPy vektörlerini ve dosya yollarını diskin üzerinden okuyoruz.
image_paths = np.load(os.path.join(OUTPUT_DIR, "image_paths.npy"))

# Vektörleri hızlı hesaplama için PyTorch tensörüne çeviriyoruz
image_embeddings_tensor = torch.tensor(image_embeddings) #: NumPy dizisini PyTorch tensörüne çeviriyoruz. Çünkü PyTorch matris çarpımlarında (kosinüs benzerliği) NumPy'a göre inanılmaz derecede daha hızlıdır

print("CLIP modeli yükleniyor...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval() #CLIP modelini sadece "metinleri" vektöre çevirmesi için belleğe alıyoruz ve eval() ile kilitliyoruz.

device = "cuda" if torch.cuda.is_available() else "cpu" 
model.to(device)
image_embeddings_tensor = image_embeddings_tensor.to(device) #Sadece modeli değil, o 1000 resimlik koca veritabanını da ekran kartının (veya işlemcinin) aktif belleğine taşıyoruz. Böylece arama yaparken diske veya RAM'e git-gel yapmadan saliseler içinde sonuç alıyoruz.

# --- 2. Arama Motoru Fonksiyonu ---
def search_images(query, top_k=3): #Kullanıcının terminale yazdığı metni (query) alıp, modelin anlayacağı "token"lara çeviriyoruz ve aktif cihaza yolluyoruz.
    print(f"\nArama yapılıyor: '{query}'")
    
    # 1. Metni tensöre çevir
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    
    # 2. Metnin vektörünü (embedding) en sağlam yöntemle çıkar
    with torch.no_grad(): #boyut uyuşmazlığı hatasını çözülüyor
        # Doğrudan metin modelini çalıştırıp havuzlanmış (pooled) çıktıyı alıyoruz ve text_projection katmanından geçirerek onu zorla 512 boyutlu ortak uzaya yansıtıyoruz. Artık metin vektörümüz resim vektörlerimizle aynı dilde konuşuyor.
        text_outputs = model.text_model(**inputs)
        pooled_output = text_outputs.pooler_output
        
        # Bu çıktıyı 512 boyutlu ortak CLIP uzayına zorluyoruz (Projection)
        text_features = model.text_projection(pooled_output)
        
    # 3. Kosinüs Benzerliği Hesaplama (Cosine Similarity)
    similarities = torch.nn.functional.cosine_similarity(text_features, image_embeddings_tensor) #1 adet metin vektörünü (1, 512), hafızadaki 1000 adet resim vektörüyle (1000, 512) aynı anda çarpıştırıyoruz. Bize 1000 adet skor (benzerlik oranı) dönüyor.
    
    # 4. En yüksek skorları ve indekslerini (sıra numaralarını) bul
    top_scores, top_indices = torch.topk(similarities, k=top_k)
    
    return top_scores.cpu().numpy(), top_indices.cpu().numpy()

# --- 3. Sonuçları Gösterme (Görselleştirme) ---
def plot_results(query, scores, indices):
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    fig.suptitle(f"Arama Sonucu: '{query}'", fontsize=16)
    
    for i, idx in enumerate(indices):
        img_path = image_paths[idx]
        score = scores[i]
        
        # Resmi diskten oku
        img = Image.open(img_path)
        
        # Ekrana bas
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Skor: {score:.4f}")
        
    plt.tight_layout()
    plt.show()

# --- 4. QA Testi (Sistemi Deneme) ---
# --- 4. QA Testi (İnteraktif Arama Sistemi) ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" CLIP GÖRSEL ARAMA MOTORUNA HOŞ GELDİNİZ ")
    print("="*50)
    
    while True:
        # Terminalden kullanıcı girişi alıyoruz
        arama_metni = input("\nAramak istediğiniz resmi tarif edin (Çıkmak için 'q' yazın): ")
        
        # Programdan çıkış kontrolü
        if arama_metni.lower() == 'q':
            print("Arama motoru kapatılıyor...")
            break
            
        # Boş bir şey girilirse uyarı verip tekrar başa dön
        if not arama_metni.strip():
            print("Lütfen geçerli bir metin girin!")
            continue
            
        # Arama ve çizdirme işlemlerini çağır
        skorlar, indeksler = search_images(arama_metni, top_k=3)
        plot_results(arama_metni, skorlar, indeksler)