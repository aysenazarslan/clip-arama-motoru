import os
import torch
import numpy as np #Tensör hesaplamaları (PyTorch) ve vektörleri matris formatında diske kaydetmek (NumPy) için.
from PIL import Image #PIL (Image): Görüntüleri diskten okumak için.
from transformers import CLIPModel, CLIPProcessor #transformers: Hugging Face'ten CLIP modelini ve ön işleyicisini (processor) çekmek için.
from tqdm import tqdm #tqdm: Terminaldeki ilerleme çubuğunu (progress bar) göstermek için.

IMAGE_DIR = "Images"
OUTPUT_DIR = "outputs"
MAX_IMAGES = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True) #Eğer outputs adında bir klasör yoksa oluşturur

print("CLIP modeli yükleniyor...")
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()  #Modeli "değerlendirme" moduna alır. Ağırlıkları kilitler, dropout gibi rastgelelik katan eğitim katmanlarını kapatır.

device = "cuda" if torch.cuda.is_available() else "cpu" #device: Sisteminde bir NVIDIA ekran kartı (CUDA) varsa işlemi onun üzerinden, yoksa normal işlemci (CPU) üzerinden yapmasını söyler ve modeli bu donanıma taşır
model.to(device)

all_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")] #os.listdir ile Images klasöründeki tüm dosyaları tarayıp sadece .jpg ile bitenleri bir listeye atıyoruz.
selected_files = all_files[:MAX_IMAGES]
image_paths = [os.path.join(IMAGE_DIR, f) for f in selected_files]

all_embeddings = [] #Boş bir all_embeddings listesi oluşturuyoruz, çıkarılan her vektörü buraya atacağız
print(f"Toplam {len(image_paths)} resim işleniyor...")

for img_path in tqdm(image_paths):
    try:
        image = Image.open(img_path).convert("RGB") #Resmi açıp .convert("RGB") ile 3 kanallı renge zorluyoruz (siyah-beyaz resimlerin hata vermesini önler).
        inputs = processor(images=image, return_tensors="pt").to(device) #inputs: Resmi CLIP'in beklediği boyutta bir tensöre çevirip aktif donanıma (to(device)) yolluyoruz.
        
        with torch.no_grad(): #torch.no_grad(): Gradyan hesaplamasını kapatarak RAM/VRAM tasarrufu sağlıyoruz
            # EN SAĞLAM YÖNTEM: Ham modeli çalıştırıp 512 boyutuna zorla yansıtıyoruz
            vision_outputs = model.vision_model(**inputs)
            image_features = model.visual_projection(vision_outputs.pooler_output) #visual projection:768 boyutlu ham veriyi, metinlerle aynı boyuta (512) sıkıştırmak için modelin projeksiyon (yansıtma) katmanından geçiriyoruz
            
        embedding = image_features[0].cpu().numpy() #Üretilen tensör ekran kartındaysa onu önce ana belleğe (cpu()) alıyoruz, sonra diske kaydedebilmek için saf sayı dizisine (numpy()) çeviriyoruz
        all_embeddings.append(embedding) #Oluşan 512'lik diziyi en baştaki listemize ekliyoruz (append)
    except Exception as e:  #ozuk bir resim dosyasına denk gelirsek programın çökmesi engellenir; hatayı ekrana yazar ve bir sonraki resme geçer
        print(f"Hata oluşan resim {img_path}: {e}")

embeddings_array = np.array(all_embeddings) #Python listelerinde biriktirdiğimiz binlerce vektörü ve dosya yolunu, daha verimli olan NumPy matrislerine çeviriyoruz
paths_array = np.array(image_paths)

# BURAYA DİKKAT: Artık terminalde (1000, 512) yazmalı!
print("\nÇıkarım tamamlandı. Vektör boyutları:", embeddings_array.shape) 
np.save(os.path.join(OUTPUT_DIR, "image_embeddings.npy"), embeddings_array)
np.save(os.path.join(OUTPUT_DIR, "image_paths.npy"), paths_array) #np.save ile bu verileri outputs klasörüne, hızlıca okunabilecek ikili (binary) .npy formatında kalıcı olarak kaydediyoruz
print("Kayıt başarılı! Vektörler eklendi.")