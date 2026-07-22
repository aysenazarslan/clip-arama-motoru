# CLIP Görsel Arama Motoru (Image Retrieval Pipeline)

Bu proje, OpenAI'ın CLIP (Contrastive Language-Image Pretraining) modelini kullanarak metin tabanlı görsel arama yapan uçtan uca (end-to-end) bir arama motoru sistemidir. 

Hugging Face `transformers` kütüphanesi kullanılarak, görüntü ve metinler ortak bir 512 boyutlu uzaya (latent space) yansıtılmış ve kosinüs benzerliği (cosine similarity) ile eşleştirilmiştir.

## Özellikler
* Flickr8k veri seti üzerinden otonom vektör çıkarımı (Feature Extraction)
* Tensör tabanlı hızlı kosinüs benzerliği hesaplaması
* `matplotlib` ile görsel pop-up sonuç ekranı
* İnteraktif CLI (Komut Satırı) arama arayüzü

## Kurulum
1. Projeyi klonlayın ve sanal ortam oluşturun.
2. Gereksinimleri yükleyin: `pip install torch torchvision transformers pillow numpy tqdm matplotlib`
3. Kaggle'dan Flickr8k veri setini indirip `Images` klasörü içine yerleştirin.

## Kullanım
* Vektör veritabanını oluşturmak için: `python 1_extract_embeddings.py`
* Arama motorunu başlatmak için: `python 2_search_images.py`