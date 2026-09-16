import os
import io
import torch
import numpy as np
import faiss
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

# --- 1. FAISS Veritabanı Hazırlığı ---
OUTPUT_DIR = "outputs"
image_embeddings = np.load(os.path.join(OUTPUT_DIR, "image_embeddings.npy"))
image_paths = np.load(os.path.join(OUTPUT_DIR, "image_paths.npy"))

faiss.normalize_L2(image_embeddings)
boyut = image_embeddings.shape[1]
index = faiss.IndexFlatIP(boyut)
index.add(image_embeddings)

# --- 2. CLIP Modelini Yükleme ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# --- 3. FastAPI Sunucusunu Başlatma ---
app = FastAPI(title="CLIP Arama Motoru")
app.mount("/images", StaticFiles(directory="Images"), name="images")

class QueryRequest(BaseModel):
    text: str
    top_k: int = 3

# --- 4. YAZI İLE ARAMA ENDPOINT'İ ---
@app.post("/search")
def search_images(req: QueryRequest):
    inputs = processor(text=[req.text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_outputs = model.text_model(**inputs)
        text_features = model.text_projection(text_outputs.pooler_output)
        
    text_vector = text_features.cpu().numpy()
    faiss.normalize_L2(text_vector)
    skorlar, indeksler = index.search(text_vector, req.top_k)
    
    return {"query": req.text, "results": format_results(skorlar, indeksler)}

# --- 5. RESİM İLE ARAMA ENDPOINT'İ (YENİ KORUMALI) ---
@app.post("/search-by-image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(3)):
    contents = await file.read()
    
    # YENİ: 10 MB = 10 * 1024 * 1024 Byte Kontrolü
    MAX_SIZE = 10 * 1024 * 1024
    if len(contents) > MAX_SIZE:
        # HTTP 400 Bad Request fırlat
        raise HTTPException(status_code=400, detail="Yüklenen resim 10 MB sınırını aşıyor. Lütfen daha küçük boyutlu bir dosya seçin.")
        
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = model.vision_model(**inputs)
        image_features = model.visual_projection(vision_outputs.pooler_output)
        
    image_vector = image_features.cpu().numpy()
    faiss.normalize_L2(image_vector)
    
    skorlar, indeksler = index.search(image_vector, top_k)
    
    return {"query": "Yüklenen Resim", "results": format_results(skorlar, indeksler)}

# --- Yardımcı Fonksiyon ---
def format_results(skorlar, indeksler):
    results = []
    for i, idx in enumerate(indeksler[0]):
        dosya_yolu = image_paths[idx]
        dosya_adi = os.path.basename(dosya_yolu)
        resim_linki = f"/images/{dosya_adi}"
        results.append({
            "score": float(skorlar[0][i]),
            "link": resim_linki,
            "filename": dosya_adi
        })
    return results

# --- 6. Frontend'i Sunma ---
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()