import os
import re
from collections import Counter
import pandas as pd
from functools import lru_cache
import time
import logging
import cProfile
import io
import base64
import requests
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image, ImageEnhance
import pytesseract
import shutil
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from rapidfuzz import fuzz
from pytrie import Trie
from pathlib import Path
from cryptography.fernet import Fernet

# Configuring logging setup to capture INFO-level messages and errors
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()]  # Force l'affichage sur stdout pour Render
)
logger = logging.getLogger(__name__)

# Initializing FastAPI application with middleware for trusted hosts
app = FastAPI()
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "ocr-cosmetics.onrender.com"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Serving static files with path validation
@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    file_path = os.path.normpath(file_path)
    full_path = os.path.join("static", file_path)
    if not os.path.exists(full_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    if "inci_only.pkl.enc" in full_path or not full_path.startswith(os.path.abspath("static/uploads")):
        return JSONResponse(content={"error": "Access denied"}, status_code=404)
    return FileResponse(full_path)

# Defining upload folder and maximum file size
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

chemical_db = []
common_db_trie = Trie()

# Loading INCI catalog and initializing Trie during application startup
@app.on_event("startup")
async def startup_event():
    global chemical_db, common_db_trie
    try:
        # Pas de load_dotenv(), car Render utilise les env vars
        TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        encrypted_path = 'static/inci_only.pkl.enc'
        if os.path.exists(encrypted_path):
            with open(encrypted_path, "rb") as f:
                encrypted_data = f.read()
            decrypted_data = Fernet(os.getenv("FERNET_KEY").encode()).decrypt(encrypted_data)
            INCI_Catalog = pd.read_pickle(io.BytesIO(decrypted_data))
            chemical_db = INCI_Catalog['INCI'].tolist()
            for chem in chemical_db:
                common_db_trie[chem.lower()] = chem
            logger.info(f"Loaded INCI catalog with {len(chemical_db)} entries and built Trie")
        else:
            raise FileNotFoundError("Encrypted .pkl file not found")
    except Exception as e:
        logger.error(f"Error loading INCI catalog or environment: {str(e)}")

# Validating image path security
def validate_image_path(image_path: str) -> bool:
    abs_image_path = os.path.abspath(image_path)
    abs_upload_folder = os.path.abspath(UPLOAD_FOLDER)
    return abs_image_path.startswith(abs_upload_folder) and os.path.exists(image_path)

# Caching top chemical matches with LRU cache
@lru_cache(maxsize=1000)
def get_top_chemicals(query: str, threshold: int = 80) -> str:
    query_lower = query.lower().strip()
    logger.info(f"Processing query: {query_lower}")
    if query_lower in common_db_trie:
        logger.info(f"Exact match found for {query}")
        return common_db_trie[query_lower]
    query_cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '', query_lower)
    query_length = len(query_cleaned)
    candidates = list(common_db_trie.keys(prefix=query_cleaned[:3]))
    if not candidates:
        candidates = list(common_db_trie.keys())
    matches = [(common_db_trie[chem], fuzz.WRatio(query_lower, chem)) for chem in candidates]
    matches.sort(key=lambda x: x[1], reverse=True)
    top_3_matches = matches[:3]
    logger.info(f"Top 3 matches for {query_lower}:")
    for chem, score in top_3_matches:
        logger.info(f"  - {chem} (score: {score})")
    best_fuzzy_match = "NF"
    best_fuzzy_score = 0
    for chem, score in top_3_matches:
        if score >= threshold:
            if score > best_fuzzy_score or (score == best_fuzzy_score and len(chem) > len(best_fuzzy_match)):
                best_fuzzy_score = score
                best_fuzzy_match = chem
    if best_fuzzy_match != "NF":
        logger.info(f"Best fuzzy match for {query}: {best_fuzzy_match} (score: {best_fuzzy_score})")
        return best_fuzzy_match
    logger.warning(f"No match found for {query_lower} with threshold {threshold}")
    return query.capitalize()

# Detecting dominant separator in text
def detect_separator(text: str) -> str | None:
    possible_separators = [' ,', ';', '*', '+', '»']
    counts = Counter(text.count(sep) for sep in possible_separators)
    logger.info(f"Separator counts: {counts}")
    if not counts or max(counts.values()) == 0:
        logger.info("No separator found")
        return None
    dominant_separator = max(possible_separators, key=lambda sep: text.count(sep))
    normalized_text = text
    for sep in possible_separators:
        if sep != ' /':
            normalized_text = normalized_text.replace(sep, ',')
    logger.info(f"Normalized text with separators replaced by comma: {normalized_text}")
    return ','

# Processing INCI list from raw text
def process_inci_list(raw_text: str) -> str:
    if not raw_text:
        logger.warning("No raw text provided")
        return ""
    logger.info(f"Raw text received: {raw_text}")
    cleaned_text = re.sub(r'[\n\r\s]+', ' ', raw_text.strip())
    logger.info(f"Cleaned text: {cleaned_text}")
    separator = detect_separator(cleaned_text)
    logger.info(f"Detected separator: {separator}")
    if separator:
        normalized_text = cleaned_text
        for sep in [' ,', ';', '*', '+', '»']:
            if sep != ' /':
                normalized_text = normalized_text.replace(sep, ',')
        normalized_text = re.sub(r'\s+', ' ', normalized_text).replace(',,', ',').strip(',')
        logger.info(f"Normalized text after cleanup: {normalized_text}")
        ingredients = [part.strip() for part in normalized_text.split(',') if part.strip()]
    else:
        ingredients = [cleaned_text]
    cleaned_ingredients = [
        re.sub(r'[^a-zA-Z0-9\s/-]', '', ingredient).strip().capitalize()
        for ingredient in ingredients
        if ingredient.strip() and len(ingredient.strip()) > 2 and not ingredient.strip().isdigit()
    ]
    if not cleaned_ingredients:
        potential_ingredients = re.findall(r'\b[A-Z][a-zA-Z0-9\s-]*\b', cleaned_text)
        cleaned_ingredients = [
            re.sub(r'[^a-zA-Z0-9\s-]', '', ingredient).strip().capitalize()
            for ingredient in potential_ingredients
            if len(ingredient.strip()) > 2 and not ingredient.strip().isdigit()
        ]
    if not cleaned_ingredients:
        logger.warning("No valid ingredients detected after processing")
        return "Aucun ingrédient détecté"
    seen = set()
    unique_ingredients = [ing for ing in cleaned_ingredients if not (ing.lower() in seen or seen.add(ing.lower()))]
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=2 if len(unique_ingredients) > 2 else 1) as executor:
        profiler = cProfile.Profile()
        profiler.enable()
        results = list(executor.map(get_top_chemicals, unique_ingredients))
        profiler.disable()
        profiler.print_stats()
    unique_ingredients = [match.title() if match != "NF" else ing for ing, match in zip(unique_ingredients, results)]
    logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds with {len(unique_ingredients)} ingredients")
    return ', '.join(unique_ingredients) if unique_ingredients else "Aucun ingrédient détecté"

# Handling HEAD request for root
@app.head("/")
async def head_root():
    logger.info("HEAD request received for /")
    return Response(status_code=200)

# Rendering index page with GET request
@app.get("/")
async def index(request: Request, extracted_text: str | None = None, image_path: str | None = None, error: str | None = None, has_submitted: bool = False, is_loading: bool = False):
    logger.info("GET request received for /")
    context = {
        "request": request,
        "extracted_text": extracted_text if extracted_text is not None else "",
        "image_path": image_path if image_path is not None else "",
        "error": error if error is not None else "",
        "has_submitted": has_submitted,
        "is_loading": is_loading
    }
    return templates.TemplateResponse("index.html", context)

# Handling image upload and OCR processing with POST request
@app.post("/")
async def index(request: Request, image: UploadFile = File(None)):
    logger.info("POST request received")
    extracted_text = None
    image_path = None
    error = None
    has_submitted = True
    total_start_time = time.time()

    if image is None or not image.filename:
        error = "Aucune image reçue ou nom de fichier vide"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}&has_submitted=true", status_code=303)

    if image.size > MAX_FILE_SIZE:
        error = f"L'image dépasse la taille maximale de {MAX_FILE_SIZE // 1024 // 1024}MB"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}&has_submitted=true", status_code=303)

    content_type = image.content_type
    if not content_type or not content_type.startswith("image/"):
        error = "Le fichier n'est pas une image valide"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}&has_submitted=true", status_code=303)

    try:
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{re.sub(r'[^a-zA-Z0-9._-]', '_', image.filename)}"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(image_path, "wb") as buffer:
            buffer.write(await image.read())
        logger.info(f"Image saved at: {image_path}")
    except Exception as e:
        error = f"Erreur lors de la sauvegarde de l'image : {str(e)}"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}&has_submitted=true", status_code=303)

    try:
        logger.info("Attempting to open image for OCR with OpenRouter")
        ocr_start_time = time.time()
        extracted_text = await ocr_with_openrouter(image_path)
        logger.info(f"OCR completed in {time.time() - ocr_start_time:.2f} seconds. Extracted text: {extracted_text}")
        start_time = time.time()
        extracted_text = process_inci_list(extracted_text)
        logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        error = f"Erreur lors de l'extraction des ingrédients : {str(e)}"
        logger.error(error)

    logger.info(f"Total processing time: {time.time() - total_start_time:.2f} seconds")
    query_params = []
    if extracted_text:
        query_params.append(f"extracted_text={quote(extracted_text)}")
    if image_path:
        query_params.append(f"image_path={quote(image_path)}")
    if error:
        query_params.append(f"error={quote(error)}")
    query_params.append(f"has_submitted=true")
    query_params.append(f"is_loading=false")
    redirect_url = "/?" + "&".join(query_params) if query_params else "/"
    return RedirectResponse(url=redirect_url, status_code=303)

# Cleaning up old images
@app.get("/cleanup")
async def cleanup():
    logger.info("Cleaning up old images")
    current_time = time.time()
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > 3600:  # 1 hour
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting old file {file_path}: {str(e)}")
    return {"message": "Cleanup completed"}

# Deleting a specific image
@app.post("/delete_image")
async def delete_image(request: Request):
    try:
        body = await request.json()
        image_path = body.get("image_path")
        if not image_path:
            logger.error("No image_path provided in delete_image request")
            return JSONResponse(content={"error": "Aucun chemin d'image fourni"}, status_code=400)
        
        if not validate_image_path(image_path):
            logger.error(f"Invalid or non-existent image path: {image_path}")
            return JSONResponse(content={"error": "Chemin d'image invalide ou non trouvé"}, status_code=400)
        
        os.remove(image_path)
        logger.info(f"Deleted image: {image_path}")
        return JSONResponse(content={"message": "Image deleted"})
    except Exception as e:
        logger.error(f"Error deleting image {image_path}: {str(e)}")
        return JSONResponse(content={"error": f"Erreur lors de la suppression de l'image : {str(e)}"}, status_code=500)

# Performing OCR with OpenRouter Qwen2.5-VL-72B-Instruct
async def ocr_with_openrouter(image_path):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("OPENROUTER_API_KEY not found in environment")

    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://ocr-cosmetics.onrender.com",
        "X-Title": "OCR Cosmetics"
    }
    payload = {
        "model": "qwen/qwen2.5-vl-72b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image as a single string, focusing on ingredient lists."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 2000
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")