import os
import re
from collections import Counter
import pandas as pd
from functools import lru_cache
import time
import logging
from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image, ImageEnhance
import pytesseract
import shutil
from urllib.parse import quote
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from pathlib import Path
from cryptography.fernet import Fernet
import cProfile
import pstats
import io
import numpy as np

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "ocr-cosmetics.onrender.com"])
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Précompiler les regex
SEPARATOR_RE = re.compile(r'[;*\+\»]+')
CLEAN_INGREDIENT_RE = re.compile(r'[^a-zA-Z0-9\s-]')

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str):
    file_path = os.path.normpath(file_path)
    full_path = os.path.join("static", file_path)
    if not os.path.exists(full_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    if "inci_only.pkl.enc" in full_path or not full_path.startswith(os.path.abspath("static/uploads")):
        return JSONResponse(content={"error": "Access denied"}, status_code=404)
    return FileResponse(full_path)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

chemical_db = []
common_db = []
chemical_set = set()

@app.on_event("startup")
async def startup_event():
    global chemical_db, common_db, chemical_set
    try:
        env_path = Path('.') / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        encrypted_path = 'static/inci_only.pkl.enc'
        if os.path.exists(encrypted_path):
            with open(encrypted_path, "rb") as f:
                encrypted_data = f.read()
            decrypted_data = Fernet(os.getenv("FERNET_KEY").encode()).decrypt(encrypted_data)
            with io.BytesIO(decrypted_data) as f:
                INCI_Catalog = pd.read_pickle(f)
                chemical_db = INCI_Catalog['INCI'].tolist()
                common_db = [chem.upper() for chem in chemical_db]  # Pré-calcul en majuscules
                chemical_set = set(common_db)
                logger.info(f"Loaded INCI catalog with {len(chemical_db)} entries")
        else:
            raise FileNotFoundError("Encrypted .pkl file not found")
    except Exception as e:
        logger.error(f"Error loading INCI catalog or environment: {str(e)}")

def validate_image_path(image_path: str) -> bool:
    abs_image_path = os.path.abspath(image_path)
    abs_upload_folder = os.path.abspath(UPLOAD_FOLDER)
    return abs_image_path.startswith(abs_upload_folder) and os.path.exists(image_path)

@lru_cache(maxsize=1000)
def get_top_chemicals_vectorized(queries: tuple) -> list:
    query_upper = [q.upper().strip() for q in queries]
    logger.debug(f"Processing queries: {query_upper}")
    matches = np.array([[fuzz.WRatio(q, chem) for chem in common_db] for q in query_upper])
    best_matches = np.argmax(matches, axis=1)
    scores = matches[np.arange(len(query_upper)), best_matches]
    result = []
    for i, (q, score, best_idx) in enumerate(zip(query_upper, scores, best_matches)):
        if score >= 85:
            match = common_db[best_idx]
            logger.debug(f"Best fuzzy match for {q}: {match} (score: {score})")
            result.append(match)
        else:
            logger.warning(f"No match found for {q} with threshold 85")
            result.append(q.capitalize())
    return result

def detect_separator(text: str) -> str | None:
    possible_separators = [',']
    counts = Counter(char for char in text if char in possible_separators)
    logger.debug(f"Separator counts: {counts}")
    return max(counts, key=counts.get) if counts else None

def process_inci_list(raw_text: str) -> str:
    if not raw_text:
        logger.warning("No raw text provided")
        return ""
    logger.debug(f"Raw text received: {raw_text}")
    
    # Remplacer les séparateurs par ','
    cleaned_text = SEPARATOR_RE.sub(',', raw_text.strip())
    logger.debug(f"Cleaned text after separator replacement: {cleaned_text}")
    
    # Détecter le séparateur
    separator = detect_separator(cleaned_text)
    logger.debug(f"Detected separator: {separator}")
    
    # Diviser en ingrédients
    ingredients = cleaned_text.split(separator) if separator else [cleaned_text]
    cleaned_ingredients = [
        CLEAN_INGREDIENT_RE.sub('', ingredient).strip()
        for ingredient in ingredients
        if ingredient.strip() and len(ingredient.strip()) > 2 and not ingredient.strip().isdigit()
    ]
    
    if not cleaned_ingredients:
        potential_ingredients = re.findall(r'\b[A-Z][a-zA-Z0-9\s-]*\b', cleaned_text)
        cleaned_ingredients = [
            CLEAN_INGREDIENT_RE.sub('', ingredient).strip()
            for ingredient in potential_ingredients
            if len(ingredient.strip()) > 2 and not ingredient.strip().isdigit()
        ]
    
    if not cleaned_ingredients:
        logger.warning("No valid ingredients detected after processing")
        return "Aucun ingrédient détecté"
    
    seen = set()
    unique_ingredients = [ing for ing in cleaned_ingredients if not (ing.lower() in seen or seen.add(ing.lower()))]
    if not unique_ingredients:
        return "Aucun ingrédient détecté"
    
    start_time = time.time()
    # Vectorisation des comparaisons floues
    results = get_top_chemicals_vectorized(tuple(unique_ingredients))
    logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds with {len(unique_ingredients)} ingredients")
    
    return ', '.join(results)

@app.head("/")
async def head_root():
    logger.info("HEAD request received for /")
    return Response(status_code=200)

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

@app.post("/")
async def index(request: Request, image: UploadFile = File(None)):
    logger.info("POST request received")
    extracted_text = None
    image_path = None
    error = None
    has_submitted = True

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
            shutil.copyfileobj(image.file, buffer)
        logger.info(f"Image saved at: {image_path}")
    except Exception as e:
        error = f"Erreur lors de la sauvegarde de l'image : {str(e)}"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}&has_submitted=true", status_code=303)

    try:
        logger.info("Attempting to open image for OCR")
        img = Image.open(image_path).convert('L')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        logger.info("Image opened and preprocessed successfully")
        raw_text = pytesseract.image_to_string(img, lang="eng+fra")
        logger.debug(f"Raw extracted text: {raw_text}")
        
        # Profilage de process_inci_list
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()
        extracted_text = process_inci_list(raw_text)
        profiler.disable()
        logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds")
        
        # Sauvegarder les résultats du profilage
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Limiter à 10 lignes
        logger.info(f"Profiling results:\n{s.getvalue()}")

    except Exception as e:
        error = f"Erreur lors de l'extraction des ingrédients : {str(e)}"
        logger.error(error)

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

@app.get("/cleanup")
async def cleanup():
    logger.info("Cleaning up old images")
    current_time = time.time()
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > 3600:  # 1 heure
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting old file {file_path}: {str(e)}")
    return {"message": "Cleanup completed"}

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