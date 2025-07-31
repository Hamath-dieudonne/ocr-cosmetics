import os
import re
from collections import Counter
import pandas as pd
from fuzzywuzzy import process, fuzz
from functools import lru_cache
import time
import logging
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image, ImageEnhance
import pytesseract
import shutil
from urllib.parse import quote
import json
from concurrent.futures import ThreadPoolExecutor

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger la base de données INCI
try:
    INCI_Catalog = pd.read_pickle('static/T_INCI_Catalog_Version_YY_2025_JJ_13_MM_02_HH_15_MN_57.pkl')
    chemical_db = INCI_Catalog['INCI'].tolist()
    logger.info(f"Loaded INCI catalog with {len(chemical_db)} entries")
except Exception as e:
    logger.error(f"Error loading INCI catalog: {str(e)}")
    chemical_db = []

# Charger une liste réduite d'ingrédients courants
try:
    with open('static/common_inci.json', 'r') as f:
        common_chemical_db = json.load(f)["common_ingredients"]
    logger.info(f"Loaded common INCI catalog with {len(common_chemical_db)} entries")
except FileNotFoundError:
    common_chemical_db = chemical_db[:3000]
    logger.warning("Common INCI list not found, using first 1000 entries")

# Configurer le chemin de Tesseract
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def validate_image_path(image_path: str) -> bool:
    abs_image_path = os.path.abspath(image_path)
    abs_upload_folder = os.path.abspath(UPLOAD_FOLDER)
    return abs_image_path.startswith(abs_upload_folder) and os.path.exists(image_path)

@lru_cache(maxsize=1000)
def get_top_chemicals(query: str, chemical_db_tuple: tuple, common_db_tuple: tuple, top_n: int = 3, threshold: int = 85) -> str:
    chemical_db = list(chemical_db_tuple)
    common_db = list(common_db_tuple)
    query_lower = query.lower().strip()
    print(f"Processing query: {query_lower}")
    
    chemical_set = set(c.lower() for c in chemical_db)
    if query_lower in chemical_set:
        logger.debug(f"Exact match found in chemical_db for {query}: {query}")
        return query.capitalize()
    
    common_set = set(c.lower() for c in common_db)
    if query_lower in common_set:
        logger.debug(f"Exact match found in common_db for {query}: {query}")
        return query.capitalize()
    
    matches = process.extract(query, common_db, scorer=fuzz.WRatio, limit=top_n)
    filtered_matches = [match for match in matches if match[1] >= threshold]
    
    if not filtered_matches:
        logger.debug("No matches in common_db, trying full chemical_db")
        matches = process.extract(query, chemical_db, scorer=fuzz.WRatio, limit=top_n)
        filtered_matches = [match for match in matches if match[1] >= threshold]
        if not filtered_matches:
            logger.debug("No matches above threshold")
            return "NF"
    
    sorted_matches = sorted(filtered_matches, key=lambda m: (-m[1], -len(m[0])))
    best_match = sorted_matches[0][0]
    logger.debug(f"Best match for {query}: {best_match}")
    return best_match

def detect_separator(text: str) -> str | None:
    possible_separators = [',', ';', '.', '*']
    counts = Counter(char for char in text if char in possible_separators)
    logger.debug(f"Separator counts: {counts}")
    if not counts:
        logger.debug("No separator found")
        return None
    return max(counts, key=counts.get)

def process_inci_list(raw_text: str) -> str:
    if not raw_text:
        logger.warning("No raw text provided")
        return ""
    
    cleaned_text = re.sub(r'[\n\r\s]+', ' ', raw_text.strip())
    logger.debug(f"Cleaned text: {cleaned_text}")
    
    separator = detect_separator(cleaned_text)
    logger.debug(f"Detected separator: {separator}")
    
    ingredients = cleaned_text.split(separator) if separator else []
    
    cleaned_ingredients = [
        re.sub(r'[^a-zA-Z0-9\s-]', '', ingredient).strip().capitalize()
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
    
    seen = set()
    unique_ingredients = [ing for ing in cleaned_ingredients if not (ing.lower() in seen or seen.add(ing.lower()))]
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        unique_ingredients = list(executor.map(
            lambda ing: get_top_chemicals(ing, tuple(chemical_db), tuple(common_chemical_db)),
            unique_ingredients
        ))
    unique_ingredients = [match.title() if match != "NF" else ing for ing, match in zip(cleaned_ingredients, unique_ingredients)]
    logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds")
    
    return ', '.join(unique_ingredients) if unique_ingredients else "Aucun ingrédient détecté"

@app.get("/")
async def index(request: Request, extracted_text: str | None = None, image_path: str | None = None, error: str | None = None):
    logger.info("GET request received for /")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "extracted_text": extracted_text, "image_path": image_path, "error": error}
    )

@app.post("/")
async def index(request: Request, image: UploadFile = File(None)):
    logger.info("POST request received")
    extracted_text = None
    image_path = None
    error = None

    if image is None or not image.filename:
        error = "Aucune image reçue ou nom de fichier vide"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}", status_code=303)

    if image.size > MAX_FILE_SIZE:
        error = f"L'image dépasse la taille maximale de {MAX_FILE_SIZE // 1024 // 1024}MB"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}", status_code=303)

    content_type = image.content_type
    if not content_type or not content_type.startswith("image/"):
        error = "Le fichier n'est pas une image valide"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}", status_code=303)

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
        return RedirectResponse(url=f"/?error={quote(error)}", status_code=303)

    try:
        logger.info("Attempting to open image for OCR")
        img = Image.open(image_path).convert('L')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        logger.info("Image opened and preprocessed successfully")
        raw_text = pytesseract.image_to_string(img, lang="eng+fra")
        logger.debug(f"Raw extracted text: {raw_text}")
        start_time = time.time()
        extracted_text = process_inci_list(raw_text)
        logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds: {extracted_text}")
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
            if file_age > 300:
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
