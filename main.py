import os
import re
from collections import Counter
import pandas as pd
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
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from pathlib import Path
from os import getenv

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Initialisation globale des dépendances
chemical_db = []
common_db = []  # Sous-ensemble pour fuzzywuzzy

@app.on_event("startup")
async def startup_event():
    global chemical_db, common_db
    try:
        # Chemin vers le fichier .env
        env_path = Path('.') / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        # Assurez-vous que le fichier .env est chargé avant d'importer les variables d'environnement
        TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        INCI_Catalog = pd.read_pickle('static/T_INCI_Catalog_Version_YY_2025_JJ_13_MM_02_HH_15_MN_57.pkl')
        chemical_db = INCI_Catalog['INCI'].tolist()
        common_db = chemical_db
        logger.info(f"Loaded INCI catalog with {len(chemical_db)} entries, common_db with {len(common_db)}")
    except Exception as e:
        logger.error(f"Error loading INCI catalog or environment: {str(e)}")

def validate_image_path(image_path: str) -> bool:
    abs_image_path = os.path.abspath(image_path)
    abs_upload_folder = os.path.abspath(UPLOAD_FOLDER)
    return abs_image_path.startswith(abs_upload_folder) and os.path.exists(image_path)

@lru_cache(maxsize=1000)
def get_top_chemicals(query: str, threshold: int = 85) -> str:
    query_lower = query.lower().strip()
    logger.debug(f"Processing query: {query_lower}")
    
    # Vérification exacte dans chemical_db (insensible à la casse)
    chemical_set = set(c.lower() for c in chemical_db)
    if query_lower in chemical_set:
        logger.debug(f"Exact match found in chemical_db for {query}: {query}")
        return query.capitalize()
    
    print("query_lower", query_lower)
    
    # Nettoyer la query pour calculer sa longueur (alphanumériques uniquement)
    query_cleaned = re.sub(r'[^a-zA-Z0-9]', '', query_lower)
    query_length = len(query_cleaned)
    logger.debug(f"Cleaned query length: {query_length} ('{query_cleaned}')")
    
    # Calculer les correspondances avec fuzzywuzzy sur common_db
    matches = []
    for chem in common_db:
        score = fuzz.WRatio(query_lower, chem.lower())
        matches.append((chem, score))
    
    # Trier par score décroissant et prendre les 3 meilleures
    matches.sort(key=lambda x: x[1], reverse=True)
    top_3_matches = matches[:3]
    
    # Logger les 3 meilleures alternatives
    logger.info(f"Top 3 matches for {query_lower}:")
    for chem, score in top_3_matches:
        logger.info(f"  - {chem} (score: {score})")
    
    # Sélectionner la meilleure correspondance, avec longueur la plus proche de la query en cas d'égalité
    best_fuzzy_match = "NF"
    best_fuzzy_score = 0
    best_fuzzy_diff = float('inf')  # Différence minimale initiale
    for chem, score in top_3_matches:
        if score >= threshold:
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy_match = chem
                best_fuzzy_diff = abs(len(chem) - query_length)
            elif score == best_fuzzy_score:
                current_diff = abs(len(chem) - query_length)
                if current_diff < best_fuzzy_diff:
                    best_fuzzy_match = chem
                    best_fuzzy_diff = current_diff
    if best_fuzzy_match != "NF":
        logger.debug(f"Best fuzzy match for {query}: {best_fuzzy_match} (score: {best_fuzzy_score}, diff: {best_fuzzy_diff})")
        return best_fuzzy_match
    return "NF"

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
        results = list(executor.map(
            lambda ing: get_top_chemicals(ing),
            unique_ingredients
        ))
    unique_ingredients = [match.title() if match != "NF" else ing for ing, match in zip(unique_ingredients, results)]
    logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds")
    
    return ', '.join(unique_ingredients) if unique_ingredients else "Aucun ingrédient détecté"

@app.get("/")
async def index(request: Request, extracted_text: str | None = None, image_path: str | None = None, error: str | None = None, has_submitted: bool = False, is_loading: bool = False):
    logger.info("GET request received for /")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "extracted_text": extracted_text,
            "image_path": image_path,
            "error": error,
            "has_submitted": has_submitted,
            "is_loading": is_loading
        }
    )

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
        start_time = time.time()
        extracted_text = process_inci_list(raw_text)
        logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds")
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