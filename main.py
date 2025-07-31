from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import pytesseract
from PIL import Image, ImageEnhance
import os
import shutil
import re
from collections import Counter
import pandas as pd
from fuzzywuzzy import process, fuzz
import time
import logging
import mimetypes
from urllib.parse import quote

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

# Configurer le chemin de Tesseract de manière portable
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def validate_image_path(image_path: str) -> bool:
    """Valide que le chemin de l'image est dans UPLOAD_FOLDER."""
    abs_image_path = os.path.abspath(image_path)
    abs_upload_folder = os.path.abspath(UPLOAD_FOLDER)
    return abs_image_path.startswith(abs_upload_folder) and os.path.exists(image_path)

def get_top_chemicals(query: str, chemical_db: list, top_n: int = 3, threshold: int = 85) -> str:
    """
    Find the top N closest matching chemical names from the database using fuzzy matching.
    """
    logger.debug(f"Fuzzy matching query: {query}")
    if not chemical_db:
        logger.warning("Chemical database is empty")
        return "NF"
    
    matches = process.extract(query, chemical_db, scorer=fuzz.WRatio, limit=top_n * 2)
    logger.debug(f"Initial matches: {matches}")
    
    filtered_matches = [match for match in matches if match[1] >= threshold]
    logger.debug(f"Filtered matches: {filtered_matches}")
    
    if not filtered_matches:
        logger.debug("No matches above threshold")
        return "NF"
    
    def sorting_criteria(match):
        name, score = match
        secondary_score = fuzz.ratio(query, name)
        return (-score, -secondary_score, -len(name))

    sorted_matches = sorted(filtered_matches, key=sorting_criteria)
    logger.debug(f"Sorted matches: {sorted_matches}")

    best_match = sorted_matches[0][0]
    logger.debug(f"Best match for {query}: {best_match}")
    return best_match

def detect_separator(text: str) -> str | None:
    """Détecte le séparateur principal parmi ',', ';', '.', '*'."""
    possible_separators = [',', ';', '.', '*']
    counts = Counter(char for char in text if char in possible_separators)
    logger.debug(f"Separator counts: {counts}")
    if not counts:
        logger.debug("No separator found")
        return None
    return max(counts, key=counts.get)

def process_inci_list(raw_text: str) -> str:
    """Traite la liste INCI extraite."""
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
    logger.debug(f"Cleaned ingredients: {cleaned_ingredients}")
    
    if not cleaned_ingredients:
        potential_ingredients = re.findall(r'\b[A-Z][a-zA-Z0-9\s-]*\b', cleaned_text)
        cleaned_ingredients = [
            re.sub(r'[^a-zA-Z0-9\s-]', '', ingredient).strip().capitalize()
            for ingredient in potential_ingredients
            if len(ingredient.strip()) > 2 and not ingredient.strip().isdigit()
        ]
        logger.debug(f"Alternative ingredients: {cleaned_ingredients}")
    
    seen = set()
    unique_ingredients = [ing for ing in cleaned_ingredients if not (ing.lower() in seen or seen.add(ing.lower()))]
    logger.debug(f"Unique ingredients before fuzzy matching: {unique_ingredients}")
    
    unique_ingredients = [
        match if match != "NF" else ing
        for ing in unique_ingredients
        for match in [get_top_chemicals(ing, chemical_db)]
    ]
    logger.debug(f"Unique INCI list after fuzzy matching: {unique_ingredients}")
    
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

    # Valider le type et la taille de l'image
    if image.size > MAX_FILE_SIZE:
        error = f"L'image dépasse la taille maximale de {MAX_FILE_SIZE // 1024 // 1024}MB"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}", status_code=303)

    content_type = image.content_type
    if not content_type or not content_type.startswith("image/"):
        error = "Le fichier n'est pas une image valide"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}", status_code=303)

    # Sauvegarder l'image avec un nom unique
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

    # OCR avec pytesseract
    try:
        logger.info("Attempting to open image for OCR")
        img = Image.open(image_path).convert('L')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        logger.info("Image opened and preprocessed successfully")
        raw_text = pytesseract.image_to_string(img, lang="eng+fra")
        logger.debug(f"Raw extracted text: {raw_text}")
        extracted_text = process_inci_list(raw_text)
        logger.info(f"Processed INCI list: {extracted_text}")
    except Exception as e:
        error = f"Erreur lors de l'extraction des ingrédients : {str(e)}"
        logger.error(error)

    # Rediriger vers GET / avec les résultats
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
            if file_age > 300:  # Supprimer les fichiers de plus de 5 minutes
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