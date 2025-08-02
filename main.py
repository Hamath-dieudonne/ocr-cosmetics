import os
import re
from collections import Counter
import pandas as pd
from functools import lru_cache
import time
import logging
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from PIL import Image, ImageEnhance
import pytesseract
import shutil
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from pathlib import Path
from os import getenv
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from cryptography.fernet import Fernet

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["your-app.onrender.com"])
app.mount("/static", StaticFiles(directory="static", exclude=["inci_only.pkl.enc"]), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Authentification de base
security = HTTPBasic()
def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "colleague"
    correct_password = "securepassword"
    if credentials.username != correct_username or credentials.password != correct_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return credentials.username

# Clé de chiffrement pour .pkl et images
FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    raise ValueError("FERNET_KEY must be set in environment variables")
cipher = Fernet(FERNET_KEY.encode())

def encrypt_file(file_path):
    logger.info(f"Encrypting file: {file_path}")
    with open(file_path, "rb") as f:
        data = f.read()
    encrypted_data = cipher.encrypt(data)
    encrypted_path = file_path + ".enc"
    with open(encrypted_path, "wb") as f:
        f.write(encrypted_data)
    os.remove(file_path)  # Supprime l'original
    return encrypted_path

def decrypt_file(encrypted_path):
    logger.info(f"Decrypting file: {encrypted_path}")
    with open(encrypted_path, "rb") as f:
        encrypted_data = f.read()
    decrypted_data = cipher.decrypt(encrypted_data)
    original_path = encrypted_path.replace(".enc", "")
    with open(original_path, "wb") as f:
        f.write(decrypted_data)
    os.remove(encrypted_path)  # Supprime le fichier chiffré après usage
    return original_path

@app.on_event("startup")
async def startup_event():
    global chemical_db, common_db
    try:
        env_path = Path('.') / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        # Déchiffrer et charger le fichier .pkl
        encrypted_path = 'static/inci_only.pkl.enc'
        if os.path.exists(encrypted_path):
            with open(encrypted_path, "rb") as f:
                encrypted_data = f.read()
            decrypted_data = cipher.decrypt(encrypted_data)
            with open("temp.pkl", "wb") as f:
                f.write(decrypted_data)
            INCI_Catalog = pd.read_pickle("temp.pkl")
            os.remove("temp.pkl")
            chemical_db = INCI_Catalog['INCI'].tolist()
            common_db = chemical_db
            logger.info(f"Loaded INCI catalog with {len(chemical_db)} entries, common_db with {len(common_db)}")
        else:
            raise FileNotFoundError("Encrypted .pkl file not found")
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
    chemical_set = set(c.lower() for c in chemical_db)
    if query_lower in chemical_set:
        logger.debug(f"Exact match found in chemical_db for {query}: {query}")
        return query.capitalize()
    print("query_lower", query_lower)
    query_cleaned = re.sub(r'[^a-zA-Z0-9]', '', query_lower)
    query_length = len(query_cleaned)
    logger.debug(f"Cleaned query length: {query_length} ('{query_cleaned}')")
    matches = [(chem, fuzz.WRatio(query_lower, chem.lower())) for chem in common_db]
    matches.sort(key=lambda x: x[1], reverse=True)
    top_3_matches = matches[:3]
    logger.info(f"Top 3 matches for {query_lower}:")
    for chem, score in top_3_matches:
        logger.info(f"  - {chem} (score: {score})")
    best_fuzzy_match = "NF"
    best_fuzzy_score = 0
    best_fuzzy_diff = float('inf')
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
async def index(request: Request, extracted_text: str | None = None, image_path: str | None = None, error: str | None = None, has_submitted: bool = False, is_loading: bool = False, username: str = Depends(get_current_username)):
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
async def index(request: Request, image: UploadFile = File(None), username: str = Depends(get_current_username)):
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
        # Chiffrer l'image après upload
        encrypted_path = encrypt_file(image_path)
        image_path = encrypted_path  # Mettre à jour le chemin pour le traitement
    except Exception as e:
        error = f"Erreur lors de la sauvegarde de l'image : {str(e)}"
        logger.error(error)
        return RedirectResponse(url=f"/?error={quote(error)}&has_submitted=true", status_code=303)

    try:
        logger.info("Attempting to open image for OCR")
        # Déchiffrer l'image pour traitement
        decrypted_path = decrypt_file(image_path)
        img = Image.open(decrypted_path).convert('L')
        img = ImageEnhance.Contrast(img).enhance(2.0)
        logger.info("Image opened and preprocessed successfully")
        raw_text = pytesseract.image_to_string(img, lang="eng+fra")
        logger.debug(f"Raw extracted text: {raw_text}")
        start_time = time.time()
        extracted_text = process_inci_list(raw_text)
        logger.info(f"Processed INCI list in {time.time() - start_time:.2f} seconds")
        os.remove(decrypted_path)  # Nettoyer le fichier déchiffré
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
async def cleanup(username: str = Depends(get_current_username)):
    logger.info("Cleaning up old images")
    current_time = time.time()
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path) and file_path.endswith(".enc"):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > 300:
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted old encrypted file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting old file {file_path}: {str(e)}")
    return {"message": "Cleanup completed"}

@app.post("/delete_image")
async def delete_image(request: Request, username: str = Depends(get_current_username)):
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