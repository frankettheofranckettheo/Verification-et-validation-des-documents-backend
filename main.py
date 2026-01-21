import io
import numpy as np
import os
from typing import List, Optional
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageOps
import pytesseract

path_to_tesseract = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

if os.path.exists(path_to_tesseract):
    pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
    print(f"✅ Tesseract trouvé : {path_to_tesseract}")
else:
    print(f"⚠️ ATTENTION : Tesseract introuvable à : {path_to_tesseract}")
    print("   -> Installe-le ou corrige le chemin dans le code.")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
app = FastAPI(title="Document Analysis API")

# 1. Configuration CORS (Accepte TOUTES les requêtes comme demandé)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Autorise tout le monde
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes (GET, POST, etc.)
    allow_headers=["*"],  # Autorise tous les headers
)



# 2. Chargement du modèle YOLO
MODEL_PATH = "E:/ProjetVerificationValidationDesDocuments/Verification-et-validation-des-Documents/api_yolo/tout_mon_travail_yolo/modele_final_complet.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Modèle chargé : {MODEL_PATH}")
except Exception as e:
    print(f"❌ Erreur chargement modèle : {e}")
    model = None

# 3. Règles de validité (Années)
VALIDITY_RULES = {
    "PASSEPORT": 5,
    "CNI_NEW": 5,     # Nouvelles CNI
    "CNI_OLD": 10,    # Anciennes CNI
    "PERMIS": 10,
    "DEFAULT": 10
}

# Si besoin, spécifiez le chemin de tesseract ici pour Windows :
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ==============================================================================
# FONCTIONS UTILITAIRES POUR REMPLACER CV2
# ==============================================================================

def preprocess_image_for_ocr(image: Image.Image):
    """
    Prétraitement de l'image pour améliorer l'OCR (remplace les opérations cv2)
    
    Args:
        image: Image PIL
        
    Returns:
        Image PIL prétraitée
    """
    # 1. Conversion en niveaux de gris
    gray_image = ImageOps.grayscale(image)
    
    # 2. Augmentation du contraste (équivalent à la binarisation d'OpenCV)
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced = enhancer.enhance(2.0)  # Augmente le contraste
    
    # 3. Binarisation simple (seuillage)
    # Convertir en numpy pour appliquer un seuil
    img_array = np.array(enhanced)
    
    # Seuillage d'Otsu simplifié (ou seuil fixe)
    threshold = np.mean(img_array)  # Seuil moyen
    binary = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    
    # Reconvertir en PIL Image
    processed_image = Image.fromarray(binary)
    
    return processed_image

# ==============================================================================
# LOGIQUE MÉTIER - PARSING DE DATE (sans dateutil)
# ==============================================================================

def parse_date_flexible(date_string):
    """
    Parse une date de manière flexible avec plusieurs formats courants.
    Remplace dateutil.parser.parse()
    """
    if not date_string:
        return None
    
    date_string = date_string.strip()
    
    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d %b %Y",
        "%d %B %Y",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Format de date non reconnu: {date_string}")

def extract_dates(text):
    """Cherche des dates au format JJ/MM/AAAA ou JJ-MM-AAAA dans le texte OCR."""
    date_pattern = r'\b(\d{2}[/.-]\d{2}[/.-]\d{4})\b'
    matches = re.findall(date_pattern, text)
    
    dates = []
    for d in matches:
        try:
            # Normalisation des séparateurs
            d_clean = d.replace('.', '/').replace('-', '/')
            dt = parse_date_flexible(d_clean)
            dates.append(dt)
        except:
            pass
    return dates

def determine_validity(doc_type, extracted_text):
    """Calcule la validité basée sur le type et les dates trouvées."""
    dates = extract_dates(extracted_text)
    now = datetime.now()
    status = "INCONNU"
    details = "Aucune date lisible trouvée."
    expiry_date = None

    if not dates:
        return status, details, None

    max_date = max(dates)
    
    if max_date > now:
        expiry_date = max_date
        status = "VALIDE"
        details = f"Expire le {max_date.strftime('%d/%m/%Y')}"
    else:
        recent_dates = [d for d in dates if d.year > 1990]
        
        if recent_dates:
            delivery_date = max(recent_dates)
            
            duration = VALIDITY_RULES.get("DEFAULT", 10)
            if "PASSEPORT" in doc_type.upper(): 
                duration = VALIDITY_RULES["PASSEPORT"]
            elif "PERMIS" in doc_type.upper(): 
                duration = VALIDITY_RULES["PERMIS"]
            elif "NEW" in doc_type.upper(): 
                duration = VALIDITY_RULES["CNI_NEW"]
            
            calculated_expiry = delivery_date + timedelta(days=duration*365.25)
            expiry_date = calculated_expiry
            
            if calculated_expiry > now:
                status = "VALIDE"
                details = f"Calculé: Valide jusqu'au {calculated_expiry.strftime('%d/%m/%Y')} (Délivré le {delivery_date.strftime('%d/%m/%Y')})"
            else:
                status = "EXPIRÉ"
                details = f"Expiré depuis le {calculated_expiry.strftime('%d/%m/%Y')}"
        else:
            status = "EXPIRÉ (Probable)"
            details = "Dates trouvées semblent anciennes ou invalides."

    return status, details, expiry_date

# ==============================================================================
# ENDPOINTS
# ==============================================================================

@app.get("/")
def home():
    return {"message": "API Analyse Documents YOLO + OCR en ligne 🚀"}


# @app.post("/analyze")
# async def analyze_document(file: UploadFile = File(...)):
#     if not model:
#         raise HTTPException(status_code=500, detail="Modèle non chargé.")

#     try:
#         # 1. Lecture de l'image avec PIL (remplace cv2.imdecode)
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
        
#         # 2. Inférence YOLO (YOLO accepte directement les images PIL)
#         results = model(image)[0]
        
#         # Récupération de la meilleure détection
#         best_conf = 0
#         best_class = "Inconnu"
        
#         for box in results.boxes:
#             conf = float(box.conf[0])
#             if conf > best_conf:
#                 best_conf = conf
#                 cls_id = int(box.cls[0])
#                 best_class = model.names[cls_id]

#         # 3. Logique OCR conditionnelle
#         ocr_text = ""
#         validity_status = "NON APPLICABLE"
#         validity_details = "Document non reconnu ou pas de date nécessaire."
#         expiry_str = None

#         trigger_ocr_keywords = ["VERSO", "BACK", "ARRIERE", "DATA", "PASSEPORT", "PERMIS", "CNI"]
#         should_run_ocr = any(k in best_class.upper() for k in trigger_ocr_keywords)

#         if should_run_ocr:
#             # Prétraitement de l'image pour l'OCR (remplace les opérations cv2)
#             processed_image = preprocess_image_for_ocr(image)
            
#             # Configuration OCR
#             custom_config = r'--oem 3 --psm 6' 
#             ocr_text = pytesseract.image_to_string(processed_image, config=custom_config)
            
#             # Analyse Validité
#             validity_status, validity_details, exp_obj = determine_validity(best_class, ocr_text)
#             if exp_obj:
#                 expiry_str = exp_obj.strftime('%Y-%m-%d')

#         return {
#             "filename": file.filename,
#             "document_type": best_class,
#             "confidence": round(best_conf, 2),
#             "analysis": {
#                 "is_verso_or_data": should_run_ocr,
#                 "validity_status": validity_status,
#                 "details": validity_details,
#                 "expiration_date": expiry_str
#             },
#             # "raw_ocr": ocr_text.strip() # Décommenter pour debugger l'OCR
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    try:
        # 1. Lecture de l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        width, height = image.size
        
        # 2. Inférence YOLO
        results = model(image)[0]
        
        if len(results.boxes) == 0:
            return {
                "filename": file.filename,
                "document_type": "Aucune détection",
                "confidence": 0,
                "analysis": {"validity_status": "ERREUR", "details": "L'IA n'a rien vu sur l'image."}
            }

        # Séparation des détections
        ID_DOC_TYPES = [0, 10, 11, 15, 18, 22, 29, 30]
        ID_DATE_ZONES = [16, 17, 28]

        all_dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            all_dets.append({
                "id": cls_id,
                "conf": conf,
                "label": model.names[cls_id],
                "box": box.xyxy[0].tolist()
            })

        # --- CHOIX DU TYPE DE DOCUMENT ---
        # Priorité aux classes "Document"
        docs_uniquement = [d for d in all_dets if d["id"] in ID_DOC_TYPES]
        
        if docs_uniquement:
            best_det = max(docs_uniquement, key=lambda x: x["conf"])
        else:
            # Si pas de doc entier, on prend la meilleure détection quelle qu'elle soit
            best_det = max(all_dets, key=lambda x: x["conf"])

        best_class = best_det["label"]
        best_conf = best_det["conf"]

        # --- LOGIQUE OCR ---
        ocr_text = ""
        # On cherche une zone de date
        date_zones = [d for d in all_dets if d["id"] in ID_DATE_ZONES]

        if date_zones:
            # On crop la zone de date pour plus de précision
            d_box = max(date_zones, key=lambda x: x["conf"])["box"]
            # Sécurité pour le crop
            x1, y1, x2, y2 = max(0, d_box[0]-5), max(0, d_box[1]-5), min(width, d_box[2]+5), min(height, d_box[3]+5)
            roi = image.crop((x1, y1, x2, y2))
            ocr_text = pytesseract.image_to_string(preprocess_image_for_ocr(roi), config='--psm 7')
        
        # Si l'OCR de zone n'a rien donné ou pas de zone, on fait le scan global
        if len(ocr_text.strip()) < 5:
            processed_image = preprocess_image_for_ocr(image)
            ocr_text = pytesseract.image_to_string(processed_image, config='--psm 6')

        # --- VALIDITÉ ---
        validity_status, validity_details, exp_obj = determine_validity(best_class, ocr_text)
        expiry_str = exp_obj.strftime('%Y-%m-%d') if exp_obj else None

        # --- RÉPONSE (Format original respecté) ---
        return {
            "filename": file.filename,
            "document_type": best_class,
            "confidence": round(best_conf, 2),
            "analysis": {
                "is_verso_or_data": True, 
                "validity_status": validity_status,
                "details": validity_details,
                "expiration_date": expiry_str
            }
        }
        
    except Exception as e:
        # En cas d'erreur, on renvoie quand même une structure que votre testeur comprend
        return {
            "filename": file.filename,
            "document_type": "ERREUR",
            "confidence": 0,
            "analysis": {"validity_status": "ERREUR", "details": str(e)}
        }


@app.post("/analyze-full")
async def analyze_full_document(
    recto: UploadFile = File(...), 
    verso: UploadFile = File(None) # Le verso est optionnel pour certains docs, mais recommandé
):
    if not model:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    try:
        # --- 1. TRAITEMENT RECTO ---
        content_r = await recto.read()
        img_r = Image.open(io.BytesIO(content_r)).convert("RGB")
        res_r = model(img_r)[0]
        
        # Récupérer la meilleure classe du Recto
        class_r = "Inconnu"
        conf_r = 0.0
        if len(res_r.boxes) > 0:
            best_box_r = max(res_r.boxes, key=lambda x: x.conf[0])
            class_r = model.names[int(best_box_r.cls[0])]
            conf_r = float(best_box_r.conf[0])
            
        # OCR Recto
        processed_r = preprocess_image_for_ocr(img_r)
        text_r = pytesseract.image_to_string(processed_r, config='--psm 6')

        # --- 2. TRAITEMENT VERSO (si présent) ---
        class_v = "Non fourni"
        conf_v = 0.0
        text_v = ""
        
        if verso:
            content_v = await verso.read()
            img_v = Image.open(io.BytesIO(content_v)).convert("RGB")
            res_v = model(img_v)[0]
            
            if len(res_v.boxes) > 0:
                best_box_v = max(res_v.boxes, key=lambda x: x.conf[0])
                class_v = model.names[int(best_box_v.cls[0])]
                conf_v = float(best_box_v.conf[0])
            
            # OCR Verso
            processed_v = preprocess_image_for_ocr(img_v)
            text_v = pytesseract.image_to_string(processed_v, config='--psm 6')

        # --- 3. ANALYSE DE COHÉRENCE ---
        final_doc_type = class_r
        is_coherent = True
        coherence_msg = "Document cohérent"

        # Si on a un verso, on vérifie si les types correspondent (ex: CNI et CNI)
        if verso and class_r != "Inconnu" and class_v != "Inconnu":
            # On simplifie les noms pour la comparaison (ex: "CNI_RECTO" vs "CNI_VERSO")
            # Cette logique dépend de vos noms de classes YOLO exacts. 
            # Ici on suppose qu'ils contiennent un mot clé commun ou sont identiques.
            if class_r != class_v:
                # Si YOLO détecte "Passeport" au recto et "Permis" au verso -> Problème
                is_coherent = False
                coherence_msg = f"Incohérence : Recto ({class_r}) vs Verso ({class_v})"
            else:
                final_doc_type = class_r

        # --- 4. FUSION DES DONNÉES OCR ---
        # On combine le texte des deux côtés pour chercher les dates partout
        combined_text = text_r + "\n" + text_v
        
        validity_status, validity_details, exp_obj = determine_validity(final_doc_type, combined_text)
        expiry_str = exp_obj.strftime('%d/%m/%Y') if exp_obj else "Non trouvée"

        # --- 5. RÉSULTAT GLOBAL ---
        global_conf = (conf_r + conf_v) / 2 if verso else conf_r
        
        return {
            "is_valid_document": is_coherent and (validity_status == "VALIDE"),
            "document_type": final_doc_type,
            "confidence": round(global_conf, 2),
            "coherence": {
                "status": is_coherent,
                "message": coherence_msg,
                "recto_type": class_r,
                "verso_type": class_v
            },
            "validity": {
                "status": validity_status,
                "expiration_date": expiry_str,
                "details": validity_details
            },
            "extracted_data": {
                "ocr_recto_preview": text_r[:50].replace('\n', ' '),
                "ocr_verso_preview": text_v[:50].replace('\n', ' ')
            }
        }

    except Exception as e:
        print(f"Erreur : {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)