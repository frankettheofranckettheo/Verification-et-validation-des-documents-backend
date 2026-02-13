import io
import numpy as np
import os
import cv2
import re
from datetime import datetime, timedelta
from typing import List, Optional
from difflib import SequenceMatcher

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import pytesseract

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# Configuration Tesseract OCR
if os.name == 'nt':  # Windows
    path_to_tesseract = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
else:  # Linux (Render/Docker)
    path_to_tesseract = '/usr/bin/tesseract'

if os.path.exists(path_to_tesseract):
    pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
    print(f"✅ Tesseract trouvé : {path_to_tesseract}")
else:
    print(f"⚠️ ATTENTION : Tesseract introuvable à : {path_to_tesseract}")

app = FastAPI(title="Document Analysis API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, 
    allow_methods=["*"], allow_headers=["*"]
)

# Chargement YOLO
MODEL_PATH = "tout_mon_travail_yolo/modele_final_complet.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Modèle chargé : {MODEL_PATH}")
except Exception as e:
    print(f"❌ Erreur chargement modèle : {e}")
    model = None

# ==============================================================================
# 2. CONSTANTES
# ==============================================================================

DOC_TYPE_MAPPING = {
    0: "CNI_ANCIENNE_RECTO",
    22: "CNI_ANCIENNE_VERSO",
    10: "CNI_NOUVELLE_RECTO",
    11: "CNI_NOUVELLE_VERSO",
    29: "PERMIS_RECTO",
    30: "PERMIS_VERSO",
    18: "PASSEPORT_DATA",
    15: "PASSEPORT_COVER"
}

# Zones OCR
DATE_ZONE_IDS = [16, 17, 28] 
NAME_ZONE_RECTO_IDS = [3]  
NAME_ZONE_VERSO_IDS = [23, 9] 

STOPWORDS = {
    "REPUBLIQUE", "DU", "CAMEROUN", "REPUBLIC", "OF", "CAMEROON", 
    "NOM", "SURNAME", "PRENOMS", "GIVEN", "NAMES", 
    "DATE", "LIEU", "NAISSANCE", "BIRTH", "SEXE", "SEX", "TAILLE", "HEIGHT",
    "PROFESSION", "SIGNATURE", "TITULAIRE", "HOLDER",
    "PERE", "FATHER", "MERE", "MOTHER", "ADRESSE", "ADDRESS",
    "AUTORITE", "AUTHORITY", "IDENTIFICATION", "NATIONALE", "SECURITY",
    "CNI", "CARTE", "CARD", "VALIDE", "EXPIRATION", "DELIVRANCE",
    "NOMS", "PARENTS", "FILIATION"
}

VALIDITY_RULES = {
    "PASSEPORT": 5, 
    "CNI_NOUVELLE": 10, 
    "CNI_ANCIENNE": 10, 
    "PERMIS": 10, 
    "DEFAULT": 10
}

# ==============================================================================
# 3. FONCTIONS UTILITAIRES
# ==============================================================================

def preprocess_image_for_ocr(pil_image: Image.Image, is_crop=False):
    img_np = np.array(pil_image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    scale_factor = 1.0
    if is_crop:
        scale_factor = 3.0
    else:
        if width < 1000: scale_factor = 2.0
        elif width > 2500: scale_factor = 0.5
        
    if scale_factor != 1.0:
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    h_val = 15 if is_crop else 10
    if width < 2000 or is_crop:
        gray = cv2.fastNlMeansDenoising(gray, h=h_val, templateWindowSize=7, searchWindowSize=21)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )

    # --- NOUVEAUTÉ : Érosion légère pour détacher les barres du 4 ou nettoyer le 1 ---
    # Uniquement si on est en mode crop (zone ciblée) pour ne pas abîmer le reste
    if is_crop:
        kernel = np.ones((2,2), np.uint8) # Petit noyau
        thresh = cv2.erode(thresh, kernel, iterations=1)

    return Image.fromarray(thresh)

def extract_zone_text(image: Image.Image, results, target_ids: List[int], is_date=False):
    zone_text = ""
    width, height = image.size
    target_boxes = [box for box in results.boxes if int(box.cls[0]) in target_ids]
    target_boxes.sort(key=lambda x: float(x.conf[0]), reverse=True)
    
    for box in target_boxes:
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = max(0, xyxy[0]-5), max(0, xyxy[1]-5), min(width, xyxy[2]+5), min(height, xyxy[3]+5)
        crop = image.crop((x1, y1, x2, y2))
        processed_crop = preprocess_image_for_ocr(crop, is_crop=True)
        # CONFIGURATION TESSERACT SPÉCIALE
        if is_date:
            # Whitelist : Uniquement chiffres, point, tiret, slash, espace
            custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789./- '
        else:
            custom_config = r'--psm 6'
        text = pytesseract.image_to_string(processed_crop, config='--psm 6')
        zone_text += f" {text} "
    return zone_text.strip()

def perform_global_ocr(image: Image.Image):
    return pytesseract.image_to_string(
        preprocess_image_for_ocr(image, is_crop=False), 
        config='--psm 6'
    )

def verify_name_match(text_recto, text_verso):
    # Logique pour CNI ANCIENNE uniquement
    words_recto = re.findall(r'\b[A-Z]{3,}\b', text_recto.upper())
    candidates = [w for w in words_recto if w not in STOPWORDS]

    if not candidates:
        return False, "Aucun nom détecté au recto.", None

    text_verso_clean = text_verso.upper()
    
    for word in candidates:
        if word in text_verso_clean:
            return True, f"Nom '{word}' validé sur les 2 faces.", word
        
        words_verso = re.findall(r'\b[A-Z]{3,}\b', text_verso_clean)
        for v_word in words_verso:
            if v_word not in STOPWORDS:
                ratio = SequenceMatcher(None, word, v_word).ratio()
                if ratio > 0.80:
                    return True, f"Correspondance : '{word}' ≈ '{v_word}'", word

    return False, "Nom du recto introuvable au verso.", None

def parse_date_flexible(date_string):
    if not date_string: return None
    s = date_string.upper().replace('O', '0').replace('I', '1').replace('L', '1').replace('B', '8')
    s = re.sub(r'[^\d/\.-]', '', s)
    # --- CORRECTIF SPÉCIFIQUE 1 vs 4 ---
    
    # Cas 1 : Année "49xx" au lieu de "19xx" (Erreur très fréquente OCR)
    # Si on trouve "49" suivi de deux chiffres, on remplace par "19"
    # Ex: 12.05.4988 -> 12.05.1988
    s = re.sub(r'\b49(\d{2})\b', r'19\1', s)

    # Cas 2 : Année "40xx" au lieu de "20xx" (Plus rare mais possible)
    # Ex: 12.05.4023 -> 12.05.2023
    s = re.sub(r'\b40(\d{2})\b', r'20\1', s)
    formats = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d%m%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError: continue
    return None

def extract_dates(text):
    matches = re.findall(r'\b(\d{2}[/.-]\d{2}[/.-]\d{4})\b', text)
    valid_dates = []
    for m in matches:
        dt = parse_date_flexible(m)
        if dt and 1950 < dt.year < 2100: valid_dates.append(dt)
    return valid_dates

def determine_validity(full_doc_type, extracted_text):
    dates = extract_dates(extracted_text)
    now = datetime.now()
    if not dates: return "INCONNU", "Aucune date lisible.", None

    max_date = max(dates)
    if max_date > now:
        return "VALIDE", f"Expire le {max_date.strftime('%d/%m/%Y')}", max_date
    
    base_type = full_doc_type.replace("_RECTO", "").replace("_VERSO", "")
    duration = 10
    for key in VALIDITY_RULES:
        if key in base_type: duration = VALIDITY_RULES[key]; break
            
    past_dates = [d for d in dates if d <= now]
    if past_dates:
        delivery_date = max(past_dates)
        calculated_expiry = delivery_date + timedelta(days=duration*365.25)
        if calculated_expiry > now:
            return "VALIDE", f"Calculé: Valide jusqu'au {calculated_expiry.strftime('%d/%m/%Y')}", calculated_expiry
        return "EXPIRÉ", f"Expiré depuis le {calculated_expiry.strftime('%d/%m/%Y')}", calculated_expiry
            
    return "EXPIRÉ", "Dates passées.", None

def get_doc_info(results):
    best_conf = 0
    doc_type = "INCONNU"
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in DOC_TYPE_MAPPING and conf > best_conf:
            best_conf = conf
            doc_type = DOC_TYPE_MAPPING[cls_id]
    return doc_type, best_conf

# ==============================================================================
# 4. ENDPOINT PRINCIPAL (CORRIGÉ)
# ==============================================================================

@app.post("/analyze-full")
async def analyze_full_document(
    recto: UploadFile = File(...), 
    verso: UploadFile = File(None)
):
    if not model: raise HTTPException(status_code=500, detail="Modèle non chargé.")

    try:
        # Variables OCR
        full_text_r, full_text_v = "", ""
        name_text_r, name_text_v = "", ""
        
        doc_type_r, conf_r = "INCONNU", 0.0
        doc_type_v, conf_v = "INCONNU", 0.0

        # --- 1. TRAITEMENT RECTO ---
        content_r = await recto.read()
        img_r = Image.open(io.BytesIO(content_r)).convert("RGB")
        res_r = model(img_r)[0]
        
        doc_type_r, conf_r = get_doc_info(res_r)
        print(f"DEBUG: Type détecté RECTO = {doc_type_r}") # DEBUG
        
        full_text_r = perform_global_ocr(img_r)
        full_text_r += " " + extract_zone_text(img_r, res_r, DATE_ZONE_IDS, is_date=True)
        name_text_r = extract_zone_text(img_r, res_r, NAME_ZONE_RECTO_IDS, is_date=False)
        if len(name_text_r) < 5: name_text_r = full_text_r 

        # --- 2. TRAITEMENT VERSO ---
        if verso:
            content_v = await verso.read()
            img_v = Image.open(io.BytesIO(content_v)).convert("RGB")
            res_v = model(img_v)[0]
            
            doc_type_v, conf_v = get_doc_info(res_v)
            print(f"DEBUG: Type détecté VERSO = {doc_type_v}") # DEBUG
            
            full_text_v = perform_global_ocr(img_v)
            full_text_v += " " + extract_zone_text(img_v, res_v, DATE_ZONE_IDS, is_date=True)
            name_text_v = extract_zone_text(img_v, res_v, NAME_ZONE_VERSO_IDS, is_date=False)
            if len(name_text_v) < 5: name_text_v = full_text_v

        # --- 3. COHÉRENCE (LOGIQUE REVUE ET CORRIGÉE) ---
        
        # Fonctions helper locales
        def get_family(name): return name.replace("_RECTO", "").replace("_VERSO", "")
        def get_side(name): return "RECTO" if "RECTO" in name else "VERSO" if "VERSO" in name else "AUTRE"

        fam_r = get_family(doc_type_r)
        final_type = fam_r # Type final présumé
        
        is_coherent = False
        msg_coherence = "Analyse..."

        # CAS 1 : PASSEPORT
        if "PASSEPORT" in doc_type_r:
            is_coherent = True
            msg_coherence = "✅ Passeport détecté."
            final_type = "PASSEPORT"

        # CAS 2 : RECTO + VERSO
        elif verso:
            fam_v = get_family(doc_type_v)
            side_r = get_side(doc_type_r)
            side_v = get_side(doc_type_v)

            # Vérification de base : Est-ce que les types correspondent ?
            types_match = (fam_r == fam_v)
            sides_diff = (side_r != side_v)
            
            # Gestion d'erreur de base
            if doc_type_r == "INCONNU" or doc_type_v == "INCONNU":
                is_coherent = False
                msg_coherence = "❌ Document non reconnu par le modèle IA."
            elif not types_match:
                is_coherent = False
                msg_coherence = f"❌ Incohérence Type : {fam_r} (Recto) vs {fam_v} (Verso)."
            elif not sides_diff:
                is_coherent = False
                msg_coherence = f"❌ Erreur Structure : Vous avez envoyé deux fois le {side_r}."
            
            else:
                # La structure (Famille + Côtés) est bonne.
                # Maintenant on applique les règles spécifiques demandées.

                if "CNI_NOUVELLE" in fam_r:
                    # ==========================================================
                    # C'EST ICI QUE CA SE JOUE
                    # SI CNI_NOUVELLE EST BIEN DÉTECTÉ, ON FORCE TRUE
                    # ==========================================================
                    is_coherent = True
                    msg_coherence = "✅ CNI Nouvelle : Validation structurelle OK."
                
                elif "PERMIS" in fam_r:
                    is_coherent = True
                    msg_coherence = "✅ Permis : Validation structurelle OK."
                
                elif "CNI_ANCIENNE" in fam_r:
                    # Seule la CNI Ancienne passe par la vérification de nom
                    match, name_msg, _ = verify_name_match(name_text_r, name_text_v)
                    if match:
                        is_coherent = True
                        msg_coherence = f"✅ CNI Ancienne : Noms correspondants ({name_msg})."
                    else:
                        is_coherent = False
                        msg_coherence = f"❌ CNI Ancienne invalide : {name_msg}"
                
                else:
                    # Cas par défaut (si vous ajoutez d'autres docs plus tard)
                    is_coherent = True
                    msg_coherence = "✅ Document reconnu."

        else:
            # Verso manquant pour un doc qui n'est pas un passeport
            is_coherent = False
            msg_coherence = "❌ Verso manquant (Requis pour ce document)."

        # --- 4. VALIDITÉ (DATES) ---
        combined_full_text = full_text_r + "\n" + full_text_v
        status, details, exp_date = determine_validity(final_type, combined_full_text)

        # RÉSULTAT FINAL
        # Valide seulement si (Cohérent) ET (Date Valide)
        final_validity = is_coherent and (status == "VALIDE")

        return {
            "is_valid_document": final_validity,
            "document_type": final_type,
            "confidence": round((conf_r + conf_v)/2 if verso else conf_r, 2),
            "coherence": {
                "status": is_coherent,
                "message": msg_coherence,
                "debug_detected_type": doc_type_r # Pour vous aider à voir si YOLO se trompe
            },
            "validity": {
                "status": status,
                "expiration_date": exp_date.strftime('%d/%m/%Y') if exp_date else None,
                "details": details
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)