# import io
# import numpy as np
# import os
# import cv2  # On réintroduit OpenCV pour la puissance de traitement
# import re
# from difflib import SequenceMatcher
# from datetime import datetime, timedelta
# from typing import List, Optional
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from ultralytics import YOLO
# from PIL import Image
# import pytesseract

# # --- CONFIG TESSERACT ---
# path_to_tesseract = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
# if os.path.exists(path_to_tesseract):
#     pytesseract.pytesseract.tesseract_cmd = path_to_tesseract
#     print(f"✅ Tesseract trouvé : {path_to_tesseract}")
# else:
#     print(f"⚠️ ATTENTION : Tesseract introuvable à : {path_to_tesseract}")

# app = FastAPI(title="Document Analysis API")

# # 1. CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 2. Chargement Modèle
# MODEL_PATH = "E:/ProjetVerificationValidationDesDocuments/Verification-et-validation-des-Documents/api_yolo/tout_mon_travail_yolo/modele_final_complet.pt"
# try:
#     model = YOLO(MODEL_PATH)
#     print(f"✅ Modèle chargé : {MODEL_PATH}")
# except Exception as e:
#     print(f"❌ Erreur chargement modèle : {e}")
#     model = None

# # ==============================================================================
# # CLASSES ET MAPPING (Basé sur vos labels)
# # ==============================================================================

# # Mapping pour l'identification du document
# DOC_TYPE_MAPPING = {
#     0: "CNI_ANCIENNE_RECTO",      # Recto
#     22: "CNI_ANCIENNE_VERSO",     # Verso
#     10: "CNI_NOUVELLE_RECTO",     # Recto
#     11: "CNI_NOUVELLE_VERSO",     # Verso
#     29: "PERMIS_CAMEROUN_RECTO",  # Recto
#     30: "PERMIS_CAMEROUN_VERSO",  # Verso
#     18: "PASSEPORT_DATA",         # Intérieur
#     15: "PASSEPORT_COVER"         # Couverture
# }

# # IDs des zones contenant spécifiquement des dates (pour l'OCR ciblé)
# DATE_ZONE_IDS = [16, 17, 28] 

# # Règles de validité
# VALIDITY_RULES = {
#     "PASSEPORT": 5,
#     "CNI_NOUVELLE": 5,
#     "CNI_ANCIENNE": 10,
#     "PERMIS_CAMEROUN": 10,
#     "DEFAULT": 10
# }


# # Zones où se trouvent les NOMS (pour la cohérence)
# # 3 = Informations personnels (Recto)
# # 23 = Bloc de gauche (Verso - contient souvent les parents)
# # On ajoute aussi 9 (Bloc droite) au cas où, selon le modèle de carte
# NAME_ZONE_RECTO_IDS = [3]  
# NAME_ZONE_VERSO_IDS = [23, 9] 



# # ==============================================================================
# # PRÉTRAITEMENT AVANCÉ (RETOUR À OPENCV)
# # ==============================================================================

# def preprocess_image_for_ocr(pil_image: Image.Image, is_crop=False):
#     """
#     Utilise OpenCV pour un prétraitement robuste (Upscaling + Adaptive Threshold) et redimensionnement intelligent.
#     """
#     # 1. Conversion PIL -> OpenCV (RGB -> BGR)
#     img_np = np.array(pil_image)
#     img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

#     # 2. Conversion Niveaux de gris
#     gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

#     # 2. Redimensionnement INTELLIGENT
#     height, width = gray.shape
    
#     scale_factor = 1.0

#     # 3. Redimensionnement (Upscaling)
#     # Tesseract fonctionne beaucoup mieux si le texte est gros.
#     # On double la taille de l'image (surtout important pour les crops de dates)
#     if is_crop:
#         # Pour les petits crops (dates), on zoom toujours fortement
#         scale_factor = 3.0
#     else:
#         # Pour l'image complète
#         if width < 1000:
#             # Petite image (Webcam) -> On agrandit pour aider l'OCR
#             scale_factor = 2.0
#         elif width > 2500:
#             # Très grande image (Photo 4K/12MP) -> On RÉDUIT pour la vitesse
#             # Tesseract n'a pas besoin de 4000px de large pour lire du texte standard
#             scale_factor = 0.5 
#         else:
#             # Taille moyenne -> On garde tel quel
#             scale_factor = 1.0    
    
#     if scale_factor != 1.0:
#         gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

#     # 4. Suppression du bruit (Denoising léger)
#     h_val = 15 if is_crop else 10
#     if width <2000:
#         gray = cv2.fastNlMeansDenoising(gray, h=h_val, templateWindowSize=7, searchWindowSize=21)

#     # 5. Seuillage Adaptatif (Le secret pour les hologrammes/reflets)
#     # Contrairement au seuil global, celui-ci calcule le seuil pour chaque petite zone.
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, 
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#         cv2.THRESH_BINARY, 
#         31, # Taille du bloc (voisinage)
#         12  # Constante soustraite (plus elle est haute, moins il y a de bruit noir)
#     )

#     # Retour en format PIL pour Tesseract
#     return Image.fromarray(thresh)

# # ==============================================================================
# # LOGIQUE MÉTIER & OCR
# # ==============================================================================

# def parse_date_flexible(date_string):
#     if not date_string: return None
#     # Nettoyage des erreurs OCR courantes (l/1, O/0, etc)
#     s = date_string.upper().replace('O', '0').replace('I', '1').replace('L', '1').replace('B', '8')
#     # Garder uniquement chiffres et séparateurs
#     s = re.sub(r'[^\d/\.-]', '', s)
    
#     formats = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d%m%Y"]
    
#     for fmt in formats:
#         try:
#             return datetime.strptime(s, fmt)
#         except ValueError:
#             continue
#     return None

# def extract_dates(text):
#     # Regex pour capturer JJ/MM/AAAA même si l'OCR a fait des erreurs (ex: 12/05/202a)
#     matches = re.findall(r'\b(\d{2}[/.-]\d{2}[/.-]\d{4})\b', text)
#     valid_dates = []
#     for m in matches:
#         dt = parse_date_flexible(m)
#         if dt and 1950 < dt.year < 2100: # Filtre les dates absurdes
#             valid_dates.append(dt)
#     return valid_dates

# def determine_validity(doc_type, extracted_text):
#     dates = extract_dates(extracted_text)
#     now = datetime.now()
    
#     if not dates:
#         return "INCONNU", "Aucune date lisible trouvée.", None

#     # Stratégie : La date la plus lointaine est souvent l'expiration
#     max_date = max(dates)
    
#     # Si la date max est dans le futur, c'est probablement l'expiration
#     if max_date > now:
#         return "VALIDE", f"Expire le {max_date.strftime('%d/%m/%Y')}", max_date
    
#     # Sinon, on essaie de calculer l'expiration depuis la date de délivrance (la plus récente passée)
#     past_dates = [d for d in dates if d <= now]
#     if past_dates:
#         delivery_date = max(past_dates)
#         duration = VALIDITY_RULES.get(doc_type, 10)
        
#         calculated_expiry = delivery_date + timedelta(days=duration*365.25)
        
#         if calculated_expiry > now:
#             return "VALIDE", f"Calculé: Valide jusqu'au {calculated_expiry.strftime('%d/%m/%Y')}", calculated_expiry
#         else:
#             return "EXPIRÉ", f"Expiré depuis le {calculated_expiry.strftime('%d/%m/%Y')}", calculated_expiry
            
#     return "EXPIRÉ", "Toutes les dates visibles sont passées.", None

# def perform_smart_ocr(image: Image.Image, results):
#     """
#     Combine l'OCR global et l'OCR ciblé sur les zones détectées par YOLO.
#     """
#     # 1. OCR Global (utile pour le contexte)
#     full_text = pytesseract.image_to_string(
#         preprocess_image_for_ocr(image, is_crop=False), 
#         config='--psm 6'
#     )
    
#     # 2. OCR Ciblé sur les zones de dates (IDs 16, 17, 28)
#     # C'est LA clé pour la précision : on découpe juste la date
#     dates_text = ""
#     width, height = image.size
    
#     detected_zones = [box for box in results.boxes if int(box.cls[0]) in DATE_ZONE_IDS]
    
#     for box in detected_zones:
#         # Coordonnées avec petite marge
#         xyxy = box.xyxy[0].tolist()
#         x1, y1, x2, y2 = max(0, xyxy[0]-5), max(0, xyxy[1]-5), min(width, xyxy[2]+5), min(height, xyxy[3]+5)
        
#         # Crop
#         crop = image.crop((x1, y1, x2, y2))
        
#         # Prétraitement agressif pour le crop (Zoom x3)
#         processed_crop = preprocess_image_for_ocr(crop, is_crop=True)
        
#         # PSM 7 = Traiter comme une seule ligne de texte (idéal pour une date isolée)
#         text = pytesseract.image_to_string(processed_crop, config='--psm 7 -c tessedit_char_whitelist=0123456789/-.')
#         dates_text += f" {text} "
        
#     return full_text + "\n" + dates_text

# def get_doc_info(results):
#     """Récupère le type de document basé sur la plus haute confiance"""
#     best_conf = 0
#     doc_type = "INCONNU"
    
#     for box in results.boxes:
#         cls_id = int(box.cls[0])
#         conf = float(box.conf[0])
        
#         if cls_id in DOC_TYPE_MAPPING and conf > best_conf:
#             best_conf = conf
#             doc_type = DOC_TYPE_MAPPING[cls_id]
            
#     return doc_type, best_conf

# def extract_zone_text(image: Image.Image, results, target_ids: List[int]):
#     """
#     Extrait spécifiquement le texte des zones ciblées (ex: ID 3 ou 23).
#     Retourne le texte concaténé de ces zones.
#     """
#     zone_text = ""
#     width, height = image.size
    
#     # Trouver les boîtes qui correspondent aux IDs demandés
#     target_boxes = [box for box in results.boxes if int(box.cls[0]) in target_ids]
    
#     # Trier par confiance (garder les meilleures détections)
#     target_boxes.sort(key=lambda x: float(x.conf[0]), reverse=True)
    
#     for box in target_boxes:
#         xyxy = box.xyxy[0].tolist()
#         # Marge de sécurité
#         x1, y1, x2, y2 = max(0, xyxy[0]-5), max(0, xyxy[1]-5), min(width, xyxy[2]+5), min(height, xyxy[3]+5)
        
#         crop = image.crop((x1, y1, x2, y2))
        
#         # Traitement spécifique "Crop" (Zoom x3)
#         processed_crop = preprocess_image_for_ocr(crop, is_crop=True)
        
#         # PSM 6 = Bloc de texte (mieux pour 'Nom Prénoms' sur plusieurs lignes)
#         text = pytesseract.image_to_string(processed_crop, config='--psm 6')
#         zone_text += f" {text} "
        
#     return zone_text.strip()



# # Liste des mots à ignorer pour ne garder que les noms propres potentiels
# STOPWORDS = {
#     "REPUBLIQUE", "DU", "CAMEROUN", "REPUBLIC", "OF", "CAMEROON", 
#     "NOM", "SURNAME", "PRENOMS", "GIVEN", "NAMES", 
#     "DATE", "LIEU", "NAISSANCE", "BIRTH", "SEXE", "SEX", "TAILLE", "HEIGHT",
#     "PROFESSION", "SIGNATURE", "TITULAIRE", "HOLDER",
#     "PERE", "FATHER", "MERE", "MOTHER", "ADRESSE", "ADDRESS",
#     "AUTORITE", "AUTHORITY", "IDENTIFICATION", "NATIONALE", "SECURITY",
#     "CNI", "CARTE", "CARD", "VALIDE", "EXPIRATION"
# }

# def verify_name_match(text_recto, text_verso):
#     """
#     Tente de trouver le nom de famille du Recto dans le texte du Verso.
#     Retourne (Succès, Détails, Mot_Trouvé).
#     """
#     # 1. Nettoyage et Extraction des candidats du Recto
#     # On garde les mots en majuscules, de plus de 2 lettres, qui ne sont pas des mots-clés
#     words_recto = re.findall(r'\b[A-Z]{3,}\b', text_recto.upper())
#     candidates = [w for w in words_recto if w not in STOPWORDS]

#     if not candidates:
#         return False, "Recto illisible ou verso illisible", None

#     # 2. Préparation du Verso
#     text_verso_clean = text_verso.upper()

#     # 3. Recherche floue (Fuzzy Matching)
#     # On cherche si l'un des candidats du Recto ressemble fortement à un mot du Verso
#     best_match_word = None
#     best_ratio = 0.0

#     for word in candidates:
#         # Si le mot est EXACTEMENT dans le verso, c'est gagné
#         if word in text_verso_clean:
#             return True, f"Nom '{word}' validé sur les deux faces.", word
        
#         # Sinon, on teste mot par mot dans le verso pour voir s'il y a une ressemblance > 85%
#         # (Utile si l'OCR lit 'MBARGA' au recto et 'MBAR6A' au verso)
#         words_verso = re.findall(r'\b[A-Z]{3,}\b', text_verso_clean)
#         for v_word in words_verso:
#             if v_word not in STOPWORDS:
#                 ratio = SequenceMatcher(None, word, v_word).ratio()
#             if ratio > 0.85: # Seuil de tolérance
#                 return True, f"Correspondance trouvée : '{word}' ≈ '{v_word}'", word

#     return False, "Le nom du recto ne semble pas apparaître au verso.", None



# # ==============================================================================
# # ENDPOINT PRINCIPAL
# # ==============================================================================

# @app.post("/analyze-full")
# async def analyze_full_document(
#     recto: UploadFile = File(...), 
#     verso: UploadFile = File(None)
# ):
#     if not model: raise HTTPException(status_code=500, detail="Modèle non chargé.")

#     try:
#         combined_text = ""
#         text_r = ""
#         text_v = ""
#         doc_type_r = "INCONNU"
#         doc_type_v = "INCONNU"
#         conf_r = 0
#         conf_v = 0

#         # --- TRAITEMENT RECTO ---
#         content_r = await recto.read()
#         img_r = Image.open(io.BytesIO(content_r)).convert("RGB")
#         res_r = model(img_r)[0]
        
#         doc_type_r, conf_r = get_doc_info(res_r)
#         combined_text += perform_smart_ocr(img_r, res_r)

#         # Si YOLO n'a pas vu la zone 3, on se rabat sur le texte global
#         if len(name_text_r) < 5: 
#             name_text_r = combined_text 


#         # --- TRAITEMENT VERSO (Optionnel) ---
#         if verso:
#             content_v = await verso.read()
#             img_v = Image.open(io.BytesIO(content_v)).convert("RGB")
#             res_v = model(img_v)[0]
            
#             doc_type_v, conf_v = get_doc_info(res_v)
#             combined_text += "\n" + perform_smart_ocr(img_v, res_v)


#         # --- COHÉRENCE INTELLIGENTE & SEMANTIQUE---
#         final_type = doc_type_r
#         is_coherent = False
#         msg_coherence = "Analyse en cours..."

#         # Fonctions helper (gardez-les ou mettez-les en dehors)
#         def get_doc_side(name):
#             if "RECTO" in name: return "RECTO"
#             if "VERSO" in name: return "VERSO"
#             return "AUTRE"

#         def get_doc_family(name):
#             return name.replace("_RECTO", "").replace("_VERSO", "")

#         # A. CAS PASSEPORT
#         if "PASSEPORT" in doc_type_r:
#             is_coherent = True
#             msg_coherence = "Passeport détecté."
            
#         # B. CAS CNI / PERMIS (Nécessite Recto + Verso)
#         elif verso:
#             # 1. Vérification Structurelle (YOLO)
#             family_r = get_doc_family(doc_type_r)
#             side_r = get_doc_side(doc_type_r)
#             family_v = get_doc_family(doc_type_v)
#             side_v = get_doc_side(doc_type_v)

#             if doc_type_r == "INCONNU" or doc_type_v == "INCONNU":
#                  is_coherent = False
#                  msg_coherence = "Type de document non reconnu."

#             if family_r != family_v:
#                 is_coherent = False
#                 msg_coherence = f"Erreur Mixte : Recto {family_r} vs Verso {family_v}."
            
#             elif side_r == side_v:
#                 is_coherent = False
#                 msg_coherence = f"Fraude suspectée : Deux fois le {side_r}."

#             else:
#                 # La structure est bonne (Même famille, côtés opposés).
#                 # MAINTENANT, ON LANCE LA VÉRIFICATION DU NOM (OCR)
                
#                 name_match, name_msg, _ = verify_name_match(text_r, text_v) # combined_text_r = text_r
                
#                 if name_match:
#                     is_coherent = True
#                     msg_coherence = f"✅ Document Authentifié. {name_msg}"
#                     final_type = family_r
#                 else:
#                     # Ici, vous avez le choix : Rejeter ou Avertir
#                     # Pour un système strict, on met False.
#                     is_coherent = False 
#                     msg_coherence = f"❌ Incohérence Textuelle : {name_msg} (Vérifiez la netteté)."

#         else:
#             is_coherent = False
#             msg_coherence = "Verso manquant."

       
#         # --- VALIDITÉ ---
#         full_text = text_r + "\n" + text_v
#         status, details, exp_date = determine_validity(final_type, full_text)

#         return {
#             "is_valid_document": is_coherent and (status == "VALIDE"),
#             "document_type": final_type.replace("_RECTO", ""),
#             "confidence": round((conf_r + conf_v)/2 if verso else conf_r, 2),
#             "coherence": {
#                 "status": is_coherent,
#                 "message": msg_coherence
#             },
#             "validity": {
#                 "status": status,
#                 "expiration_date": exp_date.strftime('%d/%m/%Y') if exp_date else None,
#                 "details": details
#             },
#             # "debug_text": combined_text # Décommenter pour voir ce que l'OCR lit
#         }

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)







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
path_to_tesseract = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

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
MODEL_PATH = "E:/ProjetVerificationValidationDesDocuments/Verification-et-validation-des-Documents/api_yolo/tout_mon_travail_yolo/modele_final_complet.pt"
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Modèle chargé : {MODEL_PATH}")
except Exception as e:
    print(f"❌ Erreur chargement modèle : {e}")
    model = None

# ==============================================================================
# 2. CONSTANTES & CONFIGURATION CIBLÉE
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

# --- ZONES CLÉS POUR L'OCR ---
# Zones où se trouvent les dates (pour la validité)
DATE_ZONE_IDS = [16, 17, 28] 

# Zones où se trouvent les NOMS (pour la cohérence)
# 3 = Informations personnels (Recto)
# 23 = Bloc de gauche (Verso - contient souvent les parents)
# On ajoute aussi 9 (Bloc droite) au cas où, selon le modèle de carte
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

VALIDITY_RULES = {"PASSEPORT": 5, "CNI_NOUVELLE": 10, "CNI_ANCIENNE": 10, "PERMIS": 10, "DEFAULT": 10}

# ==============================================================================
# 3. FONCTIONS DE TRAITEMENT
# ==============================================================================

def preprocess_image_for_ocr(pil_image: Image.Image, is_crop=False):
    """
    Prétraitement OpenCV.
    Si is_crop=True (Zone ciblée), on applique un zoom x3 agressif.
    """
    img_np = np.array(pil_image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    scale_factor = 1.0
    if is_crop:
        scale_factor = 3.0 # ZOOM PUISSANT pour les zones ciblées
    else:
        if width < 1000: scale_factor = 2.0
        elif width > 2500: scale_factor = 0.5
        
    if scale_factor != 1.0:
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Denoising un peu plus fort sur les crops pour nettoyer le fond des cartes
    h_val = 15 if is_crop else 10
    if width < 2000 or is_crop:
        gray = cv2.fastNlMeansDenoising(gray, h=h_val, templateWindowSize=7, searchWindowSize=21)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
    )
    return Image.fromarray(thresh)

def extract_zone_text(image: Image.Image, results, target_ids: List[int]):
    """
    Extrait spécifiquement le texte des zones ciblées (ex: ID 3 ou 23).
    Retourne le texte concaténé de ces zones.
    """
    zone_text = ""
    width, height = image.size
    
    # Trouver les boîtes qui correspondent aux IDs demandés
    target_boxes = [box for box in results.boxes if int(box.cls[0]) in target_ids]
    
    # Trier par confiance (garder les meilleures détections)
    target_boxes.sort(key=lambda x: float(x.conf[0]), reverse=True)
    
    for box in target_boxes:
        xyxy = box.xyxy[0].tolist()
        # Marge de sécurité
        x1, y1, x2, y2 = max(0, xyxy[0]-5), max(0, xyxy[1]-5), min(width, xyxy[2]+5), min(height, xyxy[3]+5)
        
        crop = image.crop((x1, y1, x2, y2))
        
        # Traitement spécifique "Crop" (Zoom x3)
        processed_crop = preprocess_image_for_ocr(crop, is_crop=True)
        
        # PSM 6 = Bloc de texte (mieux pour 'Nom Prénoms' sur plusieurs lignes)
        text = pytesseract.image_to_string(processed_crop, config='--psm 6')
        zone_text += f" {text} "
        
    return zone_text.strip()

def perform_global_ocr(image: Image.Image):
    """OCR sur toute l'image (contexte général + dates)"""
    return pytesseract.image_to_string(
        preprocess_image_for_ocr(image, is_crop=False), 
        config='--psm 6'
    )

def verify_name_match(text_recto, text_verso):
    """
    Compare les noms.
    text_recto: Doit idéalement venir de la zone 'Informations personnels' (ID 3)
    text_verso: Doit idéalement venir de la zone 'Bloc gauche' (ID 23)
    """
    # Extraction mots significatifs recto
    words_recto = re.findall(r'\b[A-Z]{3,}\b', text_recto.upper())
    candidates = [w for w in words_recto if w not in STOPWORDS]

    if not candidates:
        return False, "Aucun nom détecté au recto (Image floue ?)", None

    text_verso_clean = text_verso.upper()
    
    match_details = []
    
    for word in candidates:
        # Match exact
        if word in text_verso_clean:
            return True, f"Nom '{word}' validé sur les 2 faces.", word
        
        # Match approximatif
        words_verso = re.findall(r'\b[A-Z]{3,}\b', text_verso_clean)
        for v_word in words_verso:
            if v_word not in STOPWORDS:
                ratio = SequenceMatcher(None, word, v_word).ratio()
                if ratio > 0.80: # Seuil tolérant
                    return True, f"Correspondance : '{word}' ≈ '{v_word}'", word

    return False, "Nom du recto introuvable au verso.", None

# --- Fonctions Dates ---
def parse_date_flexible(date_string):
    if not date_string: return None
    s = date_string.upper().replace('O', '0').replace('I', '1').replace('L', '1').replace('B', '8')
    s = re.sub(r'[^\d/\.-]', '', s)
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
# 4. ENDPOINT PRINCIPAL
# ==============================================================================

@app.post("/analyze-full")
async def analyze_full_document(
    recto: UploadFile = File(...), 
    verso: UploadFile = File(None)
):
    if not model: raise HTTPException(status_code=500, detail="Modèle non chargé.")

    try:
        # Variables OCR
        full_text_r, full_text_v = "", "" # Texte global (pour les dates)
        name_text_r, name_text_v = "", "" # Texte ciblé (pour les noms)
        
        doc_type_r, conf_r = "INCONNU", 0.0
        doc_type_v, conf_v = "INCONNU", 0.0

        # --- 1. TRAITEMENT RECTO ---
        content_r = await recto.read()
        img_r = Image.open(io.BytesIO(content_r)).convert("RGB")
        res_r = model(img_r)[0]
        
        doc_type_r, conf_r = get_doc_info(res_r)
        
        # A. OCR Global (Pour trouver la date n'importe où)
        full_text_r = perform_global_ocr(img_r)
        # B. OCR Ciblé Dates
        full_text_r += " " + extract_zone_text(img_r, res_r, DATE_ZONE_IDS)
        
        # C. OCR Ciblé NOMS (Classe 3: Info Personnelles)
        name_text_r = extract_zone_text(img_r, res_r, NAME_ZONE_RECTO_IDS)
        
        # Si YOLO n'a pas vu la zone 3, on se rabat sur le texte global
        if len(name_text_r) < 5: 
            name_text_r = full_text_r 

        # --- 2. TRAITEMENT VERSO ---
        if verso:
            content_v = await verso.read()
            img_v = Image.open(io.BytesIO(content_v)).convert("RGB")
            res_v = model(img_v)[0]
            
            doc_type_v, conf_v = get_doc_info(res_v)
            
            full_text_v = perform_global_ocr(img_v)
            full_text_v += " " + extract_zone_text(img_v, res_v, DATE_ZONE_IDS)
            
            # C. OCR Ciblé NOMS VERSO (Classe 23: Bloc gauche)
            name_text_v = extract_zone_text(img_v, res_v, NAME_ZONE_VERSO_IDS)
            
            # Si YOLO n'a pas vu la zone 23, on se rabat sur le texte global
            if len(name_text_v) < 5:
                name_text_v = full_text_v

        # --- 3. COHÉRENCE ---
        final_type = doc_type_r
        is_coherent = False
        msg_coherence = "Analyse..."

        def get_family(name): return name.replace("_RECTO", "").replace("_VERSO", "")
        def get_side(name): return "RECTO" if "RECTO" in name else "VERSO" if "VERSO" in name else "AUTRE"

        if "PASSEPORT" in doc_type_r:
            is_coherent = True
            msg_coherence = "Passeport détecté."
        elif verso:
            fam_r, side_r = get_family(doc_type_r), get_side(doc_type_r)
            fam_v, side_v = get_family(doc_type_v), get_side(doc_type_v)

            if doc_type_r == "INCONNU" or doc_type_v == "INCONNU":
                 is_coherent = False
                 msg_coherence = "Document non reconnu."
            elif fam_r != fam_v:
                is_coherent = False
                msg_coherence = f"Incohérence : {fam_r} vs {fam_v}"
            elif side_r == side_v:
                is_coherent = False
                msg_coherence = f"Erreur : Deux fois le {side_r}."
            else:
                final_type = fam_r

                # --- EXCEPTION POUR LE PERMIS ---
                # Le permis n'a pas de lien textuel fort entre recto et verso.
                # On valide la cohérence dès que la structure (Recto+Verso) est bonne.
                if "PERMIS" in fam_r:
                    is_coherent = True
                    msg_coherence = "✅ Permis : Structure Recto/Verso validée."                
                else:
                    # Structure OK, on vérifie les NOMS avec les textes CIBLÉS
                    name_match, name_msg, _ = verify_name_match(name_text_r, name_text_v)
                    
                    if name_match:
                        is_coherent = True
                        msg_coherence = f"✅ Document Authentique. {name_msg}"
                        # final_type = fam_r
                    else:
                        # Ici on rejette si le nom n'est pas trouvé
                        is_coherent = False
                        msg_coherence = f"❌ Incohérence Nom : {name_msg}"

        else:
            is_coherent = False
            msg_coherence = "Verso manquant."

        # --- 4. VALIDITÉ ---
        combined_full_text = full_text_r + "\n" + full_text_v
        status, details, exp_date = determine_validity(final_type, combined_full_text)

        return {
            "is_valid_document": is_coherent and (status == "VALIDE"),
            "document_type": final_type.replace("_RECTO", ""),
            "confidence": round((conf_r + conf_v)/2 if verso else conf_r, 2),
            "coherence": {
                "status": is_coherent,
                "message": msg_coherence
            },
            "validity": {
                "status": status,
                "expiration_date": exp_date.strftime('%d/%m/%Y') if exp_date else None,
                "details": details
            },
            # "debug_name_r": name_text_r, # Décommenter pour voir ce que l'OCR lit dans la zone 3
            # "debug_name_v": name_text_v
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)