import os
import requests
import glob
import json

# ==============================================================================
# CONFIGURATION DU TEST
# ==============================================================================
API_URL = "http://127.0.0.1:8000/analyze"
TEST_IMAGES_DIR = "./test_dataset"  # Le dossier contenant tes images à tester

# Extensions d'images supportées
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp']

def run_test_scenario():
    # 1. Récupérer toutes les images du dossier
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(TEST_IMAGES_DIR, ext)))
    
    if not image_files:
        print(f"⚠️ Aucune image trouvée dans le dossier '{TEST_IMAGES_DIR}'")
        return

    print(f"🚀 Démarrage du test sur {len(image_files)} images...\n")
    print(f"{'FICHIER':<30} | {'TYPE DÉTECTÉ':<20} | {'STATUT VALIDITÉ':<15} | {'DÉTAILS'}")
    print("-" * 100)

    # 2. Boucle d'envoi
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        try:
            # Ouverture du fichier en mode binaire
            with open(img_path, 'rb') as f:
                files = {'file': (filename, f, 'image/jpeg')}
                
                # Envoi POST à l'API
                response = requests.post(API_URL, files=files)
                
            if response.status_code == 200:
                data = response.json()
                
                doc_type = data.get('document_type', 'N/A')
                analysis = data.get('analysis', {})
                status = analysis.get('validity_status', 'N/A')
                details = analysis.get('details', '')
                
                # Affichage formaté
                print(f"{filename[:28]:<30} | {doc_type[:18]:<20} | {status:<15} | {details}")
            else:
                print(f"{filename:<30} | ERREUR {response.status_code}: {response.text}")

        except Exception as e:
            print(f"{filename:<30} | EXCEPTION: {e}")

    print("\n✅ Fin du scénario de test.")

if __name__ == "__main__":
    # Vérification que le serveur semble tourner (simple check de port ou juste lancer)
    try:
        requests.get("http://127.0.0.1:8000")
        run_test_scenario()
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de contacter le serveur.")
        print("👉 Assurez-vous d'avoir lancé 'python main.py' dans un autre terminal d'abord !")