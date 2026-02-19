import os
import subprocess
import sys
import shutil
from pathlib import Path

# ================= CONFIGURATION DES CHEMINS FIXES =================
base_dir = Path(__file__).resolve().parent
# INPUT_DIR = base_dir / "extractionOutStyle"


# 1. Chemin vers votre projet de classification (PROJET PYCHARM)
CLASSIF_PROJECT_ROOT = base_dir / "classification"

# 2. Chemin vers vos résultats d'extraction (Entrée)
EXTRACTION_DIR = base_dir / "extractionOut"

# 3. NOUVEAU : Chemin vers le dossier de sortie de classification
CLASSIF_OUTPUT_DIR = base_dir / "classificationOut"

# Définition des fichiers requis dans le projet de classification
INFERENCE_SCRIPT = CLASSIF_PROJECT_ROOT / "src" / "inference.py"
MODEL_PATH = CLASSIF_PROJECT_ROOT / "modeles" / "ex_classif" / "saved_model_classification_ft_camembert.pt"
BASE_MODEL = CLASSIF_PROJECT_ROOT / "modeles" / "camembert-base"


# ===================================================================

def run_batch_classification():
    # Création du dossier de sortie s'il n'existe pas
    CLASSIF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"--- Configuration ---")
    print(f"Dossier Classif : {CLASSIF_PROJECT_ROOT}")
    print(f"Dossier Extraits : {EXTRACTION_DIR}")
    print(f"Dossier Sortie   : {CLASSIF_OUTPUT_DIR}\n")

    # Vérifications de sécurité
    if not CLASSIF_PROJECT_ROOT.exists():
        print(f"[ERR] Dossier malin-local introuvable.")
        return
    if not INFERENCE_SCRIPT.exists():
        print(f"[ERR] inference.py introuvable.")
        return
    if not EXTRACTION_DIR.exists():
        print(f"[ERR] Dossier extractionOut introuvable.")
        return

    # Liste des fichiers .tsv générés par Gemini
    tsv_files = list(EXTRACTION_DIR.glob("*.tsv"))
    # On filtre pour ne pas re-traiter des fichiers "pred_" si le dossier est mélangé
    tsv_files = [f for f in tsv_files if not f.name.startswith("pred_")]

    if not tsv_files:
        print(f"[WARN] Aucun fichier .tsv d'origine trouvé.")
        return

    print(f"--- Début de la classification ({len(tsv_files)} fichiers) ---\n")

    for tsv_file in tsv_files:
        print(f">> Traitement de : {tsv_file.name}")

        # Fichiers de sortie dans le NOUVEAU dossier
        output_txt = CLASSIF_OUTPUT_DIR / f"pred_{tsv_file.stem}.txt"
        output_tsv = CLASSIF_OUTPUT_DIR / f"pred_{tsv_file.stem}.tsv"

        # Construction de la commande
        cmd = [
            sys.executable, str(INFERENCE_SCRIPT),
            "--testfile", str(tsv_file),
            "-c1", "instruction_hint_example",
            "-c2", "statement",
            "--modele", str(MODEL_PATH),
            "--modelebase", str(BASE_MODEL),
            "--bertarchi", "single",
            "--ypredtxtfile", str(output_txt),
            "--ypredtsvfile", str(output_tsv)
        ]

        try:
            # Exécution
            subprocess.run(cmd, check=True, cwd=str(CLASSIF_PROJECT_ROOT))
            print(f"[OK] Succès ! Résultats dans : classificationOut/\n")
        except subprocess.CalledProcessError as e:
            print(f"[ERR] Échec du traitement pour {tsv_file.name}\n")


if __name__ == "__main__":
    run_batch_classification()