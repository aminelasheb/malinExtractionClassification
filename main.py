import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

# Définition du chemin de base
BASE_DIR = Path(__file__).resolve().parent

# Dossiers à nettoyer
DIRS_TO_RESET = [
    "files", "files_style", "files_images", "output",
    "files-out", "extractionOut", "extractionOutStyle",
     "classificationOut",
]


def reset_directories():
    """Supprime et recrée les dossiers de sortie."""
    for directory in DIRS_TO_RESET:
        dir_path = BASE_DIR / directory
        shutil.rmtree(dir_path, ignore_errors=True)
        os.makedirs(dir_path, exist_ok=True)
        print(f"[OK] Reset: {dir_path}")


def run_script(script_name, *args):
    """Exécute un script python externe avec des arguments et l'encodage forcé."""
    script_path = BASE_DIR / script_name

    if not script_path.exists():
        print(f"[ERR] Script not found: {script_path}")
        sys.exit(1)

    print(f"\n[RUN] Running {script_name}...")

    # On convertit tous les arguments en string pour subprocess
    cmd_args = [sys.executable, str(script_path), *map(str, args)]

    # --- MODIFICATION ICI : Création d'un environnement modifié ---
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"
    # --------------------------------------------------------------

    process = subprocess.Popen(
        cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=my_env
    )

    for line in iter(process.stdout.readline, ''):
        if line:
            print(line, end="")
        else:
            break

    process.stdout.close()
    process.wait()

    if process.returncode == 0:
        print(f"[OK] {script_name} finished successfully")
    else:
        print(f"[ERR] {script_name} failed. Stopping pipeline.")
        sys.exit(1)


def str2bool(v):
    """Fonction utilitaire pour convertir un string en booléen."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # --- CONFIGURATION DES ARGUMENTS CLI ---
    parser = argparse.ArgumentParser(description="Pipeline d'extraction PDF")

    # 1. Nom du fichier PDF (Obligatoire)
    parser.add_argument("pdf_name", type=str, help="Nom du fichier PDF (doit être dans le dossier PdfSource)")

    # 2. Gestion des pages (--all OU --first & --last)
    parser.add_argument("--all", action="store_true", help="Traiter toutes les pages du PDF")
    parser.add_argument("--first", type=int, help="Numéro de la première page")
    parser.add_argument("--last", type=int, help="Numéro de la dernière page")

    # 3. Gestion du style (Optionnel, défaut False)
    parser.add_argument("--style", type=str, default="false", help="Mode style (true/false)")

    args = parser.parse_args()

    # --- INITIALISATION DES VARIABLES ---
    PDF_PATH = BASE_DIR / "PdfSource" / args.pdf_name

    if not PDF_PATH.exists():
        print(f"[ERREUR] Le fichier PDF est introuvable ici : {PDF_PATH}")
        sys.exit(1)

    STYLE_MODE = str2bool(args.style)
    ALL_PAGES = args.all
    FIRST_PAGE = ""
    LAST_PAGE = ""

    if ALL_PAGES:
        all_flag = "true"
        print(f"[INFO] Mode : TOUTES les pages")
    else:
        if args.first is None or args.last is None:
            print("[ERREUR] Si vous n'utilisez pas --all, vous DEVEZ spécifier --first et --last.")
            sys.exit(1)

        all_flag = "false"
        FIRST_PAGE = args.first
        LAST_PAGE = args.last
        print(f"[INFO] Mode : Pages {FIRST_PAGE} à {LAST_PAGE}")

    print(f"[INFO] Fichier : {args.pdf_name}")
    print(f"[INFO] Style Mode : {STYLE_MODE}")

    # --- EXÉCUTION DU PIPELINE ---

    reset_directories()
    run_script("pdfToImages.py", PDF_PATH, BASE_DIR / "files", all_flag, FIRST_PAGE, LAST_PAGE)
    run_script("pdfToTxtStyle.py", PDF_PATH, BASE_DIR / "files_style", all_flag, FIRST_PAGE, LAST_PAGE)
    run_script("detectImages.py")
    run_script("cropImages.py")
    run_script("drawBoxes.py")
    run_script("extraction-gemini-vision.py", "--style", "false")
    run_script("classification.py")
    run_script("style-post.py")
    run_script("organize_outputs.py", args.pdf_name)

    print("\n[DONE] All tasks completed successfully!")