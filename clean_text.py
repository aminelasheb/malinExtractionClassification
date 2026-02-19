import os
import json
import sys
from pathlib import Path

# --- FIX WINDOWS ENCODING ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


def clean_string(text):
    """
    Remplace les espaces insécables et fins par des espaces normaux.
    """
    if not isinstance(text, str):
        return text

    replacements = {
        '\u2009': ' ',  # THSB (Thin Space)
        '\u202F': ' ',  # NNBSP (Narrow No-Break Space)
        '\u00A0': ' ',  # NBSP (Non-Breaking Space classique)
        ' ': ' ',  # Caractère littéral THSB
        ' ': ' '  # Caractère littéral NNBSP
    }

    cleaned = text
    for char, replacement in replacements.items():
        cleaned = cleaned.replace(char, replacement)

    return cleaned


def recursive_clean(data):
    """
    Parcourt récursivement le JSON (Dict ou List) pour nettoyer les strings.
    """
    if isinstance(data, dict):
        return {k: recursive_clean(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_clean(i) for i in data]
    elif isinstance(data, str):
        return clean_string(data)
    else:
        return data


def process_cleaning():
    base_dir = Path(__file__).resolve().parent
    extraction_dir = base_dir / "extractionOut"

    if not extraction_dir.exists():
        print(f"[ERR] Dossier introuvable : {extraction_dir}")
        return

    json_files = list(extraction_dir.glob("*.json"))
    print(f"[INFO] Nettoyage de {len(json_files)} fichiers JSON dans extractionOut...")

    count = 0
    for json_path in json_files:
        try:
            # 1. Lecture
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 2. Nettoyage
            cleaned_data = recursive_clean(data)

            # 3. Écriture (On écrase le fichier pour qu'il soit propre pour la suite)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

            count += 1
        except Exception as e:
            print(f"[ERR] Erreur sur {json_path.name} : {e}")

    print(f"[OK] {count} fichiers nettoyés (THSB/NNBSP retirés).")


if __name__ == "__main__":
    process_cleaning()