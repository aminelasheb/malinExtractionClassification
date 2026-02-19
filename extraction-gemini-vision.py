import os
import time
import json
import csv
import re
import sys
import argparse  # <--- Ajouté
from typing import Optional
from pathlib import Path

import PIL.Image
from google import genai
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- FIX WINDOWS ENCODING ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# =========================
# GESTION DES ARGUMENTS
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--style", type=str, default="false", help="Mode style (true/false)")
args = parser.parse_args()

# Conversion string "true"/"false" en booléen
STYLE_MODE = args.style.lower() in ('true', '1', 'yes')
print(f"[INFO] Extraction Gemini - Style Mode: {STYLE_MODE}")


MODEL_NAME = "gemini-2.5-flash"

base_dir = Path(__file__).parent.resolve()
api_key_path = base_dir / "apikey.txt"

# Dossiers d'entrée/sortie
image_dir = os.path.join(base_dir, "files-out")
text_dir = os.path.join(base_dir, "files_style")

if STYLE_MODE:
    prompt_file = os.path.join(base_dir, "promptStyle.txt")
    output_dir = os.path.join(base_dir, "extractionOutStyle")
else:
    prompt_file = os.path.join(base_dir, "prompt.txt")
    output_dir = os.path.join(base_dir, "extractionOut")

os.makedirs(output_dir, exist_ok=True)


# =========================
# UTILITAIRES
# =========================
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_fenced_json(text: str) -> str:
    """Nettoie les balises Markdown ```json autour de la réponse."""
    t = text.strip()
    if t.startswith("```json"):
        t = t[len("```json"):].strip()
    if t.startswith("```"):
        t = t[len("```"):].strip()
    if t.endswith("```"):
        t = t[:-3].strip()
    return t


def clean_text_for_tsv(text) -> str:
    """Nettoie le texte pour le format TSV (pas de tabulations ni sauts de ligne)."""
    if text is None:
        return ""
    # On remplace les tabulations et nouvelles lignes par des espaces
    return str(text).replace('\t', ' ').replace('\n', ' ').strip()


def save_json_safely(raw_text: str, out_path: str) -> None:
    """Sauvegarde le JSON proprement, ou le texte brut si le parsing échoue."""
    cleaned = clean_fenced_json(raw_text)
    try:
        # strict=False permet d'accepter les sauts de ligne dans les strings (fréquent avec les LLM)
        parsed = json.loads(cleaned, strict=False)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
    except Exception:
        # Si ça échoue quand même, on sauvegarde le texte brut pour debug/réparation
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)


def load_api_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_json_robust(json_path: str):
    """Tente de charger un JSON même s'il contient des caractères invalides."""
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        # 1. Tentative standard mais permissive
        return json.loads(content, strict=False)
    except json.JSONDecodeError:
        try:
            # 2. Tentative de nettoyage des caractères de contrôle (sauf \n formatage)
            # On échappe les retours à la ligne qui semblent être dans des valeurs
            cleaned = re.sub(r'(?<!\\)\n', '\\n', content)
            return json.loads(cleaned, strict=False)
        except:
            # 3. Echec
            raise ValueError("Impossible de parser le JSON même après nettoyage.")


# =========================
# CONVERTISSEUR JSON -> TSV
# =========================
def convert_json_to_tsv(json_path: str, tsv_path: str):
    """Convertit un fichier JSON extrait en fichier TSV pour la classification."""
    try:
        # Utilisation du chargeur robuste
        data = load_json_robust(json_path)

        # Normalisation : on veut une liste d'exercices
        if isinstance(data, dict):
            if "items" in data:
                data = data["items"]
            elif "$defs" in data:
                print(f"[WARN] Structure JSON complexe (schema) ignorée pour TSV : {json_path}")
                return
            else:
                data = [data]
        elif not isinstance(data, list):
            print(f"[WARN] Format JSON inattendu pour {json_path}")
            return

        with open(tsv_path, "w", encoding="utf-8", newline="") as tsv_file:
            writer = csv.writer(tsv_file, delimiter="\t")

            # En-têtes exacts requis
            writer.writerow([
                "textbook", "id", "full_ex", "num", "indicator", "instruction",
                "hint", "example", "statement", "instruction_hint_example",
                "label", "grandtype", "stratify_key"
            ])

            for ex in data:
                # Récupération des propriétés (parfois imbriquées, parfois à plat)
                props = ex.get("properties", {}) if "properties" in ex else ex

                # Champs simples
                id_val = ex.get("id", "none")
                numero = clean_text_for_tsv(props.get("number") or props.get("numero"))
                if not numero: numero = "none"

                # Textes
                instruction = clean_text_for_tsv(props.get("instruction") or props.get("consignes"))
                hint = clean_text_for_tsv(props.get("hint") or props.get("conseil"))
                example = clean_text_for_tsv(props.get("example") or props.get("exemple"))

                # Enoncé + Labels
                raw_statement = clean_text_for_tsv(props.get("statement") or props.get("enonce"))
                labels_list = props.get("labels", [])

                if labels_list and isinstance(labels_list, list):
                    labels_text = " ".join([clean_text_for_tsv(l) for l in labels_list])
                    statement = f"{raw_statement} {labels_text}".strip()
                else:
                    statement = raw_statement

                # Colonnes composées (concaténation pour le modèle BERT)
                parts_ihe = [p for p in [instruction, hint, example] if p]
                instruction_hint_example = " ".join(parts_ihe)

                parts_full = [p for p in [instruction, hint, example, statement] if p]
                full_ex = " ".join(parts_full)

                # Écriture de la ligne
                writer.writerow([
                    "manual_CE1",  # textbook
                    id_val,  # id
                    full_ex,  # full_ex
                    numero,  # num
                    "none",  # indicator
                    instruction,  # instruction
                    hint,  # hint
                    example,  # example
                    statement,  # statement (inclut les labels)
                    instruction_hint_example,  # Colonne clé pour la classification
                    "none",  # label (cible)
                    "none",  # grandtype
                    "none"  # stratify_key
                ])

        print(f" [TSV] Généré : {os.path.basename(tsv_path)}")

    except Exception as e:
        print(f" [ERR] Echec conversion TSV pour {json_path}: {e}")


# =========================
# APPEL GEMINI
# =========================
def generate_content_safe(client: genai.Client, contents) -> Optional[str]:
    """Appelle Gemini avec gestion simple des erreurs de quota."""
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=contents)
            return getattr(resp, "text", None)
        except Exception as e:
            msg = str(e).lower()
            # Si erreur 429 ou quota, on attend un peu
            retryable = any(k in msg for k in (
                "rate", "limit", "quota", "exceeded", "resource exhausted",
                "temporarily", "please try again"
            ))
            if not retryable:
                print(f" [ERR] Erreur fatale Gemini : {e}")
                return None

            backoff = min(60, max(2, 2 ** attempt))  # Max 60s d'attente
            print(f" [INFO] Quota atteint. Pause de {backoff}s...")
            time.sleep(backoff)

    print(" [ERR] Abandon après plusieurs tentatives.")
    return None


# =========================
# PIPELINE PRINCIPAL
# =========================
def process_image_file(client: genai.Client, image_path: str) -> None:
    name = os.path.basename(image_path)
    stem, _ = os.path.splitext(name)

    # Chemins des fichiers
    csv_path = os.path.join(text_dir, f"{stem}.csv")
    txt_path = os.path.join(text_dir, f"{stem}.txt")
    out_json = os.path.join(output_dir, f"{stem}.json")
    out_tsv = os.path.join(output_dir, f"{stem}.tsv")

    # 1. VÉRIFICATION INTELLIGENTE JSON/TSV
    if os.path.exists(out_json):
        # Le JSON existe déjà
        if not os.path.exists(out_tsv):
            # MAIS le TSV manque -> on le génère et on s'arrête là
            print(f" [INFO] JSON trouvé pour {stem}, génération du TSV manquant...")
            convert_json_to_tsv(out_json, out_tsv)
        else:
            print(f" [SKIP] {stem} déjà fait (JSON & TSV présents).")
        return

    # Si on arrive ici, c'est que le JSON n'existe pas. On lance l'extraction.
    print(f" [RUN] Traitement de {stem}...")

    # 2. CONVERSION CSV -> TXT (Entrée texte pour Gemini)
    if not os.path.exists(csv_path):
        print(f" [WARN] Fichier CSV introuvable pour {stem}")
        return

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_content = f.read()
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(csv_content)
    except Exception as e:
        print(f" [ERR] Impossible de lire/écrire le CSV/TXT : {e}")
        return

    # 3. CHARGEMENT IMAGE ET PROMPT
    try:
        image = PIL.Image.open(image_path)
    except Exception as e:
        print(f" [ERR] Impossible d'ouvrir l'image {image_path}: {e}")
        return

    base_prompt = read_file(prompt_file)
    try:
        side_text = read_file(txt_path)
    except Exception:
        side_text = ""

    full_prompt = (
            base_prompt
            + '\n\n--- { CSV input :  "\n'
            + side_text
            + '\n"}'
    )

    # 4. APPEL GEMINI
    contents = [full_prompt, image]
    resp_text = generate_content_safe(client, contents)

    if not resp_text:
        print(f" [ERR] Pas de réponse de Gemini pour {name}")
        return

    # 5. SAUVEGARDE JSON
    save_json_safely(resp_text, out_json)
    print(f" [OK] JSON sauvegardé : {out_json}")

    # 6. GÉNÉRATION IMMÉDIATE DU TSV
    convert_json_to_tsv(out_json, out_tsv)


def main():
    api_key = load_api_key(api_key_path)
    client = genai.Client(api_key=api_key)

    print(f"Dossier images : {image_dir}")
    print(f"Dossier sortie : {output_dir}")

    # Parcours des images
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            fpath = os.path.join(image_dir, fname)
            process_image_file(client, fpath)

    print("\n[DONE] Extraction terminée.")


if __name__ == "__main__":
    main()