from pathlib import Path

# =========================
# ⚙️ CONFIGURATION GÉNÉRALE
# =========================

# 1. MODE D'EXTRACTION
# False = Texte brut (rapide, pour downstream tasks)
# True  = Texte avec style (gras, couleurs, italique via LaTeX)
STYLE_MODE = False

# 2. FICHIER PDF D'ENTRÉE
# Utilisez r"..." pour éviter les erreurs avec les backslashs Windows
PDF_PATH = Path(
    r"C:\Users\lasheb\Desktop\PFE\PFE\les manuels scolaires\manual_CE1_FRANCAIS_MAGNARD.pdf"
)

# 3. CONTRÔLE DES PAGES
ALL_PAGES = False        # True = tout le PDF, False = sous-ensemble ci-dessous
FIRST_PAGE = 9           # Numéro de la première page (1-based)
LAST_PAGE = 10           # Numéro de la dernière page (inclus)

