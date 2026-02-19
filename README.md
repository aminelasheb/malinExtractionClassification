Voici la version corrigée, sans nom personnel et rédigée de manière totalement générale.

Tu peux remplacer entièrement ton README par celui-ci 👇

---

# 📚 MALIN – Extraction & Classification d’Exercices PDF

Pipeline complet d’extraction et de classification automatique d’exercices scolaires à partir de PDF natifs.

Le projet permet :

* 🔎 Détection des zones d’exercices
* 📝 Extraction du texte (avec ou sans style)
* 🧠 Structuration automatique en JSON
* 🏷 Classification par typologie d’exercice
* 📂 Organisation automatique des sorties

---

# 1️⃣ Guide d’installation

## 🔧 Dépendances système

### Installer Ghostscript

Ghostscript est nécessaire pour la conversion **PDF → images**.

Télécharger et installer :
[https://ghostscript.com/](https://ghostscript.com/)

Vérifier l’installation :

```bash
gswin64c -v
```

---

## 🐍 Environnement Python (3.9 recommandé)

Installer Python 3.9.

Créer un environnement virtuel :

```bash
py -3.9 -m venv venv39
venv39\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 🔐 Clé API Gemini

Créer un fichier :

```
apikey.txt
```

à la racine du projet contenant uniquement :

```
VOTRE_CLE_API
```

---

## 🧠 Modèle YOLO (détection d’exercices)

Télécharger le poids `.pt` et le placer dans :

```
models/detImages/
```

---

## 🏷 Modèle de classification (CamemBERT fine-tuned)

Télécharger les poids du modèle et placer le contenu dans :

```
classification/
```

---

## 📂 Dossier source PDF

Créer un dossier à la racine :

```
PdfSource/
```

Y placer les PDF à traiter.

---

# 2️⃣ Guide d’utilisation

## ▶ Lancement du pipeline

Syntaxe générale :

```bash
python main.py <nom_du_pdf.pdf> [--all] [--first N] [--last N]
```

Exemple :

```bash
python main.py document.pdf --first 7 --last 10
```

---

## 🔎 Exemples

### Tester sur quelques pages

```bash
python main.py document.pdf --first 9 --last 10
```

### Lancer sur tout le PDF

```bash
python main.py document.pdf --all
```

---

# 📁 Sorties & Arborescence

À la fin de l'exécution, un dossier est généré automatiquement :

```
SORTIES/<nom_du_pdf>/
```

Exemple :

```
SORTIES/document/
│
├── Extraction_exercices/
├── Extraction_exercices --style/
├── CategorisationExercices/
└── CategorisationExercices --style/
```

---

## 📄 Extraction

Contient un fichier JSON par page :

```
Extraction_exercices/
    page_7.json
```

---

## 🎨 Extraction avec style

Le dossier :

```
Extraction_exercices --style/
```

Préserve la mise en forme LaTeX :

* Gras : `\bf{}`
* Italique : `\it{}`
* Couleur : `\color{"txt",#HEX}`
* Images : `\image{id}`

---

## 🏷 Classification

Les exercices sont ensuite séparés et triés automatiquement :

```
CategorisationExercices --style/
    CM/
        P9Ex11.json
        P9Ex5.json
        P9Ex6.json
```

Chaque dossier correspond à une typologie d’exercice.

---

# 📦 Format JSON

Chaque page génère un tableau d’objets `Exercise`.

```json
[
  {
    "id": "string | null",
    "type": "exercise",
    "images": true,
    "image_type": "none | single | ordered | unordered | composite",
    "properties": {
      "number": "string | null",
      "instruction": "string | null",
      "labels": ["string"],
      "statement": "string | null",
      "hint": "string | null",
      "example": "string | null",
      "references": "string | null"
    }
  }
]
```
