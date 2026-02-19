import argparse
import numpy as np

np.set_printoptions(threshold=np.inf)
import pandas as pd

pd.set_option('display.max_columns', None)
from pprint import pprint
import os
import sys  # <--- Ajouté

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer
from datasets import Dataset

import json

from prepare_data import load_data
from models_bert_torch import compute_input_arrays, MAX_SEQUENCE_LENGTH, SingleBert, DualBert, SiameseBert

if __name__ == "__main__":

    # ✔ Pour l'inférence sur 1 fichier d'exercices
    # ✘ Pour l'évaluation sur les données annotées : inference_with_eval.py

    # Inputs : exercices extraits dans un format TSV avec les colonnes suivantes :
    # 'textbook', 'id', 'full_ex', 'num', 'indicator', 'instruction', 'hint', 'example', 'statement', 'instruction_hint_example', 'label', 'grandtype', 'stratify_key'
    # Pour l'inférence, sont nécessaires les colonnes suivantes :
    # 'textbook', 'id', 'full_ex', 'instruction_hint_example', 'statement',

    # Run ce script avec :
    # python3 ./src/inference_avec_eval.py\
    # --test <fichier tsv>\
    # -c1 <colonne correspondant à la partie 1 de l'input>\
    # -c2 <colonne correspondant à la partie 2 de l'input>\
    # --modele <modele fine-tuné sur la tâche de classification>\
    # --modelebase <modele de base (avant fine-tuning)>\
    # --bertarchi <single|dual|siamese>\
    # --ypredtxtfile <chemin de sauvegarde des prédictions seules en txt> (optionnel) \
    # --ypredtsvfile <chemin de sauvegarde du df complet avec les prédictions en tsv> (optionnel)

    # Exemple :
    # python3 ./src/inference.py
    # --test ../../extraction/images/1.tsv \
    # -c1 instruction_hint_example \
    # -c2 statement \
    # --modele modeles/ex_classif/saved_model_classification_ft_camembert.pt \
    # --modelebase modeles/camembert-base \
    # --bertarchi single \
    # --ypredtxtfile pred_1.txt \
    # --ypredtsvfile pred_1.tsv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("GPU available :", device)
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("-te", "--testfile")
    parser.add_argument("-c1", "--column1")
    parser.add_argument("-c2", "--column2")
    parser.add_argument("-m", "--modele")
    parser.add_argument("-mb", "--modelebase")
    parser.add_argument("-a", "--bertarchi")
    parser.add_argument("-txt", "--ypredtxtfile", default=None)
    parser.add_argument("-tsv", "--ypredtsvfile", default=None)

    args = parser.parse_args()
    testfile = args.testfile
    column1 = args.column1  # instruction_hint_example
    column2 = args.column2  # statement
    modele = args.modele
    modelebase = args.modelebase
    bertarchi = args.bertarchi
    pred_file_txt = args.ypredtxtfile
    pred_file_tsv = args.ypredtsvfile

    # LABELS RELANCE MARS 2025
    # TODO mettre à jour dynamiquement selon le modèle
    labelDict = {
        'Associe': 0,
        'AssocieCoche': 1,
        'CM': 2,
        'CacheIntrus': 3,
        'Classe': 4,
        'ClasseCM': 5,
        'CliqueEcrire': 6,
        'CocheGroupeMots': 7,
        'CocheIntrus': 8,
        'CocheLettre': 9,
        'CocheMot': 10,
        'CocheMot*': 11,
        'CochePhrase': 12,
        'Echange': 13,
        'EditPhrase': 14,
        'EditTexte': 15,
        'ExpressionEcrite': 16,
        'GenreNombre': 17,
        'Phrases': 18,
        'Question': 19,
        'RC': 20,
        'RCCadre': 21,
        'RCDouble': 22,
        'RCImage': 23,
        'Texte': 24,
        'Trait': 25,
        'TransformeMot': 26,
        'TransformePhrase': 27,
        'VraiFaux': 28
    }
    inverseLabelDict = {v: k for k, v in labelDict.items()}
    labels = labelDict.keys()
    print("LABELS :")
    pprint(labelDict)
    print()

    # 1 ou 2 inputs ?
    if bertarchi == "single":
        double = False
    else:
        double = True

    # Tokenizer
    print("TOKENIZER:")
    tokenizer = AutoTokenizer.from_pretrained(modelebase, do_lower_case=True, use_fast=False)
    print(tokenizer)
    print()

    # Chargement des données en df avec les colonnes consigne + énoncé
    # TODO : charger le df directement sorti de la tâche d'extraction
    ex_ids = pd.read_csv(testfile, sep='\t')['id'].tolist()
    df_full = pd.read_csv(testfile, header=[0], sep="\t")  # Lecture du tsv en dataframe
    df_test = load_data(df_full, ["textbook", "id", column1, column2], only_cats=[], merge_dict={})
    x_test = df_test[[column1, column2]].fillna("")

    print("INPUT DATA:")
    try:
        print(x_test)
    except UnicodeEncodeError:
        # Si la console plante, on affiche une version "nettoyée" (ASCII)
        print(x_test.to_string().encode('ascii', 'replace').decode('ascii'))
    print()

    print("*ENCODE DATA*")
    input_test = compute_input_arrays(x_test, [column1, column2], tokenizer, MAX_SEQUENCE_LENGTH, double=double,
                                      labels=False)
    print()

    eval_dataset = Dataset.from_dict(input_test)
    eval_dataset.set_format(type="torch", device=device)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=1)

    ############# Modèle ################
    print("*LOAD MODEL*")
    print()

    # if bertarchi == "single":
    #     model = SingleBert(modele,labels)
    # elif bertarchi == "dual":
    #     model = DualBert(modele,labels)
    # elif bertarchi == "siamese":
    #     model = SiameseBert(modele,labels)
    model = torch.load(modele, map_location=torch.device('cpu'), weights_only=False)
    model.to(device)

    # criterion = CrossEntropyLoss()

    ############ Prediction #############
    print("*PREDICT*")
    print()

    model.eval()
    preds = []
    pred_label_ids = []

    # for batch in tqdm(eval_dataloader, desc="Evaluating"):
    for batch in eval_dataloader:
        with torch.no_grad():
            if double is True:
                ids = [batch['input_ids_1'], batch['input_ids_2']]
                mask = [batch['attention_mask_1'], batch['attention_mask_2']]
                token_type_ids = [batch['token_type_ids_1'], batch['token_type_ids_2']]
                labels = batch["labels"]
            else:
                ids, mask, token_type_ids = batch['input_ids'], batch['attention_mask'], batch['token_type_ids']

            outputs = model(ids, attention_mask=mask, token_type_ids=token_type_ids)  # scores

            # compute the predictions
            for pred in outputs.detach().cpu().numpy():
                preds.append(pred)
                pred_label_ids.append(pred.argmax(-1))

    # convert ids to labels
    y_pred = [inverseLabelDict[id] for id in pred_label_ids]

    # Merge in dataframe
    df_test.loc[:, "pred"] = y_pred
    print("PREDICTIONS:")
    try:
        print(df_test.head(10))
    except UnicodeEncodeError:
        # On remplace les caractères spéciaux par des '?' juste pour l'affichage console
        print(df_test.head(10).to_string().encode('cp1252', 'replace').decode('cp1252'))
    print()

    # Sauvegarde des prédictions en txt
    if pred_file_txt is not None:
        with open(pred_file_txt, "w") as f:
            for pred in y_pred:
                f.write(f"{pred}\n")
            # json.dump(y_pred, f, indent=4)
        print("Saved in", pred_file_txt)

    # Sauvegarde des prédictions en tsv
    if pred_file_tsv is not None:
        df_test.to_csv(pred_file_tsv, sep="\t", index=False)
