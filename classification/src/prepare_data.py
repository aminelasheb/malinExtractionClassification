import pandas as pd
from sklearn.model_selection import train_test_split

# Remplace les étiquettes par les chiffres dans le df
# to_categorical est un dictionnaire clé = nom de la classe, valeur = nombre
def label_to_categorical(df, to_categorical):
    df["label"].replace(to_categorical,inplace=True)
    if (df['label'].apply(lambda x: isinstance(x, str)).all()):
        print("WARNING labels sans id. Lignes supprimées !")
        df[df['label'].apply(lambda x: type(x) != str)]
    return df

# Charge les données du tsv et renvoie un df avec les colonnes, lignes et labels demandés ---> COLONNES MULTIPLES
def load_data(df, columns=[], 
              merge_dict={}, only_cats=[], drop_cats=[], books=[], limit=None, order=False,
              training=True):

    if training:
        # For training, set a max limit per class
        if limit is not None:
            # Sélection d'exercices (max limit) en maintenant le ratio manuels/labels
            df["manuel_label"] = df["manuel"].astype(str) + '_' + df['label'].astype(str)
            class_counts = df['manuel_label'].value_counts()
            classes_1 = class_counts[class_counts == 1].index
            df_1 = df[df['manuel_label'].isin(classes_1)]
            df = df[~df['manuel_label'].isin(classes_1)]
            manuel_label = df["manuel_label"]
            limit -= df_1.shape[0]
            df_dropped, df, manuel_label_dropped, manuel_label = train_test_split(df, manuel_label, test_size=limit, stratify=manuel_label, random_state=42)
            df = pd.concat([df, df_1])

        if "label" not in columns: columns.append("label")
        # For training on specific textbooks
        if books != []: # Que les manuels demandés
            df = df.loc[df["manuel"].isin(books)]
        # For training on specific labels
        for cat in drop_cats: # Omission des classes
            df = df.loc[df["label"] != cat]
        if only_cats != []:
            df = df.loc[df['label'].isin(only_cats)]
        # For training on coarse-grained labels
        df["label"].replace(merge_dict,inplace=True) # Fusion des classes

    if order is True:
        df = df.sort_values(by=['manuel', 'id'])

    # Final df with specified columns
    df = df[columns]

    return df