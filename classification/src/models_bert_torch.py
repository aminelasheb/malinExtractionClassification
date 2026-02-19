import torch
from torch import cat
from torch.nn import Linear, Dropout

from transformers import AutoConfig, CamembertModel, AutoModel

# Variables globales
global MAX_SEQUENCE_LENGTH
MAX_SEQUENCE_LENGTH = 256 # pour truncation_strategy dans convert_to_transformer_inputs
#MAX_SEQUENCE_LENGTH = 512


### ENCODAGE DES INPUTS POUR LE TRANSFORMER

def convert_to_transformer_inputs(str1, str2, tokenizer, max_sequence_length, double=True):

    def return_id(str1, str2, length):

        inputs = tokenizer(
            str1, # text, first sequence to be encoded
            str2, #text_pair, second sequence to be encoded (optional)
            max_length=max_sequence_length, # maximum length to use by truncation or padding parameters
            truncation=True, # activates and controls truncation : True=longest_first, only_first, only_second, False=do_not_truncate
            return_token_type_ids=True,
            add_special_tokens=True # whether or not to encode the sequences with the special tokens relative to their model
            )

        # list of token indices, numerical representations of tokens building the sequences that will be used as input by the model
        input_ids = inputs["input_ids"]
        # Attention mask : sert à ignorer le padding. On donnera au modèle l'attention mask en même temps que les input ids
        # 1hot tensor with the same shape as the input ids. The mask has 1 for real tokens and 0 for padding tokens
        input_masks = inputs["attention_mask"] # <=> [1] * len(input_ids)
        # Si plusieurs séquences ne forment qu'une entrée, les token_type_ids indiquent à quelle séquence chaque token correspond (ici str1 : 0, str2 si not None : 1 --> le cas avec BERT mais pas avec les modèles de type RoBERTa qui n'en tient pas compte et sépare les segments par le sep_token)
        input_segments = inputs["token_type_ids"]

        # Ajout du padding
        padding_length = length - len(input_ids)

        padding_id = tokenizer.pad_token_id # récupère l'id du token padding, ici 1

        input_ids = input_ids + ([padding_id] * padding_length) # padding avec 1
        input_masks = input_masks + ([0] * padding_length) # padding avec 0
        input_segments = input_segments + ([0] * padding_length) # padding avec 0

        return [input_ids, input_masks, input_segments]

    if double==True:
        # 2 inputs séparés : listes input ids, masks, segments pour chaque input
        input_ids_1, input_masks_1, input_segments_1 = return_id(str1, None, max_sequence_length)
        input_ids_2, input_masks_2, input_segments_2 = return_id(str2, None, max_sequence_length)

        return [input_ids_1, input_masks_1, input_segments_1,
                input_ids_2, input_masks_2, input_segments_2]

    else:
        # 2 inputs en 1 : une seule liste input ids, masks, segments. Les indices des segments (bert) ou le sep_token (roberta, camembert) permettent de spécifier dequel input (str1/str2) vient le token
        input_ids, input_masks, input_segments = return_id(str1, str2, max_sequence_length)

        return [input_ids, input_masks, input_segments,
                None, None, None]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length, double=True, labels=True):
    if labels is True:
        columns.append("label")
        y_1 = []
        y_2 = []
    input_ids_1, input_masks_1, input_segments_1 = [], [], []
    input_ids_2, input_masks_2, input_segments_2 = [], [], []
    # Parcours de chaque ligne du df = chaque exercice
    for _, instance in df[columns].iterrows():
        # Récupération des 2 inputs (consigne / énoncé)
        str1, str2 = instance[columns[0]], instance[columns[1]]

        # Vectorisation + conversion en transformer inputs
        ids_1, masks_1, segments_1, ids_2, masks_2, segments_2 = convert_to_transformer_inputs(str1, str2, tokenizer, max_sequence_length, double=double)

        # Récupération de l'étiquette
        if labels is True:
            y = instance["label"]

        # Ajout aux listes des listes d'input ids, masks, segments
        input_ids_1.append(ids_1)
        input_masks_1.append(masks_1)
        input_segments_1.append(segments_1)
        if labels is True:
            y_1.append(y)

        input_ids_2.append(ids_2)
        input_masks_2.append(masks_2)
        input_segments_2.append(segments_2)
        if labels is True:
            y_2.append(y)

    # Conversion en numpy array 
    if double:

        if labels is True:
            return {
                'input_ids_1':torch.tensor(input_ids_1),
                'token_type_ids_1':torch.tensor(input_segments_1),
                'attention_mask_1':torch.tensor(input_masks_1),
                'input_ids_2':torch.tensor(input_ids_2),
                'token_type_ids_2':torch.tensor(input_segments_2),
                'attention_mask_2':torch.tensor(input_masks_2),
                'labels':torch.tensor(y_1)
            }
        return {
            'input_ids_1':torch.tensor(input_ids_1),
            'token_type_ids_1':torch.tensor(input_segments_1),
            'attention_mask_1':torch.tensor(input_masks_1),
            'input_ids_2':torch.tensor(input_ids_2),
            'token_type_ids_2':torch.tensor(input_segments_2),
            'attention_mask_2':torch.tensor(input_masks_2),
        }
    else:
        if labels is True:
            return {
                'input_ids':torch.tensor(input_ids_1),
                'token_type_ids':torch.tensor(input_segments_1),
                'attention_mask':torch.tensor(input_masks_1),
                'labels':torch.tensor(y_1)
            }
        return {
            'input_ids':torch.tensor(input_ids_1),
            'token_type_ids':torch.tensor(input_segments_1),
            'attention_mask':torch.tensor(input_masks_1),
        }

### CREATE MODELS

# Utilisation directe de CamembertForSequenceClassification, ou bien custom classes :

class SingleBert(torch.nn.Module):
    def __init__(self,modele,labels):
        super().__init__()
        config = AutoConfig.from_pretrained(modele)
        if "camembert" in modele:
            self.model = CamembertModel.from_pretrained(modele)
        else:
            self.model = AutoModel.from_pretrained(modele)
        self.drop = Dropout(0.2)
        self.classifier = Linear(config.hidden_size, len(labels)) # final layer
    def forward(self, ids, attention_mask, token_type_ids):
        embedding= self.model(ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0] # torch.Size([16, 250, 768]) (16=batch)
        pooled_output = embedding.mean(axis=1)
        pooled_output = self.drop(pooled_output)
        scores = self.classifier(pooled_output)
        return scores

class SiameseBert(torch.nn.Module):
    def __init__(self,modele,labels):
        super().__init__()
        config = AutoConfig.from_pretrained(modele)
        self.model = CamembertModel.from_pretrained(modele)
        self.drop = Dropout(0.2)
        self.dense = Linear(2*config.hidden_size,config.hidden_size)
        self.classifier = Linear(config.hidden_size, len(labels)) # final layer
    def forward_once(self, ids, mask, token_type_ids):
        _, output= self.model(ids, attention_mask = mask, token_type_ids = token_type_ids)
        return output
    def forward(self, ids, attention_mask, token_type_ids):
        embedding_1 = self.model(ids[0], attention_mask = attention_mask[0], token_type_ids = token_type_ids[0])[0]
        embedding_2 = self.model(ids[1], attention_mask = attention_mask[1], token_type_ids = token_type_ids[1])[0]
        pooled_output_1 = embedding_1.mean(axis=1) #torch.Size([16, 768])
        pooled_output_2 = embedding_2.mean(axis=1) #torch.Size([16, 768])
        pooled_output = cat([pooled_output_1,pooled_output_2],1) #torch.Size([16, 1536])
        pooled_output = self.drop(pooled_output)
        pooled_output = self.dense(pooled_output) #torch.Size([16, 768])
        scores = self.classifier(pooled_output)
        return scores

class DualBert(torch.nn.Module):
    def __init__(self,modele,labels):
        super().__init__()
        config = AutoConfig.from_pretrained(modele)
        self.model_1 = CamembertModel.from_pretrained(modele)
        self.model_2 = CamembertModel.from_pretrained(modele)
        self.drop = Dropout(0.2)
        self.dense = Linear(2*config.hidden_size,config.hidden_size)
        self.classifier = Linear(config.hidden_size, len(labels)) # final layer
    def forward(self, ids, attention_mask, token_type_ids):
        embedding_1 = self.model_1(ids[0], attention_mask = attention_mask[0], token_type_ids = token_type_ids[0])[0]
        embedding_2 = self.model_2(ids[1], attention_mask = attention_mask[1], token_type_ids = token_type_ids[1])[0] #torch.Size([16, 250, 768])
        pooled_output_1 = embedding_1.mean(axis=1) #torch.Size([16, 768])
        pooled_output_2 = embedding_2.mean(axis=1) #torch.Size([16, 768])
        pooled_output = cat([pooled_output_1,pooled_output_2],1) #torch.Size([16, 1536])
        pooled_output = self.drop(pooled_output)
        pooled_output = self.dense(pooled_output) #torch.Size([16, 768])
        scores = self.classifier(pooled_output)
        return scores