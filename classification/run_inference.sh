python3 ./src/inference.py \
  --test ../extraction/data/1.tsv \
  -c1 instruction_hint_example \
  -c2 statement \
  --modele modeles/ex_classif/saved_model_classification_ft_camembert.pt \
  --modelebase modeles/camembert-base \
  --bertarchi single \
  --ypredtxtfile pred_1.txt \
  --ypredtsvfile pred_1.tsv
