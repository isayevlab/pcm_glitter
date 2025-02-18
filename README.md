This is the data and code repository of the preprint titled: "All that glitters is not gold: Importance of rigorous evaluation of proteochemometric models" 
[Link: https://chemrxiv.org/engage/chemrxiv/article-details/678f32006dde43c908774ef1]

The data used to train XGBoost models using AlphaFold2 (AF), ESM and ProtBERT (PB) embeddings is found under ___.
The embeddings in both raw, padded and trimmed forms are found under __.

The main pipeline used to train the models is found in "optuna_main_reduced.py". It requires that the kinase activity data, chembl.csv and embedding files are found in the same folder.
It can be run from the command line as "python optuna_main_reduced.py SPLIT SNA EMB CHEMBLDIR".
