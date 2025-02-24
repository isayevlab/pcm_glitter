This is the data and code repository of the preprint titled: "All that glitters is not gold: Importance of rigorous evaluation of proteochemometric models" 
[Link: https://chemrxiv.org/engage/chemrxiv/article-details/678f32006dde43c908774ef1]

The Data[add link to figshare] used to train XGBoost models using AlphaFold2 (AF), ESM2 (ESM) and ProtBERT (PB) embeddings is found under `*_Embeddings` subfolders as well as the embeddings in both raw, padded and trimmed forms. See folder structure below.
```bash
Data/
├── ProtBERT_2D_Embeddings/    # Folder containing ProtBERT-based 2D embeddings
│   ├── protbert_trimmed_embeddings_0.95.pkl (924.2 MB)# Trimmed embeddings at 0.95 threshold
│   ├── protbert_raw_embeddings.pkl (561.1 MB)         # Raw embeddings from ProtBERT
│   ├── protbert_padded_embeddings_whole.pkl (9.0 GB)  # Padded full embeddings dataset
├── ESM_2D_Embeddings/         # Folder containing ESM-based 2D embeddings
│   ├── trimmed_embeddings_0.95.pkl (1.2 GB)           # Trimmed embeddings at 0.95 threshold
│   ├── trimmed_embeddings_0.15.pkl (1.5 GB)           # Trimmed embeddings at 0.15 threshold
│   ├── trimmed_embeddings_0.9.pkl (1.2 GB)            # Trimmed embeddings at 0.9 threshold
│   ├── trimmed_embeddings_0.8.pkl (1.3 GB)            # Trimmed embeddings at 0.8 threshold
│   ├── trimmed_embeddings_0.2.pkl (1.4 GB)            # Trimmed embeddings at 0.2 threshold
│   ├── trimmed_embeddings_0.1.pkl (1.7 GB)            # Trimmed embeddings at 0.1 threshold
│   ├── normed_embeddings.pkl (1.3 GB)                 # Normalized ESM embeddings
│   ├── ESM_padded_embeddings_whole.json (11.3 GB)     # Padded full embeddings dataset in JSON format
│   ├── ESM_embeddings_33.json (6.5 GB)                # ESM embeddings from layer 33
│   ├── ESM_embeddings_33_mean.json (21.5 MB)          # Mean-pooled embeddings from layer 33
├── AlphaFold2_2D_Embeddings/  # Folder containing AlphaFold2-based 2D embeddings
│   ├── trimmed_embeddings_0.95.pkl (346.6 MB)         # Trimmed embeddings at 0.95 threshold
│   ├── trimmed_embeddings_0.15.pkl (458.1 MB)         # Trimmed embeddings at 0.15 threshold
│   ├── trimmed_embeddings_0.9.pkl (368.0 MB)          # Trimmed embeddings at 0.9 threshold
│   ├── trimmed_embeddings_0.8.pkl (377.2 MB)          # Trimmed embeddings at 0.8 threshold
│   ├── trimmed_embeddings_0.2.pkl (423.0 MB)          # Trimmed embeddings at 0.2 threshold
│   ├── trimmed_embeddings_0.1.pkl (497.8 MB)          # Trimmed embeddings at 0.1 threshold
│   ├── normed_latent_embeddings.pkl (440.3 MB)        # Normalized latent embeddings
│   ├── normed_embeddings.pkl (390.4 MB)               # Normalized AlphaFold2 embeddings
│   ├── AF_padded_embeddings_whole.json (3.4 GB)       # Padded full embeddings dataset in JSON format
│   ├── AF_embeddings.json (9.4 GB)                    # AlphaFold2 embeddings in JSON format
│   ├── AF_embeddings_mean.json (30.4 MB)              # Mean-pooled AlphaFold2 embeddings
├── Optuna_Results.zip (141.6 MB)                # Results from Optuna hyperparameter optimization trials
├── Copy of Clean_Data_06042024.pkl (111.3 MB)   # Copy of cleaned dataset
├── Copy of chembl.csv (562.7 MB)                # Copy of ChEMBL dataset in CSV format
```

The main pipeline used to train the models is found in "optuna_main_reduced.py". It requires that the kinase activity data, chembl.csv and embedding files are found in the same folder.
It can be run from the command line as "python optuna_main_reduced.py SPLIT SNA EMB CHEMBLDIR".
