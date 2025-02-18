import os
import pickle
import pandas as pd
import numpy as np
import sys
import logging
import time
from datetime import timedelta

import sklearn

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit import rdBase

import xgboost as xgb
from sklearn.metrics import *
from sklearn.utils import shuffle as sh
from sklearn.model_selection import *
import matplotlib.pyplot as plt

import optuna
from optuna_integration import XGBoostPruningCallback
import kaleido
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

rdBase.DisableLog('rdApp.warning')


def rebalancing(chemblList, protein, ligand, kinase, groups, y, ratio, fold, split="kinase"):
    '''
    Given a list of chembl smiles, kinase labels, group labels and target labels, produces a matrix of inactive kinase-ligand pairs, all with label 0.
    Inputs:
        chemblList (list of str): list of SMILES strings
        protein, ligand, kinase, groups, y (np arrays): for each row of training data, protein embedding, ligand embedding, kinase label, group label and target label are given as separate arrays.
        ratio (int): Desired ratio of inactives: actives. 0 means no rebalancing.
        fold (int): Split number of current CV fold
        split (str): What kind of splitting strategy is used for CV. Options = ["equal", "kinase", "family"]. Default = "kinase"
    Outputs:
        sna_protein (np array): protein embedding vector of balanced ratio
        sna_ligand (np array): ligand embedding vector of balanced ratio
        sna_kinase (np array): kinase label of balanced ratio
        sna_groups (np array): group label vector of balanced ratio
        sna_y (np array): target vector of balanced ratio
    '''
    # 1. calculate current ratio of inactives: actives per kinase
    uniqueKinases = list(set(kinase))

    currRatioDict = {"Family": [], "Kinase": [], "0": [],
                     "1": [], "Ratio": [], "numInactives": []}

    # Make copies to avoid any accidental modifications of original arrays
    sna_protein = np.copy(protein)
    sna_ligand = np.copy(ligand)
    sna_kinase = np.copy(kinase)
    sna_groups = np.copy(groups)
    sna_y = np.copy(y)

    for k in uniqueKinases:
        # find indices of each occurrence of the kinase
        k_idx = np.where(kinase == k)[0]
        # print(k_idx)
        # Retrieve embedding for this kinase from original array
        k_prot = protein[k_idx][0]
        # Retrieve group for this kinase from original array
        k_group = str(groups[k_idx][0])
        k_tgt = y[k_idx]  # Retrieve labels for this kinase from original array

        numInactives = len(np.where(k_tgt == 0)[0])
        numActives = len(np.where(k_tgt == 1)[0])
        # avoid division by zero
        currRatio = numInactives / (numActives + 0.00001)

        # Add existing data points to dictionary
        currRatioDict["Family"].append(k_group)
        currRatioDict["Kinase"].append(k)
        currRatioDict["0"].append(numInactives)
        currRatioDict["1"].append(numActives)
        currRatioDict["Ratio"].append(currRatio)

        # 2. calculuate number of inactives needed for each kinase
        numInactivesNeeded = ratio * numActives - numInactives
        currRatioDict["numInactives"].append(numInactivesNeeded)
        print(
            f"Kinase: {k} | Current Ratio: {currRatio} | Number of inactives needed for ratio {ratio}: {numInactivesNeeded}")
        if numInactivesNeeded > 0:
            # 3. sample inactives for each kinase and store in a list
            randomSMILES = np.random.choice(
                chemblList, size=numInactivesNeeded, replace=False)
            randomMolecules = np.vstack(np.vectorize(generate_ecfp, otypes=[
                object])(randomSMILES))  # ligand embeddings for sampled inactives

            # 4. concatenate inactives with current training dataset
            sna_ligand = np.append(sna_ligand, randomMolecules, axis=0)

            new_prot = np.vstack([k_prot for i in range(numInactivesNeeded)])
            sna_protein = np.append(sna_protein, new_prot, axis=0)

            new_kinase = np.array([k for i in range(numInactivesNeeded)])
            sna_kinase = np.append(sna_kinase, new_kinase)

            new_groups = np.array(
                [k_group for i in range(numInactivesNeeded)])
            sna_groups = np.append(sna_groups, new_groups)

            new_y = np.array([0 for i in range(numInactivesNeeded)])
            sna_y = np.append(sna_y, new_y)

    columns = ["Family", "Kinase", "0", "1", "Ratio", "numInactives"]
    ratioDf = pd.DataFrame(currRatioDict, columns=columns)
    ratioDf.to_csv(f"Ratio_{ratio}_split_{split}_fold_{fold}_stats.csv")
    return sna_protein, sna_ligand, sna_kinase, sna_groups, sna_y


def featurize_smiles(smiles, nBits=1024):
    mol = Chem.MolFromSmiles(smiles)

    # Check if the SMILES is valid
    if mol is None:
        return None

    # Generate Morgan fingerprints (Circular fingerprints)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=5, nBits=nBits)

    # Convert the fingerprint to a NumPy array
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


def generate_ecfp(smiles, radius=5, nBits=1024):
    """
    Generate ECFP (Extended-Connectivity Fingerprints) for a list of SMILES.

    Args:
    - smiles_list (list): List of SMILES strings.
    - radius (int): Fingerprint radius.
    - n_bits (int): Number of bits in the fingerprint.

    Returns:
    - ecfp_list (list): List of ECFP fingerprints.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Generate Morgan fingerprints (ECFP)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        ecfp_arr = np.zeros((1,), dtype=np.int8)
        # Convert fingerprint to numpy array
        AllChem.DataStructs.ConvertToNumpyArray(ecfp, ecfp_arr)
    else:
        # Handle invalid SMILES
        print(f"Invalid SMILES: {smiles}")
    return ecfp


def cv_splitter(type, data):
    if type == "random":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return skf.split(data, data["ACTIVITY_STATUS"])
    elif type == "kinase":
        gss = GroupShuffleSplit(n_splits=5, train_size=.7, random_state=42)
        return gss.split(data, data["ACTIVITY_STATUS"], data["2_Gene_x"])
    elif type == "family":
        gss = GroupShuffleSplit(n_splits=5, train_size=.7, random_state=42)
        return gss.split(data, data["ACTIVITY_STATUS"], data["1_Group_x"])


def run_optuna(embName, data, embedding, model, split, sna):

    study_name = f"{model}_{embName}_{split}_SNA_{sna}"
    from optuna.samplers import TPESampler

    print("Begin optimization.")

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(),
                                storage=f"sqlite:///{study_name}.db",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10))

    study.set_user_attr("embedding", embName)
    study.set_user_attr("sna", sna)
    study.set_user_attr("model", model)
    study.set_user_attr("split", SPLIT)
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))

    study.optimize(lambda trial: objective(
        trial, data, embedding, sna, split), n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    trial_with_highest_f1 = max(study.best_trials, key=lambda t: t.values[0])
    logging.info(f"Trial with highest f1-score: ")
    logging.info(f"\tnumber: {trial_with_highest_f1.number}")
    logging.info(f"\tparams: {trial_with_highest_f1.params}")
    logging.info(f"\tvalues: {trial_with_highest_f1.values}")

    # impt = optuna.visualization.plot_param_importances(
    #    study, target=lambda t: t.values[0], target_name="F1-Score"
    #)
    # impt.write_image(f"hyp_impt_{model}_{split}_{embName}_{sna}.webp")

    return trial_with_highest_f1.params


def objective(trial, data, embedding, sna, split):

    # Define parameters
    model_params = {"random_state": 0,
                    "verbosity": 0,
                    "max_depth": trial.suggest_int("max_depth", 10, 60),
                    "max_leaves": trial.suggest_int("n_estimators", 1, 30),
                    "n_estimators": trial.suggest_int("max_leaves", 100, 2000),
                    "gamma": trial.suggest_float("gamma", 0.01, 0.5),
                    "n_jobs": 10, "device": "cpu",
                    "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 5, 200),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 0.5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 0.5),
                    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2),
                    "objective": "binary:logistic",
                    "eval_metric": "auc", }

    if split == "family":
        cv_inner = cv_splitter("kinase", data)
    else:
        cv_inner = cv_splitter(split, data)

    y = data["ACTIVITY_STATUS"].to_numpy()
    X_ligand_smiles = data["LIGAND_SMILES"].to_numpy()
    groups = data["1_Group_x"].to_numpy()
    kinase = data["2_Gene_x"].to_numpy()

    molecules = np.vectorize(generate_ecfp, otypes=[
        object])(X_ligand_smiles)
    molecules = np.vstack(molecules)

    all_f1 = []
    all_auc = []

    for i, (train_index, test_index) in enumerate(cv_inner):
        print(f"Inner Split {i+1}")

        protein_train, protein_test = embedding[train_index], embedding[test_index]
        ligand_train, ligand_test = molecules[train_index], molecules[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train, groups_test = groups[train_index], groups[test_index]
        kinase_train, kinase_test = kinase[train_index], kinase[test_index]

        ###########################
        # perform SNA here        #
        ###########################
        if sna > 0:
            print("Reading ChEMBL")

            chemDf = pd.read_csv(chemblDir, index_col=0, sep=";")
            chemDf = chemDf[["Smiles", "Inchi Key"]].dropna(axis=0, how="all")

            chemDf["Hash"] = chemDf['Inchi Key'].map(lambda x: x.split("-")[0])

            print("Removing actives from ChEMBL file.")

            data_inchi = data_no_mutant[["INCHIKEY", "LIGAND_SMILES"]].dropna(
                axis=0, how="all")
            data_inchi["Hash"] = data_inchi['INCHIKEY'].map(
                lambda x: x.split("-")[0])

            chemhash = set(chemDf["Hash"].to_list())
            datahash = set(data_inchi["Hash"].to_list())
            overlap = chemhash.intersection(datahash)

            print(
                f"There are {len(overlap)} ligands in common between ChEMBL and our data.")

            inactivesDf = chemDf[chemDf["Hash"].isin(overlap) == False]

            chemblList = inactivesDf["Smiles"].to_numpy()
            print(f"Rebalancing ratio during training")
            np.random.seed(0)

            protein_train, ligand_train, kinase_train, groups_train, y_train = rebalancing(
                chemblList, protein_train, ligand_train, kinase_train, groups_train, y_train, sna, i)

            print(
                f"Outputs of rebalancing: protein {protein_train.shape} | ligand {ligand_train.shape} | target {y_train.shape}")

            print(f"Rebalancing ratio during testing")
            np.random.seed(0)

            protein_test, ligand_test, kinase_test, groups_test, y_test = rebalancing(
                chemblList, protein_test, ligand_test, kinase_test, groups_test, y_test, sna, i)

            print(
                f"Outputs of rebalancing: protein {protein_test.shape} | ligand {ligand_test.shape} | target {y_test.shape}")

        # Concatenate protein and ligand embeddings
        X_train = np.concatenate((protein_train, ligand_train), axis=1)
        X_test = np.concatenate((protein_test, ligand_test), axis=1)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)

        # print("Here: ", dtrain.num_col())

        # Train Model
        if i == 0:
            pruning_callback = optuna.integration.XGBoostPruningCallback(
                trial, "validation-auc")
            model = xgb.train(model_params, dtrain, evals=[
                              (dvalid, "validation")], verbose_eval=0, callbacks=[pruning_callback])
        else:
            model = xgb.train(model_params, dtrain, evals=[
                              (dvalid, "validation")], verbose_eval=0)

        y_pred = model.predict(dvalid)
        y_pred = np.rint(y_pred)
        # y_pred_prob = model.predict_proba(X_test)
        # auc = roc_auc_score(y_test, y_pred_prob[:, 1])

        f1 = classification_report(y_test, y_pred, output_dict=True, zero_division=1)[
            '1']["f1-score"]
        all_f1.append(f1)
        # all_auc.append(auc)

    f1_mean = np.mean(all_f1)
    # auc_mean = np.mean(all_auc)

    return f1_mean  # , auc_mean


if __name__ == "__main__":
    start_time = time.time()
    dataFile = "Clean_Data_06042024.pkl"
    SPLIT = sys.argv[1]
    SNA = int(sys.argv[2])
    MODEL = "XgBoost"
    embDir = sys.argv[3]
    chemblDir = sys.argv[4]
    embName = sys.argv[5]
    ROOT = os.getcwd()

    logging.basicConfig(filename=f"{embName}_{MODEL}_{SNA}_{SPLIT}.log",
                        level=logging.INFO, filemode="w")
    logger = logging.getLogger('__name__')
    logging.getLogger().setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    logger.addHandler(ch)

    logger.info("Logger initialized. Reading in data.")

    # Read in data and ChEMBL
    # Perform CV split randomly

    np.random.seed(42)

    # Read data again to ensure same data is used
    data = pd.read_pickle(dataFile)

    allColumns = data.columns

    keepColumns = ["1_Group_x", "2_Gene_x",
                   "LIGAND_SMILES", "INCHIKEY", "ACTIVITY_STATUS", "PROTEIN_EMBEDDING"]
    dropColumns = [i for i in allColumns if i not in keepColumns]
    data_no_mutant = data.drop(columns=dropColumns)

    logger.info("Performing CV split")

    cv_splits = cv_splitter(SPLIT, data_no_mutant)

    ###################################
    # Run hyperparameter optimization #
    ###################################

    CV_splits = []
    logger.info("Reading in embedding file")

    embs = pd.read_json(embDir)
    embs["embeddings"] = embs["embeddings"].apply(
        lambda x: np.mean(np.array(x), axis=0))
    # keep only embeddings relevant for our data
    embs = embs[embs["family"].isin(["CMGC", "TKL", "TYR"])]

    best_fold_f1 = 0
    best_fold_auc = 0
    best_fold = np.inf
    model_best_params = {}

    all_f1 = []
    all_auc = []

    logger.info("Running outer cross-validation.")

    for i, (train_index, test_index) in enumerate(cv_splits):
        logger.info(f"Outer Split {i+1}")
        logger.info("Extracting embeddings.")

        # Split dataset into train and test folds
        data_train = data_no_mutant.iloc[train_index]
        data_test = data_no_mutant.iloc[test_index]

        # Extract kinase and group names for train and test folds
        kinase_train, kinase_test = data_train["2_Gene_x"].to_numpy(
        ), data_test["2_Gene_x"].to_numpy()
        groups_train, groups_test = data_train["1_Group_x"].to_numpy(
        ), data_test["1_Group_x"].to_numpy()

        # Extract protein embeddings and split into train and test folds
        protein_train = np.array(
            [embs[embs["kinase"] == k]["embeddings"].to_numpy()[0] for k in kinase_train])
        assert protein_train.shape[1] == 1280
        assert protein_train.shape[0] == data_train.shape[0]

        protein_test = np.array(
            [embs[embs["kinase"] == k]["embeddings"].to_numpy()[0] for k in kinase_test])
        assert protein_test.shape[1] == 1280
        assert protein_test.shape[0] == data_test.shape[0]

        # Run hyperparameter optimization

        logger.info("Running hyperparameter optimization.")

        best_params = run_optuna(
            embName, data_train, protein_train, model=MODEL, split=SPLIT, sna=SNA)

        # Extract ligand and label for train and test folds
        y_train, y_test = data_train["ACTIVITY_STATUS"].to_numpy(
        ), data_test["ACTIVITY_STATUS"].to_numpy()

        smiles = data_no_mutant["LIGAND_SMILES"].to_numpy()
        molecules = np.vectorize(generate_ecfp, otypes=[
            object])(smiles)
        molecules = np.vstack(molecules)

        ligand_train, ligand_test = molecules[train_index], molecules[test_index]

        ###########################
        # perform SNA here        #
        ###########################
        if SNA > 0:
            logger.info("Reading ChEMBL")

            chemDf = pd.read_csv(chemblDir, index_col=0, sep=";")
            chemDf = chemDf[["Smiles", "Inchi Key"]].dropna(axis=0, how="all")

            chemDf["Hash"] = chemDf['Inchi Key'].map(lambda x: x.split("-")[0])

            print("Removing actives from ChEMBL file.")

            data_inchi = data_no_mutant[["INCHIKEY", "LIGAND_SMILES"]].dropna(
                axis=0, how="all")
            data_inchi["Hash"] = data_inchi['INCHIKEY'].map(
                lambda x: x.split("-")[0])

            chemhash = set(chemDf["Hash"].to_list())
            datahash = set(data_inchi["Hash"].to_list())
            overlap = chemhash.intersection(datahash)

            print(
                f"There are {len(overlap)} ligands in common between ChEMBL and our data.")

            inactivesDf = chemDf[chemDf["Hash"].isin(overlap) == False]

            chemblList = inactivesDf["Smiles"].to_numpy()
            print(f"Rebalancing ratio during training")
            np.random.seed(0)

            protein_train, ligand_train, kinase_train, groups_train, y_train = rebalancing(
                chemblList, protein_train, ligand_train, kinase_train, groups_train, y_train, SNA, i)

            logger.info(
                f"Outputs of rebalancing train: protein {protein_train.shape} | ligand {ligand_train.shape} | target {y_train.shape}")

            print(f"Rebalancing ratio during testing")
            np.random.seed(0)

            protein_test, ligand_test, kinase_test, groups_test, y_test = rebalancing(
                chemblList, protein_test, ligand_test, kinase_test, groups_test, y_test, SNA, i)

            logger.info(
                f"Outputs of rebalancing test: protein {protein_test.shape} | ligand {ligand_test.shape} | target {y_test.shape}")

        # Concatenate protein and ligand embeddings
        X_train = np.concatenate((protein_train, ligand_train), axis=1)
        X_test = np.concatenate((protein_test, ligand_test), axis=1)

        # Train optimal model with best hyperparameters
        logger.info("Training optimal model")
        opt_model = xgb.XGBClassifier().set_params(**best_params)
        opt_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

        # Run prediction
        logger.info("Predicting for test data")
        y_pred = opt_model.predict(X_test)
        y_pred_prob = opt_model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred_prob[:, 1])
        f1 = classification_report(y_test, y_pred, output_dict=True, zero_division=1)[
            '1']["f1-score"]

        all_f1.append(f1)
        all_auc.append(auc)

        # Compare with history
        if f1 > best_fold_f1:
            best_fold_f1 = f1
            best_fold_auc = auc
            best_fold = i
            model_best_params = best_params

    logger.info(f"Best F1-Score: {best_fold_f1}")
    logger.info(f"Best AUC: {best_fold_auc}")
    logger.info(f"Best Fold: {best_fold}")
    logger.info(f"Best Params: {model_best_params}")

    mean_f1 = np.mean(all_f1)
    mean_auc = np.mean(all_auc)

    std_f1 = np.std(all_f1)
    std_auc = np.std(all_auc)

    logger.info(f"Mean F1-Score: {mean_f1}")
    logger.info(f"Standard Deviation F1-Score: {std_f1}")
    logger.info(f"Mean AUC: {mean_auc}")
    logger.info(f"Standard Deviation AUC: {std_auc}")

    logger.info("Saving results")

    embSaveDir = os.path.join(ROOT, embName)
    os.makedirs(embSaveDir, exist_ok=True)

    with open(os.path.join(embSaveDir, f"best_params_{MODEL}_{SPLIT}_{SNA}.pkl"), "wb") as f:
        pickle.dump(model_best_params, f)

    with open(os.path.join(embSaveDir, f"best_params_{MODEL}_{SPLIT}_{SNA}.pkl"), "rb") as f:
        model_best_params = pickle.load(f)

    shuffleList = ["None", "Protein", "Ligand", "Label"]
    allResults = []

    ##########################################################
    # Initialize figure to compare across shuffle conditions #
    ##########################################################

    fig1, ax1 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig1.suptitle(f"ROC for {embName} | {MODEL} | Split: {SPLIT} | SNA: {SNA}")

    ax1[0].plot([0, 1], [0, 1], linestyle='--', color='black')
    ax1[0].set_title("Train")
    ax1[0].set_xlabel('False Positive Rate')
    ax1[0].set_ylabel('True Positive Rate')

    ax1[1].plot([0, 1], [0, 1], linestyle='--', color='black')
    ax1[1].set_title("Test")
    ax1[1].set_xlabel('False Positive Rate')
    ax1[1].set_ylabel('True Positive Rate')

    auc_test = 0
    auc_train = 0
    fpr_train, tpr_train, fpr_test, tpr_test = 0, 0, 0, 0

    for shuffle in shuffleList:
        logger.info(f"Running {shuffle} shuffle")

        f1_scores = []
        auc_scores = []
        baseline_scores = {}

        ######################################################
        #      Cross-validation for each shuffle type        #
        ######################################################
        cv_splits = cv_splitter(SPLIT, data_no_mutant)

        for i, (train_index, test_index) in enumerate(cv_splits):
            logger.info(f"Split {i+1}")

            logger.info("Extracting vectors")

            # Split dataset into train and test folds
            data_train = data_no_mutant.iloc[train_index]
            data_test = data_no_mutant.iloc[test_index]

            # Extract kinase and group names for train and test folds
            kinase_train, kinase_test = data_train["2_Gene_x"].to_numpy(
            ), data_test["2_Gene_x"].to_numpy()
            groups_train, groups_test = data_train["1_Group_x"].to_numpy(
            ), data_test["1_Group_x"].to_numpy()

            # Extract protein embeddings and split into train and test folds
            protein_train = np.array(
                [embs[embs["kinase"] == k]["embeddings"].to_numpy()[0] for k in kinase_train])
            assert protein_train.shape[1] == 1280
            assert protein_train.shape[0] == data_train.shape[0]

            protein_test = np.array(
                [embs[embs["kinase"] == k]["embeddings"].to_numpy()[0] for k in kinase_test])
            assert protein_test.shape[1] == 1280
            assert protein_test.shape[0] == data_test.shape[0]

            # Extract ligand and label for train and test folds
            y_train, y_test = data_train["ACTIVITY_STATUS"].to_numpy(
            ), data_test["ACTIVITY_STATUS"].to_numpy()

            smiles = data_no_mutant["LIGAND_SMILES"].to_numpy()
            molecules = np.vectorize(generate_ecfp, otypes=[
                object])(smiles)
            molecules = np.vstack(molecules)

            ligand_train, ligand_test = molecules[train_index], molecules[test_index]

            ###########################
            #     perform SNA here    #
            ###########################
            if SNA > 0:
                logger.info("Reading ChEMBL")

                chemDf = pd.read_csv(chemblDir, index_col=0, sep=";")
                chemDf = chemDf[["Smiles", "Inchi Key"]].dropna(
                    axis=0, how="all")

                chemDf["Hash"] = chemDf['Inchi Key'].map(
                    lambda x: x.split("-")[0])

                print("Removing actives from ChEMBL file.")

                data_inchi = data_no_mutant[["INCHIKEY", "LIGAND_SMILES"]].dropna(
                    axis=0, how="all")
                data_inchi["Hash"] = data_inchi['INCHIKEY'].map(
                    lambda x: x.split("-")[0])

                chemhash = set(chemDf["Hash"].to_list())
                datahash = set(data_inchi["Hash"].to_list())
                overlap = chemhash.intersection(datahash)

                print(
                    f"There are {len(overlap)} ligands in common between ChEMBL and our data.")

                inactivesDf = chemDf[chemDf["Hash"].isin(overlap) == False]

                chemblList = inactivesDf["Smiles"].to_numpy()
                logger.info(f"Rebalancing ratio during training")
                np.random.seed(0)

                protein_train, ligand_train, kinase_train, groups_train, y_train = rebalancing(
                    chemblList, protein_train, ligand_train, kinase_train, groups_train, y_train, SNA, i)

                logger.info(
                    f"Outputs of rebalancing: protein {protein_train.shape} | ligand {ligand_train.shape} | target {y_train.shape}")

                logger.info(f"Rebalancing ratio during testing")
                np.random.seed(0)

                protein_test, ligand_test, kinase_test, groups_test, y_test = rebalancing(
                    chemblList, protein_test, ligand_test, kinase_test, groups_test, y_test, SNA, i)

                logger.info(
                    f"Outputs of rebalancing: protein {protein_test.shape} | ligand {ligand_test.shape} | target {y_test.shape}")

            ###########################
            # perform shuffling here  #
            ###########################

            if shuffle == "Protein":
                np.random.shuffle(protein_train)
            elif shuffle == "Ligand":
                np.random.shuffle(ligand_train)
            elif shuffle == "Label":
                np.random.shuffle(y_train)

            # Concatenate protein and ligand embeddings
            X_train = np.concatenate((protein_train, ligand_train), axis=1)
            X_test = np.concatenate((protein_test, ligand_test), axis=1)

            # Train optimal model with best hyperparameters
            logger.info("Training model")
            opt_model = xgb.XGBClassifier(
                importance_type="gain").set_params(**model_best_params)
            opt_model.fit(X_train, y_train, eval_set=[
                          (X_test, y_test)], verbose=0)

            # Run prediction
            logger.info("Predicting for train data")
            y_pred = opt_model.predict(X_train)
            y_pred_prob = opt_model.predict_proba(X_train)
            auc_train = roc_auc_score(y_train, y_pred_prob[:, 1])
            fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_prob[:, 1])

            logger.info("Predicting for test data")
            y_pred = opt_model.predict(X_test)
            y_pred_prob = opt_model.predict_proba(X_test)
            auc_test = roc_auc_score(y_test, y_pred_prob[:, 1])
            fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_prob[:, 1])
            f1 = classification_report(y_test, y_pred, output_dict=True, zero_division=1)[
                '1']["f1-score"]
            print("F1: ", f1)
            f1_scores.append(f1)
            auc_scores.append(auc_test)

            ###########################
            # Generate baseline scores#
            ###########################

            if shuffle == "None":
                logger.info("Generating baseline scores")
                baseline_scores[i] = {}
                # Baseline 1: All-Positive predictions
                pos_score = classification_report(y_test, np.ones_like(
                    y_pred), output_dict=True, zero_division=1)
                # print(pos_score)
                baseline_scores[i]["Positive"] = pos_score['1']["f1-score"]

                # Baseline 2: All-Random predictions (50 simulations)
                random_predictions = np.array([np.random.choice(
                    np.random.permutation(y_test), size=y_test.shape[0]) for _ in range(50)])
                # print(random_predictions.shape)
                random_scores = [classification_report(y_test, random_predictions[i, :], output_dict=True, zero_division=1)[
                    '1']["f1-score"] for i in range(50)]
                baseline_scores[i]["Random"] = np.mean(random_scores)

            else:
                baseline_scores[i] = {}

        ################
        # Save results #
        ################
        logger.info("Saving results")
        print(f1_scores)
        mean_f1 = np.mean(f1_scores)
        mean_auc = np.mean(auc_scores)

        std_f1 = np.std(f1_scores)
        std_auc = np.std(auc_scores)

        if shuffle == "None":
            # Plot baseline results
            mean_pos_f1 = np.mean([baseline_scores[i]["Positive"]
                                   for i in baseline_scores])
            mean_random_f1 = np.mean(
                [baseline_scores[i]["Random"] for i in baseline_scores])
            ax1[2].axhline(mean_pos_f1, linestyle='--',
                           color='black', label="All-Positive")
            ax1[2].axhline(mean_random_f1, linestyle='-',
                           color='black', label="All-Random")

        else:
            mean_pos_f1 = np.nan
            mean_random_f1 = np.nan

        result = {"ratio": SNA, "embedding": embName, "model": MODEL,
                  "shuffle": shuffle, "Mean_F1": mean_f1, "Std_F1": std_f1, "Mean_AUC": mean_auc, "Std_AUC": std_auc,
                  "Baseline_Pos_F1": mean_pos_f1, "Baseline_Ran_F1": mean_random_f1, "split": SPLIT}

        allResults.append(result)

        ############################
        #  Analyze model mistakes  #
        ############################
        logger.info("Analyzing model mistakes.")

        logger.info("Plotting results.")

        fig, ax = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
        fig.suptitle(
            f"Model performance | {embName} | {MODEL} | Ratio: {SNA} | Shuffle: {shuffle} | Split: {SPLIT}")

        ax[0].hist(y_pred_prob[:, 1])
        ax[0].set_xlabel("Prediction Probability")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title(f"Probability of predicting \"Active\"")

        # Plot confusion matrix
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
        disp = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=opt_model.classes_)
        disp.plot(ax=ax[1])
        ax[1].grid(False)
        ax[1].set_xlabel("Predicted Label")
        ax[1].set_ylabel("True Label")
        ax[1].set_title(f"Confusion Matrix")

        features = opt_model.feature_importances_
        prot_feats = np.sum(features[:1280])
        lig_feats = np.sum(features[1280:])
        ax[2].bar(x=["Protein", "Ligand"], height=[prot_feats, lig_feats])
        ax[2].set_xlabel("Feature")
        ax[2].set_ylabel("Importance")
        ax[2].set_title(f"Feature Importance")

        fig.savefig(os.path.join(
            embSaveDir, f"Performance_{embName}_{MODEL}_{SNA}_{SPLIT}_{shuffle}.png"))

        ######################
        #   Plot ROC-curve   #
        ######################
        ax1[0].plot(fpr_train, tpr_train,
                    label=f"Shuffle={shuffle} | AUC = {auc_train:.3f}")
        ax1[1].plot(fpr_test, tpr_test,
                    label=f"Shuffle={shuffle} | AUC = {auc_test:.3f}")

    logger.info("Writing results to file.")

    with open(os.path.join(embSaveDir, f"results_{MODEL}_{SPLIT}_{SNA}.pkl"), "wb") as f:
        pickle.dump(allResults, f)

    ######################
    #   Plot F1-scores   #
    ######################

    logger.info("Plotting F-1 Scores")

    ax1[0].legend()
    ax1[1].legend()

    f1_scores = [result["Mean_F1"] for result in allResults]
    f1_scores_std = [result["Std_F1"] for result in allResults]
    print(f1_scores, f1_scores_std, shuffleList)
    ax1[2].bar(x=shuffleList, height=f1_scores, yerr=f1_scores_std)
    ax1[2].set_title("Mean F1-score across CV folds")
    ax1[2].set_xlabel("Shuffle")
    ax1[2].set_ylabel("F1-score")
    ax1[2].legend()

    fig1.savefig(os.path.join(
        embSaveDir, f"ROC_{embName}_{MODEL}_{SNA}_{SPLIT}.png"))

    end_time = time.time()
    duration = end_time - start_time
    duration_timedelta = str(timedelta(seconds=duration))
    logger.info('Time elapsed (hh:mm:ss.ms) {}'.format(duration_timedelta))
    logger.info("Finished.")
