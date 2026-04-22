
# Importing required libraries
import random
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from econml.dr import SparseLinearDRLearner, ForestDRLearner, LinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from scipy.stats import expon
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm
from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier
from dowhy.causal_estimator import CausalEstimate
from sklearn.preprocessing import StandardScaler
from econml.dr import DRLearner
from sklearn.linear_model import LassoCV
from econml.dml import DML, DML



# Set seeds for reproducibility
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)



#%%

file_path = '/content/drive/MyDrive/malaria_final.csv'
data_all_malaria = pd.read_csv(file_path, encoding='latin1')


data_all_malaria['excess'] = (data_all_malaria['sir'] > 1).astype(int)

data_all_malaria = data_all_malaria.dropna()

columnas_to_drop = ['population', 'min_masl', 'cases', 'sir']

# 1. Label Encoding DANE
le = LabelEncoder()
data_all_malaria['DANE_labeled'] = le.fit_transform(data_all_malaria['DANE'])
scaler = MinMaxScaler()
data_all_malaria['DANE_normalized'] = scaler.fit_transform(
    data_all_malaria[['DANE_labeled']]
)

# 2. Label Encoding Deparment_DANE
le_year = LabelEncoder()
data_all_malaria['DANE_year_labeled'] = le_year.fit_transform(data_all_malaria['DANEYear'])
scaler_DDANE = MinMaxScaler()
data_all_malaria['DANE_year_normalized'] = scaler_DDANE.fit_transform(
    data_all_malaria[['DANE_year_labeled']]
)


data_all_malaria.drop(columns=columnas_to_drop, inplace=True)

std_HFP = data_all_malaria['HFP'].std()
print(f"std of HFP: {std_HFP}")

median_HFP = data_all_malaria['HFP'].median()
print(f"median of HFP: {median_HFP}")

scaler = StandardScaler()
data_all_malaria['temperature'] = scaler.fit_transform(data_all_malaria[['temperature']])
data_all_malaria['rainfall'] = scaler.fit_transform(data_all_malaria[['rainfall']])
data_all_malaria['Forest'] = scaler.fit_transform(data_all_malaria[['Forest']])
data_all_malaria['Deforest'] = scaler.fit_transform(data_all_malaria[['Deforest']])
data_all_malaria['Fire'] = scaler.fit_transform(data_all_malaria[['Fire']])
data_all_malaria['Mining'] = scaler.fit_transform(data_all_malaria[['Mining']])
data_all_malaria['Coca'] = scaler.fit_transform(data_all_malaria[['Coca']])
data_all_malaria['rMisery'] = scaler.fit_transform(data_all_malaria[['rMisery']])
data_all_malaria['uMisery'] = scaler.fit_transform(data_all_malaria[['uMisery']])
data_all_malaria['HFP'] = scaler.fit_transform(data_all_malaria[['HFP']])

# std
data_std_malaria = data_all_malaria

data_std_malaria = data_std_malaria.sort_values(
    by=['DANE_normalized', 'DANE_year_normalized']
).reset_index(drop=True)

data_std_malaria = data_std_malaria.dropna()

data_std_malaria = data_std_malaria[['DANE_normalized', 'DANE_year_normalized',
                     'temperature', 'rainfall', 'Forest', 'Deforest', 'Fire', 'Mining', 'Coca', 'rMisery', 'uMisery',
                     'HFP', 'excess']]

model_malaria = CausalModel(
    data=data_std_malaria,
    treatment=['HFP'],
    outcome=['excess'],
    graph="""
    digraph {

        rainfall -> temperature
        rainfall -> Forest
        temperature -> Forest
        rainfall -> HFP
        temperature -> HFP
        rainfall -> excess
        temperature -> excess


        Forest -> HFP
        Forest -> excess


        Forest -> Deforest
        rMisery -> Deforest
        uMisery -> Deforest
        Deforest -> HFP
        Deforest -> excess


        Forest -> Coca
        rMisery -> Coca
        uMisery -> Coca
        Coca -> HFP
        Coca -> excess


        Forest -> Mining
        rMisery -> Mining
        uMisery -> Mining
        Mining -> HFP
        Mining -> excess


        Forest -> Fire
        Deforest -> Fire
        Coca -> Fire
        rMisery -> Fire
        uMisery -> Fire
        Fire -> HFP
        Fire -> excess


        rMisery -> HFP
        rMisery -> excess
        uMisery -> HFP
        uMisery -> excess


        HFP -> excess

        DANE_normalized -> excess

        DANE_year_normalized -> excess


    }
    """
)

# Identifying effects
identified_estimand_malaria = model_malaria.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_malaria)

import numpy as np
import random
import os

SEED = 123
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'  

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from econml.dml import DML
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
import pandas as pd

Y = data_std_malaria['excess'].values.astype(int)
T = data_std_malaria['HFP'].values                 
X = data_std_malaria[['rMisery', 'uMisery', 'rainfall', 'temperature', 'DANE_normalized', 'DANE_year_normalized']].values

W = data_std_malaria[['Coca', 'Forest', 'Mining', 'Fire', 'Deforest', 'rMisery', 'uMisery', 'rainfall', 'temperature']].values

model_configs = [
    {"name": "Model 1", "n_estimators": 20, "max_depth": 5},
    {"name": "Model 2", "n_estimators": 20, "max_depth": 7},
    {"name": "Model 3", "n_estimators": 30, "max_depth": 5},
    {"name": "Model 4", "n_estimators": 30, "max_depth": 8},
    {"name": "Model 5", "n_estimators": 50, "max_depth": 5},
    {"name": "Model 6", "n_estimators": 50, "max_depth": 7},
    {"name": "Model 7", "n_estimators": 75, "max_depth": 5},
    {"name": "Model 8", "n_estimators": 75, "max_depth": 8},
    {"name": "Model 9", "n_estimators": 100, "max_depth": 4},
    {"name": "Model 10", "n_estimators": 100, "max_depth": 7}
]

fixed_params = {
    "eta": 0.0001,
    "reg_lambda": 1.5,
    "alpha": 0.001,
    "random_state": SEED,
    "tree_method": "exact",    
    "nthread": 1,               
    "verbosity": 0              
}


results = []
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for i, config in enumerate(model_configs, 1):
    print(f"  n_estimators={config['n_estimators']}, max_depth={config['max_depth']}")

    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        T_tr, T_val = T[train_idx], T[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]
        W_tr, W_val = W[train_idx], W[val_idx]

        model_y = xgb.XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            objective="binary:logistic",
            **fixed_params
        )

        model_t = xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            objective="reg:squarederror",
            **fixed_params
        )

        cv_internal = KFold(n_splits=3, shuffle=True, random_state=SEED)

        est = DML(
            model_y=model_y,
            model_t=model_t,
            model_final= LassoCV(alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1],
                                 fit_intercept=False,
                                 max_iter=50000,
                                 tol=1e-3,
                                 cv=3,
                                 n_jobs=-1),
            featurizer=PolynomialFeatures(degree=3, include_bias=False),
            discrete_treatment=False,
            discrete_outcome=True,  
            cv=cv_internal,          
            random_state=SEED,
            fit_cate_intercept=True
        )

        try:
            est.fit(Y=Y_tr, T=T_tr, X=X_tr, W=W_tr)

            fold_score = est.score(Y_val, T_val, X=X_val, W=W_val)
            scores.append(fold_score)
            print(f"    Fold {fold}: score = {fold_score:.6f}  (R-loss ≈ {-fold_score:.6f})")

        except Exception as e:
            scores.append(np.nan)

    scores_clean = [s for s in scores if not np.isnan(s)]
    if len(scores_clean) == 0:
        mean_score = np.nan
        std_score = np.nan
    else:
        mean_score = np.mean(scores_clean)
        std_score = np.std(scores_clean)

    r_loss_mean = -mean_score if not np.isnan(mean_score) else np.nan
    r_loss_std = std_score if not np.isnan(std_score) else np.nan

    results.append({
        "name": config["name"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "score_mean": mean_score,
        "score_std": std_score,
        "r_loss_mean": r_loss_mean,
        "r_loss_std": r_loss_std,
        "valid_folds": len(scores_clean)
    })


results_df = pd.DataFrame(results).sort_values("r_loss_mean").reset_index(drop=True)

print(results_df[['name', 'n_estimators', 'max_depth', 'r_loss_mean', 'r_loss_std', 'valid_folds']].to_string(index=False, float_format="%.6f"))

best = results_df.iloc[0]

#%%

file_path = 'D:/dengue_final.csv'
data_all_dengue = pd.read_csv(file_path, encoding='latin1')


data_all_dengue['excess'] = (data_all_dengue['sir'] > 1).astype(int)

data_all_dengue = data_all_dengue.dropna()

columnas_to_drop = ['population', 'min_masl', 'cases', 'sir']

# 1. Label Encoding DANE
le = LabelEncoder()
data_all_dengue['DANE_labeled'] = le.fit_transform(data_all_dengue['DANE'])
scaler = MinMaxScaler()
data_all_dengue['DANE_normalized'] = scaler.fit_transform(
    data_all_dengue[['DANE_labeled']]
)

# 2. Label Encoding Deparment_DANE
le_year = LabelEncoder()
data_all_dengue['DANE_year_labeled'] = le_year.fit_transform(data_all_dengue['DANEYear'])
scaler_DDANE = MinMaxScaler()
data_all_dengue['DANE_year_normalized'] = scaler_DDANE.fit_transform(
    data_all_dengue[['DANE_year_labeled']]
)


data_all_dengue.drop(columns=columnas_to_drop, inplace=True)

std_HFP = data_all_dengue['HFP'].std()
print(f"std of HFP: {std_HFP}")

median_HFP = data_all_dengue['HFP'].median()
print(f"median of HFP: {median_HFP}")

scaler = StandardScaler()
data_all_dengue['temperature'] = scaler.fit_transform(data_all_dengue[['temperature']])
data_all_dengue['rainfall'] = scaler.fit_transform(data_all_dengue[['rainfall']])
data_all_dengue['House'] = scaler.fit_transform(data_all_dengue[['House']])
data_all_dengue['Services'] = scaler.fit_transform(data_all_dengue[['Services']])
data_all_dengue['Overcrowding'] = scaler.fit_transform(data_all_dengue[['Overcrowding']])
data_all_dengue['Urban'] = scaler.fit_transform(data_all_dengue[['Urban']])
data_all_dengue['Ethnic'] = scaler.fit_transform(data_all_dengue[['Ethnic']])
data_all_dengue['rMisery'] = scaler.fit_transform(data_all_dengue[['rMisery']])
data_all_dengue['uMisery'] = scaler.fit_transform(data_all_dengue[['uMisery']])
data_all_dengue['HFP'] = scaler.fit_transform(data_all_dengue[['HFP']])

# std
data_std_dengue = data_all_dengue

data_std_dengue = data_std_dengue.sort_values(
    by=['DANE_normalized', 'DANE_year_normalized']
).reset_index(drop=True)

data_std_dengue = data_std_dengue.dropna()

data_std_dengue = data_std_dengue[['DANE_normalized', 'DANE_year_normalized',
                     'temperature', 'rainfall', 'House', 'Services', 'Overcrowding', 'Urban', 'Ethnic', 'rMisery', 'uMisery',
                     'HFP', 'excess']]


model_dengue = CausalModel(
    data=data_std_dengue,
    treatment=['HFP'],
    outcome=['excess'],
    graph="""
    digraph {

        rainfall -> temperature
        rainfall -> HFP
        temperature -> HFP
        rainfall -> excess
        temperature -> excess

        House -> HFP
        House -> excess
        Ethnic -> HFP
        Ethnic -> excess
        Services -> HFP
        Services -> excess
        Overcrowding -> HFP
        Overcrowding -> excess
        Urban -> HFP
        Urban -> excess
        rMisery -> HFP
        rMisery -> excess
        uMisery -> HFP
        uMisery -> excess

        House -> Services
        House -> Overcrowding
        House -> Ethnic
        House -> rMisery
        House -> uMisery


        Ethnic -> Overcrowding
        Ethnic -> Services
        Ethnic -> Urban
        Ethnic -> rMisery
        Ethnic -> uMisery

        Urban -> Services
        Urban -> rMisery
        Urban -> uMisery

        Overcrowding -> Services
        Overcrowding -> rMisery
        Overcrowding -> uMisery

        Services -> rMisery
        Services -> uMisery

        rMisery -> excess
        uMisery -> excess

        HFP -> excess

        DANE_normalized -> excess
        DANE_year_normalized -> excess

    }
    """
)

# Identifying effects
identified_estimand_dengue = model_dengue.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_dengue)

import numpy as np
import random
import os

SEED = 123
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'  

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

Y = data_std_dengue['excess'].values.astype(int)
T = data_std_dengue['HFP'].values                 
X = data_std_dengue[['rMisery', 'uMisery', 'rainfall', 'temperature', 'DANE_normalized', 'DANE_year_normalized']].values

W = data_std_dengue[['rainfall','temperature','House','uMisery','Overcrowding','rMisery','Services','Ethnic','Urban']].values

model_configs = [
    {"name": "Model 1", "n_estimators": 20, "max_depth": 5},
    {"name": "Model 2", "n_estimators": 20, "max_depth": 7},
    {"name": "Model 3", "n_estimators": 30, "max_depth": 5},
    {"name": "Model 4", "n_estimators": 30, "max_depth": 8},
    {"name": "Model 5", "n_estimators": 50, "max_depth": 5},
    {"name": "Model 6", "n_estimators": 50, "max_depth": 7},
    {"name": "Model 7", "n_estimators": 75, "max_depth": 5},
    {"name": "Model 8", "n_estimators": 75, "max_depth": 8},
    {"name": "Model 9", "n_estimators": 100, "max_depth": 4},
    {"name": "Model 10", "n_estimators": 100, "max_depth": 7}
]

fixed_params = {
    "eta": 0.0001,
    "reg_lambda": 1.5,
    "alpha": 0.001,
    "random_state": SEED,
    "tree_method": "exact",    
    "nthread": 1,              
    "verbosity": 0            
}

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for i, config in enumerate(model_configs, 1):
    print(f"  n_estimators={config['n_estimators']}, max_depth={config['max_depth']}")

    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        T_tr, T_val = T[train_idx], T[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]
        W_tr, W_val = W[train_idx], W[val_idx]

        model_y = xgb.XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            objective="binary:logistic",
            **fixed_params
        )

        model_t = xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            objective="reg:squarederror",
            **fixed_params
        )

        cv_internal = KFold(n_splits=3, shuffle=True, random_state=SEED)

        est = DML(
            model_y=model_y,
            model_t=model_t,
            model_final= LassoCV(alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1],  
                                 fit_intercept=False,
                                 max_iter=50000,
                                 tol=1e-3,
                                 cv=3,           
                                 n_jobs=-1),
            featurizer=PolynomialFeatures(degree=3, include_bias=False),
            discrete_treatment=False,
            discrete_outcome=True,  
            cv=cv_internal,          
            random_state=SEED,
            fit_cate_intercept=True
        )

        try:
            est.fit(Y=Y_tr, T=T_tr, X=X_tr, W=W_tr)

            fold_score = est.score(Y_val, T_val, X=X_val, W=W_val)
            scores.append(fold_score)
            print(f"    Fold {fold}: score = {fold_score:.6f}  (R-loss ≈ {-fold_score:.6f})")

        except Exception as e:
            scores.append(np.nan)

    scores_clean = [s for s in scores if not np.isnan(s)]
    if len(scores_clean) == 0:
        mean_score = np.nan
        std_score = np.nan
    else:
        mean_score = np.mean(scores_clean)
        std_score = np.std(scores_clean)

    r_loss_mean = -mean_score if not np.isnan(mean_score) else np.nan
    r_loss_std = std_score if not np.isnan(std_score) else np.nan

    results.append({
        "name": config["name"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "score_mean": mean_score,
        "score_std": std_score,
        "r_loss_mean": r_loss_mean,
        "r_loss_std": r_loss_std,
        "valid_folds": len(scores_clean)
    })

results_df = pd.DataFrame(results).sort_values("r_loss_mean").reset_index(drop=True)

print(results_df[['name', 'n_estimators', 'max_depth', 'r_loss_mean', 'r_loss_std', 'valid_folds']].to_string(index=False, float_format="%.6f"))

best = results_df.iloc[0]


#%%

file_path = 'D:/visceral_final.csv'
data_all_visceral = pd.read_csv(file_path, encoding='latin1')

data_all_visceral['excess'] = (data_all_visceral['sir'] > 1).astype(int)

data_all_visceral = data_all_visceral.dropna()

columnas_to_drop = ['population', 'min_masl', 'cases', 'sir']

# 1. Label Encoding DANE
le = LabelEncoder()
data_all_visceral['DANE_labeled'] = le.fit_transform(data_all_visceral['DANE'])
scaler = MinMaxScaler()
data_all_visceral['DANE_normalized'] = scaler.fit_transform(
    data_all_visceral[['DANE_labeled']]
)

# 2. Label Encoding Deparment_DANE
le_year = LabelEncoder()
data_all_visceral['DANE_year_labeled'] = le_year.fit_transform(data_all_visceral['DANEYear'])
scaler_DDANE = MinMaxScaler()
data_all_visceral['DANE_year_normalized'] = scaler_DDANE.fit_transform(
    data_all_visceral[['DANE_year_labeled']]
)


data_all_visceral.drop(columns=columnas_to_drop, inplace=True)

std_HFP = data_all_visceral['HFP'].std()
print(f"std of HFP: {std_HFP}")

median_HFP = data_all_visceral['HFP'].median()
print(f"median of HFP: {median_HFP}")

scaler = StandardScaler()
data_all_visceral['temperature'] = scaler.fit_transform(data_all_visceral[['temperature']])
data_all_visceral['rainfall'] = scaler.fit_transform(data_all_visceral[['rainfall']])
data_all_visceral['Forest'] = scaler.fit_transform(data_all_visceral[['Forest']])
data_all_visceral['Deforest'] = scaler.fit_transform(data_all_visceral[['Deforest']])
data_all_visceral['Fire'] = scaler.fit_transform(data_all_visceral[['Fire']])
data_all_visceral['Mining'] = scaler.fit_transform(data_all_visceral[['Mining']])
data_all_visceral['Coca'] = scaler.fit_transform(data_all_visceral[['Coca']])
data_all_visceral['House'] = scaler.fit_transform(data_all_visceral[['House']])
data_all_visceral['Services'] = scaler.fit_transform(data_all_visceral[['Services']])
data_all_visceral['Overcrowding'] = scaler.fit_transform(data_all_visceral[['Overcrowding']])
data_all_visceral['Urban'] = scaler.fit_transform(data_all_visceral[['Urban']])
data_all_visceral['Ethnic'] = scaler.fit_transform(data_all_visceral[['Ethnic']])
data_all_visceral['rMisery'] = scaler.fit_transform(data_all_visceral[['rMisery']])
data_all_visceral['uMisery'] = scaler.fit_transform(data_all_visceral[['uMisery']])
data_all_visceral['HFP'] = scaler.fit_transform(data_all_visceral[['HFP']])

# std
data_std_visceral = data_all_visceral

data_std_visceral = data_std_visceral.sort_values(
    by=['DANE_normalized', 'DANE_year_normalized']
).reset_index(drop=True)

data_std_visceral = data_std_visceral.dropna()

data_std_visceral = data_std_visceral[['DANE_normalized', 'DANE_year_normalized',
                     'temperature', 'rainfall', 'Forest', 'Deforest', 'Fire', 'Mining', 'Coca',
                     'House', 'Services', 'Overcrowding', 'Urban', 'Ethnic',
                     'rMisery', 'uMisery',
                     'HFP', 'excess']]

model_visceral = CausalModel(
    data=data_std_visceral,
    treatment=['HFP'],
    outcome=['excess'],
    graph="""
    digraph {

        rainfall -> temperature
        rainfall -> Forest
        temperature -> Forest
        rainfall -> HFP
        temperature -> HFP
        rainfall -> excess
        temperature -> excess

        Forest -> HFP
        Forest -> excess
        Deforest -> HFP
        Deforest -> excess
        Coca -> HFP
        Coca -> excess
        Mining -> HFP
        Mining -> excess
        Fire -> HFP
        Fire -> excess
        rMisery -> HFP
        rMisery -> excess

        Forest -> Deforest
        Forest -> Fire
        Forest -> Coca
        Forest -> rMisery

        Deforest -> Fire
        Deforest -> Coca

        Mining -> Fire
        Mining -> Deforest

        Coca -> uMisery

        rMisery -> Deforest
        rMisery -> Fire
        rMisery -> Coca
        rMisery -> Mining
        rMisery -> Services

        Fire -> uMisery
        Deforest -> uMisery
        Mining -> uMisery
        Deforest -> Urban
        Mining -> Urban
        Mining -> Services
        Forest -> Services
        Overcrowding -> Coca

        House -> HFP
        House -> excess
        Ethnic -> HFP
        Ethnic -> excess
        Services -> HFP
        Services -> excess
        Overcrowding -> HFP
        Overcrowding -> excess
        Urban -> HFP
        Urban -> excess
        uMisery -> HFP
        uMisery -> excess

        House -> Services
        House -> Overcrowding
        House -> Ethnic
        House -> uMisery

        Ethnic -> Overcrowding
        Ethnic -> Forest
        Ethnic -> Services
        Ethnic -> Urban
        Ethnic -> uMisery

        Urban -> Services
        Urban -> uMisery

        Overcrowding -> Services
        Overcrowding -> uMisery

        Services -> uMisery

        HFP -> excess

        DANE_normalized -> excess
        DANE_year_normalized -> excess

    }
    """
)

# Identifying effects
identified_estimand_visceral = model_visceral.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_visceral)


SEED = 123
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'  

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


Y = data_std_visceral['excess'].values.astype(int)  
T = data_std_visceral['HFP'].values                 
X = data_std_visceral[['rMisery', 'uMisery', 'rainfall', 'temperature', 'DANE_normalized', 'DANE_year_normalized']].values

W = data_std_visceral[['temperature','rainfall','Forest','Coca','Deforest','Fire','House','Overcrowding','Services','Ethnic','Urban','Mining','uMisery','rMisery']].values

model_configs = [
    {"name": "Model 1", "n_estimators": 20, "max_depth": 5},
    {"name": "Model 2", "n_estimators": 20, "max_depth": 7},
    {"name": "Model 3", "n_estimators": 30, "max_depth": 5},
    {"name": "Model 4", "n_estimators": 30, "max_depth": 8},
    {"name": "Model 5", "n_estimators": 50, "max_depth": 5},
    {"name": "Model 6", "n_estimators": 50, "max_depth": 7},
    {"name": "Model 7", "n_estimators": 75, "max_depth": 5},
    {"name": "Model 8", "n_estimators": 75, "max_depth": 8},
    {"name": "Model 9", "n_estimators": 100, "max_depth": 4},
    {"name": "Model 10", "n_estimators": 100, "max_depth": 7}
]

fixed_params = {
    "eta": 0.0001,
    "reg_lambda": 1.5,
    "alpha": 0.001,
    "random_state": SEED,
    "tree_method": "exact",    
    "nthread": 1,              
    "verbosity": 0           
}


results = []
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for i, config in enumerate(model_configs, 1):
    print(f"  n_estimators={config['n_estimators']}, max_depth={config['max_depth']}")

    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        T_tr, T_val = T[train_idx], T[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]
        W_tr, W_val = W[train_idx], W[val_idx]

        model_y = xgb.XGBClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            objective="binary:logistic",
            **fixed_params
        )

        model_t = xgb.XGBRegressor(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            objective="reg:squarederror",
            **fixed_params
        )

        cv_internal = KFold(n_splits=3, shuffle=True, random_state=SEED)

        est = DML(
            model_y=model_y,
            model_t=model_t,
            model_final= LassoCV(alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1], 
                                 fit_intercept=False,
                                 max_iter=50000,
                                 tol=1e-3,
                                 cv=3,           
                                 n_jobs=-1),
            featurizer=PolynomialFeatures(degree=3, include_bias=False),
            discrete_treatment=False,
            discrete_outcome=True,  
            cv=cv_internal,         
            random_state=SEED,
            fit_cate_intercept=True
        )

        try:
            est.fit(Y=Y_tr, T=T_tr, X=X_tr, W=W_tr)

            fold_score = est.score(Y_val, T_val, X=X_val, W=W_val)
            scores.append(fold_score)
            print(f"    Fold {fold}: score = {fold_score:.6f}  (R-loss ≈ {-fold_score:.6f})")

        except Exception as e:
            scores.append(np.nan)

    scores_clean = [s for s in scores if not np.isnan(s)]
    if len(scores_clean) == 0:
        mean_score = np.nan
        std_score = np.nan
    else:
        mean_score = np.mean(scores_clean)
        std_score = np.std(scores_clean)

    r_loss_mean = -mean_score if not np.isnan(mean_score) else np.nan
    r_loss_std = std_score if not np.isnan(std_score) else np.nan

    results.append({
        "name": config["name"],
        "n_estimators": config["n_estimators"],
        "max_depth": config["max_depth"],
        "score_mean": mean_score,
        "score_std": std_score,
        "r_loss_mean": r_loss_mean,
        "r_loss_std": r_loss_std,
        "valid_folds": len(scores_clean)
    })


results_df = pd.DataFrame(results).sort_values("r_loss_mean").reset_index(drop=True)
print("\n" + "="*70)
print("="*70)
print(results_df[['name', 'n_estimators', 'max_depth', 'r_loss_mean', 'r_loss_std', 'valid_folds']].to_string(index=False, float_format="%.6f"))

