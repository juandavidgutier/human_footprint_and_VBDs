
# Importing required libraries
import os, warnings, random
import dowhy
import econml
from dowhy import CausalModel
import pandas as pd
import numpy as np
from econml.dml import DML
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import scipy.stats as stats
from econml.dml import SparseLinearDML, LinearDML, CausalForestDML
from econml.orf import DMLOrthoForest
from econml.score import RScorer
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from econml.inference import BootstrapInference
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from econml.inference import BootstrapInference
from sklearn.linear_model import Lasso


np.int = np.int32
np.float = np.float64
np.bool = np.bool_

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

#%%
# Dataframe of ATEs
# Set display options for pandas to show 5 decimal places
pd.set_option('display.float_format', lambda x: '{:.5f}'.format(x))

# Create data frame of ATE results
df_ATE = pd.DataFrame(0.0, index=range(0, 3), columns=['ATE', '95% CI']).astype({'ATE': 'float64'})

# Convert the second column to tuples with 5 decimal places
df_ATE['95% CI'] = [((0.0, 0.0)) for _ in range(3)]  

# Display the DataFrame
print(df_ATE)

#%%



##################################################################################################################################
##################################################################################################################################

# Malaria
file_path = 'D:/clases/UDES/artículo huella humana/manuscript/reviewers/otros_nuevos_revisores/ml/malaria_final.csv'
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

mean_HFP = data_all_malaria['HFP'].mean()
print(f"mean of HFP: {mean_HFP}")

std_rMisery = data_all_malaria['rMisery'].std()
print(f"std of rMisery: {std_rMisery}")

mean_rMisery = data_all_malaria['rMisery'].mean()
print(f"mean of rMisery: {mean_rMisery}")

std_uMisery = data_all_malaria['uMisery'].std()
print(f"std of uMisery: {std_uMisery}")

mean_uMisery = data_all_malaria['uMisery'].mean()
print(f"mean of uMisery: {mean_uMisery}")

std_rainfall = data_all_malaria['rainfall'].std()
print(f"std of rainfall: {std_rainfall}")

mean_rainfall = data_all_malaria['rainfall'].mean()
print(f"mean of rainfall: {mean_rainfall}")

std_temperature = data_all_malaria['temperature'].std()
print(f"std of temperature: {std_temperature}")

mean_temperature = data_all_malaria['temperature'].mean()
print(f"mean of temperature: {mean_temperature}")


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

#%%

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

    
#%% 

# Identifying effects
identified_estimand_malaria = model_malaria.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_malaria)


#%%

reg1 = lambda: XGBRegressor(
    n_estimators=20,
    max_depth=5,
    random_state=123,
    learning_rate=0.0001,
    reg_lambda=1.5,
    alpha=0.01,
    tree_method="hist",
    verbosity=0,
    n_jobs=-1
)

effect_modifiers = ['rMisery', 'uMisery', 'rainfall', 'temperature', 'DANE_normalized', 'DANE_year_normalized']

method_params = {
    "init_params": {
        "model_y": reg1(),
        "model_t": reg1(),
        "model_final": LassoCV(
                                alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1],  
                                fit_intercept=False,
                                max_iter=50000,
                                tol=1e-3,
                                cv=3,
                                n_jobs=-1
                            ),

        "featurizer": PolynomialFeatures(degree=3, include_bias=False),
        "discrete_outcome": True,
        "discrete_treatment": False,
        "cv": 3,
        "random_state": 123
    },
    "fit_params": {
        "inference": BootstrapInference(bootstrap_type="normal")
    }
}

causal_estimate_malaria = model_malaria.estimate_effect(
    identified_estimand_malaria,
    method_name="backdoor.econml.dml.DML",
    effect_modifiers=effect_modifiers,
    confidence_intervals=True,
    method_params=method_params
)


estimator = causal_estimate_malaria.estimator.estimator
print("Has inference object?:", getattr(estimator, "_inference", None) is not None)

X_data = data_std_malaria[effect_modifiers].dropna()

ate = estimator.ate(X=X_data)
ate_ci = estimator.ate_interval(X=X_data, alpha=0.05)

print(f"ATE: {ate}")
print(f"95% CI ATE: {ate_ci}")

df_ATE.at[0, 'ATE'] = ate
df_ATE.at[0, '95% CI'] = ate_ci
print(df_ATE)


#%%

X = data_std_malaria[['rMisery', 'uMisery', 'rainfall', 'temperature',
                       'DANE_normalized', 'DANE_year_normalized']].to_numpy()


plt.style.use('ggplot')


def build_grid(X, col_idx, n_points=100):
    col      = X[:, col_idx]
    mn, mx   = col.min(), col.max()
    delta    = (mx - mn) / n_points
    grid_col = np.arange(mn, mx + delta - 1e-9, delta)
    n        = len(grid_col)
    parts    = []
    for j in range(X.shape[1]):
        parts.append(grid_col if j == col_idx
                     else np.full(n, np.mean(X[:, j])))
    return np.column_stack(parts)


def plot_cate(X, col_idx, estimator, x_label, panel_label, filename):
    """
    Parameters
    ----------
    X           : np.ndarray — matriz completa (n x 6): rMisery, uMisery,
                               rainfall, temperature, DANE_norm, DANE_year_norm
    col_idx     : int        — columna de la variable moderadora a graficar
    estimator   : objeto DML extraído como causal_estimate.estimator.estimator
    x_label     : str        — etiqueta eje X
    panel_label : str        — letra del ('a', 'b', 'c', 'd')
    filename    : str        — nombre del archivo PNG de salida
    """
    grid         = build_grid(X, col_idx)

    cate         = estimator.effect(grid).ravel()
    lower, upper = estimator.effect_interval(grid, alpha=0.05)
    lower        = lower.ravel()
    upper        = upper.ravel()
    x_plot       = grid[:, col_idx].ravel()   

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#EBEBEB')
    ax.grid(True, color='white', linewidth=0.8, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    ax.fill_between(x_plot, lower, upper, alpha=0.2, color='steelblue', zorder=2)

    ax.plot(x_plot, cate, color='steelblue', linewidth=1.8, zorder=3)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.0, zorder=4)

    ax.set_xlabel(x_label, fontsize=12, labelpad=6)
    ax.set_ylabel('Effect of HFP on excess malaria cases', fontsize=12, labelpad=6)
    ax.set_title(panel_label, fontsize=14, loc='center', pad=8)
    ax.tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# ── a — Rural misery   (col 0) ────────────────────────────────────────
plot_cate(X, col_idx=0, estimator=estimator,
          x_label='Rural misery (SD)',
          panel_label='a',
          filename='cate_rmisery_malaria.png')

# ── b — Urban misery   (col 1) ────────────────────────────────────────
plot_cate(X, col_idx=1, estimator=estimator,
          x_label='Urban misery (SD)',
          panel_label='b',
          filename='cate_umisery_malaria.png')

# ── c — Rainfall       (col 2) ────────────────────────────────────────
plot_cate(X, col_idx=2, estimator=estimator,
          x_label='Rainfall (SD)',
          panel_label='c',
          filename='cate_rainfall_malaria.png')

# ── d — Temperature    (col 3) ────────────────────────────────────────
plot_cate(X, col_idx=3, estimator=estimator,
          x_label='Temperature (SD)',
          panel_label='d',
          filename='cate_temperature_malaria.png')


#%%
#with random
random_malaira = model_malaria.refute_estimate(identified_estimand_malaria, causal_estimate_malaria,
                                         method_name="random_common_cause", random_state=123, num_simulations=50)
print(random_malaira)

#with subset
subset_malaira  = model_malaria.refute_estimate(identified_estimand_malaria, causal_estimate_malaria,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=50)
print(subset_malaira)

#with bootstrap
bootstrap_malaira  = model_malaria.refute_estimate(identified_estimand_malaria, causal_estimate_malaria,
                                             method_name="bootstrap_refuter", random_state=123, num_simulations=50)
print(bootstrap_malaira)

#with placebo
placebo_malaira  = model_malaria.refute_estimate(identified_estimand_malaria, causal_estimate_malaria,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=50)
print(placebo_malaira)



#%%

# non-parametric partial R²

X = data_std_malaria[['rainfall','temperature','Deforest','Fire','rMisery','uMisery','Mining','Coca','Forest']]
T = data_std_malaria["HFP"]
Y = data_std_malaria["excess"]

mi_T = mutual_info_regression(X, T,random_state=123)
mi_Y = mutual_info_classif(X, Y,random_state=123)

score = mi_T * mi_Y
ranking = pd.Series(score, index=X.columns).sort_values(ascending=False)
print(ranking.head(10)) 

# non-parametric partial R2
partialR2_malaria = model_malaria.refute_estimate(
    identified_estimand_malaria,
    causal_estimate_malaria,
    method_name="add_unobserved_common_cause",
    simulation_method="non-parametric-partial-R2",
    benchmark_common_causes=["rMisery"],
    effect_fraction_on_treatment=0.1,
    effect_fraction_on_outcome=0.1,
    plugin_reisz=True,
    num_simulations=500,
    plot_estimate=False
)

print(partialR2_malaria)
print(partialR2_malaria.RV)
print(partialR2_malaria.RV_alpha)


# ===============================
# PARTIAL R2 BENCHMARK rMisery
# ===============================

covariates = [
'rainfall','temperature','Deforest','Fire','uMisery','Mining','Coca','Forest', #excluir el confusor del benchmark
]

X = data_std_malaria[covariates]
T = data_std_malaria["HFP"]
Y = data_std_malaria["excess"]
Z = data_std_malaria["rMisery"]


from sklearn.model_selection import KFold


kf = KFold(n_splits=5, shuffle=True, random_state=123)

T_res = np.zeros(len(T))
Y_res = np.zeros(len(Y))
Z_res = np.zeros(len(Z))

for train, test in kf.split(X):

    mt = reg1()
    my = reg1()
    mz = reg1()

    mt.fit(X.iloc[train], T.iloc[train])
    my.fit(X.iloc[train], Y.iloc[train])
    mz.fit(X.iloc[train], Z.iloc[train])

    T_res[test] = T.iloc[test] - mt.predict(X.iloc[test])
    Y_res[test] = Y.iloc[test] - my.predict(X.iloc[test])
    Z_res[test] = Z.iloc[test] - mz.predict(X.iloc[test])


# partial R²
r2_z_t = np.corrcoef(Z_res, T_res)[0,1]**2
r2_z_y = np.corrcoef(Z_res, Y_res)[0,1]**2

print("Partial R² rMisery→T | X:", r2_z_t)
print("Partial R² rMisery→Y | X:", r2_z_y)

# ==========================
# STRENGTH MULTIPLIER
# ==========================
RV_point   = partialR2_malaria.RV
RV_alpha   = partialR2_malaria.RV_alpha          
r2_bench_T = r2_z_t
r2_bench_Y = r2_z_y

# ─────────────────────────────────────────────
# RV_alpha
# ─────────────────────────────────────────────
k_T_alpha = RV_alpha / r2_bench_T if r2_bench_T > 0 else np.inf
k_Y_alpha = RV_alpha / r2_bench_Y if r2_bench_Y > 0 else np.inf
k_binding_alpha = max(k_T_alpha, k_Y_alpha)


#%%
##################################################################################################################################
##################################################################################################################################

# Dengue
file_path = 'D:/clases/UDES/artículo huella humana/manuscript/reviewers/otros_nuevos_revisores/ml/dengue_final.csv'
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

mean_HFP = data_all_dengue['HFP'].mean()
print(f"mean of HFP: {mean_HFP}")

std_rMisery = data_all_dengue['rMisery'].std()
print(f"std of rMisery: {std_rMisery}")

mean_rMisery = data_all_dengue['rMisery'].mean()
print(f"mean of rMisery: {mean_rMisery}")

std_uMisery = data_all_dengue['uMisery'].std()
print(f"std of uMisery: {std_uMisery}")

mean_uMisery = data_all_dengue['uMisery'].mean()
print(f"mean of uMisery: {mean_uMisery}")

std_rainfall = data_all_dengue['rainfall'].std()
print(f"std of rainfall: {std_rainfall}")

mean_rainfall = data_all_dengue['rainfall'].mean()
print(f"mean of rainfall: {mean_rainfall}")

std_temperature = data_all_dengue['temperature'].std()
print(f"std of temperature: {std_temperature}")

mean_temperature = data_all_dengue['temperature'].mean()
print(f"mean of temperature: {mean_temperature}")


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

data_std_dengue[['DANE_normalized', 'DANE_year_normalized',
                     'temperature', 'rainfall', 'House', 'Services', 'Overcrowding', 'Urban', 'Ethnic', 'rMisery', 'uMisery',
                     'HFP', 'excess']]


#%%

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

    
#%% 

# Identifying effects
identified_estimand_dengue = model_dengue.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_dengue)


#%%

from econml.inference import BootstrapInference
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor


reg1 = lambda: XGBRegressor(
    n_estimators=20,
    max_depth=5,
    random_state=123,
    learning_rate=0.0001,
    reg_lambda=1.5,
    alpha=0.01,
    tree_method="hist",
    verbosity=0,
    n_jobs=-1
)

effect_modifiers = ['rMisery', 'uMisery', 'rainfall', 'temperature', 'DANE_normalized', 'DANE_year_normalized']

method_params = {
    "init_params": {
        "model_y": reg1(),
        "model_t": reg1(),
        "model_final": LassoCV(
                                alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1],  
                                fit_intercept=False,
                                max_iter=50000,
                                tol=1e-3,
                                cv=3,         
                                n_jobs=-1
                            ),

        "featurizer": PolynomialFeatures(degree=3, include_bias=False),
        "discrete_outcome": True,
        "discrete_treatment": False,
        "cv": 3,
        "random_state": 123
    },
    "fit_params": {
        "inference": BootstrapInference(bootstrap_type="normal")
    }
}

causal_estimate_dengue = model_dengue.estimate_effect(
    identified_estimand_dengue,
    method_name="backdoor.econml.dml.DML",
    effect_modifiers=effect_modifiers,
    confidence_intervals=True,
    method_params=method_params
)


estimator = causal_estimate_dengue.estimator.estimator
print("Has inference object?:", getattr(estimator, "_inference", None) is not None)

X_data = data_std_dengue[effect_modifiers].dropna()

ate = estimator.ate(X=X_data)
ate_ci = estimator.ate_interval(X=X_data, alpha=0.05)

print(f"ATE: {ate}")
print(f"95% CI ATE: {ate_ci}")

df_ATE.at[1, 'ATE'] = ate
df_ATE.at[1, '95% CI'] = ate_ci
print(df_ATE)


#%%

X = data_std_dengue[['rMisery', 'uMisery', 'rainfall', 'temperature',
                      'DANE_normalized', 'DANE_year_normalized']].to_numpy()

plt.style.use('ggplot')

def build_grid(X, col_idx, n_points=100):
    col      = X[:, col_idx]
    mn, mx   = col.min(), col.max()
    delta    = (mx - mn) / n_points
    grid_col = np.arange(mn, mx + delta - 1e-9, delta)
    n        = len(grid_col)
    parts    = []
    for j in range(X.shape[1]):
        parts.append(grid_col if j == col_idx
                     else np.full(n, np.mean(X[:, j])))
    return np.column_stack(parts)


def plot_cate(X, col_idx, estimator, x_label, panel_label, filename,
              y_label='Effect of HFP on excess dengue cases'):

    grid         = build_grid(X, col_idx)
    cate         = estimator.effect(grid).ravel()
    lower, upper = estimator.effect_interval(grid, alpha=0.05)
    lower        = lower.ravel()
    upper        = upper.ravel()
    x_plot       = grid[:, col_idx].ravel()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#EBEBEB')
    ax.grid(True, color='white', linewidth=0.8, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    ax.fill_between(x_plot, lower, upper,
                    alpha=0.2, color='steelblue', zorder=2)
    ax.plot(x_plot, cate,
            color='steelblue', linewidth=1.8, zorder=3)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.0, zorder=4)

    y_range = max(np.max(np.abs(upper)), np.max(np.abs(lower)),
                  np.max(np.abs(cate)))

    if y_range < 1e-10:
        ax.set_ylim(-0.05, 0.05)
        print(f"[{panel_label}] AVISO: CATE de '{x_label}' ≈ 0 "
              f"(y_range = {y_range:.2e}). "
              f"Variable no es modificador del efecto.")
    else:
        y_pad = (np.max(upper) - np.min(lower)) * 0.10
        ax.set_ylim(np.min(lower) - y_pad, np.max(upper) + y_pad)

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'{val:.4f}')
    )

    ax.set_xlabel(x_label,    fontsize=12, labelpad=6)
    ax.set_ylabel(y_label,    fontsize=12, labelpad=6)
    ax.set_title(panel_label, fontsize=14, loc='center', pad=8)
    ax.tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# ── a — Rural misery   (col 0) ────────────────────────────────────────
plot_cate(X, col_idx=0, estimator=estimator,
          x_label='Rural misery (SD)',
          panel_label='a',
          filename='cate_rmisery_dengue.png')

# ── b — Urban misery   (col 1) ────────────────────────────────────────
plot_cate(X, col_idx=1, estimator=estimator,
          x_label='Urban misery (SD)',
          panel_label='b',
          filename='cate_umisery_dengue.png')

# ── c — Rainfall       (col 2) ────────────────────────────────────────
plot_cate(X, col_idx=2, estimator=estimator,
          x_label='Rainfall (SD)',
          panel_label='c',
          filename='cate_rainfall_dengue.png')

# ── d — Temperature    (col 3) ────────────────────────────────────────
plot_cate(X, col_idx=3, estimator=estimator,
          x_label='Temperature (SD)',
          panel_label='d',
          filename='cate_temperature_dengue.png')




#%%
#with random
random_dengue = model_dengue.refute_estimate(identified_estimand_dengue, causal_estimate_dengue,
                                         method_name="random_common_cause", random_state=123, num_simulations=50)
print(random_dengue)

#with subset
subset_dengue  = model_dengue.refute_estimate(identified_estimand_dengue, causal_estimate_dengue,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=50)
print(subset_dengue)

#with bootstrap
bootstrap_dengue  = model_dengue.refute_estimate(identified_estimand_dengue, causal_estimate_dengue,
                                             method_name="bootstrap_refuter", random_state=123, num_simulations=50)
print(bootstrap_dengue)

#with placebo
placebo_dengue  = model_dengue.refute_estimate(identified_estimand_dengue, causal_estimate_dengue,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=50)
print(placebo_dengue)


#%%

##################################################################################################################################
##################################################################################################################################

# Visceral
file_path = 'D:/clases/UDES/artículo huella humana/manuscript/reviewers/otros_nuevos_revisores/ml/visceral_final.csv'
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

mean_HFP = data_all_visceral['HFP'].mean()
print(f"mean of HFP: {mean_HFP}")

std_rMisery = data_all_visceral['rMisery'].std()
print(f"std of rMisery: {std_rMisery}")

mean_rMisery = data_all_visceral['rMisery'].mean()
print(f"mean of rMisery: {mean_rMisery}")

std_uMisery = data_all_visceral['uMisery'].std()
print(f"std of uMisery: {std_uMisery}")

mean_uMisery = data_all_visceral['uMisery'].mean()
print(f"mean of uMisery: {mean_uMisery}")

std_rainfall = data_all_visceral['rainfall'].std()
print(f"std of rainfall: {std_rainfall}")

mean_rainfall = data_all_visceral['rainfall'].mean()
print(f"mean of rainfall: {mean_rainfall}")

std_temperature = data_all_visceral['temperature'].std()
print(f"std of temperature: {std_temperature}")

mean_temperature = data_all_visceral['temperature'].mean()
print(f"mean of temperature: {mean_temperature}")


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

#%%

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

    
#%% 

# Identifying effects
identified_estimand_visceral = model_visceral.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_visceral)


#%%

reg1 = lambda: XGBRegressor(
    n_estimators=20,
    max_depth=7,
    random_state=123,
    learning_rate=0.0001,
    reg_lambda=1.5,
    alpha=0.01,
    tree_method="hist",
    verbosity=0,
    n_jobs=-1
)

effect_modifiers = ['rMisery', 'uMisery', 'rainfall', 'temperature', 'DANE_normalized', 'DANE_year_normalized']

method_params = {
    "init_params": {
        "model_y": reg1(),
        "model_t": reg1(),
        "model_final": LassoCV(
                                alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1], 
                                fit_intercept=False,
                                max_iter=50000,
                                tol=1e-3,
                                cv=3,           
                                n_jobs=-1
                            ),

        "featurizer": PolynomialFeatures(degree=3, include_bias=False),
        "discrete_outcome": True,
        "discrete_treatment": False,
        "cv": 3,
        "random_state": 123
    },
    "fit_params": {
        "inference": BootstrapInference(bootstrap_type="normal")
    }
}

causal_estimate_visceral = model_visceral.estimate_effect(
    identified_estimand_visceral,
    method_name="backdoor.econml.dml.DML",
    effect_modifiers=effect_modifiers,
    confidence_intervals=True,
    method_params=method_params
)


estimator = causal_estimate_visceral.estimator.estimator
print("Has inference object?:", getattr(estimator, "_inference", None) is not None)

X_data = data_std_visceral[effect_modifiers].dropna()

ate = estimator.ate(X=X_data)
ate_ci = estimator.ate_interval(X=X_data, alpha=0.05)

print(f"ATE: {ate}")
print(f"95% CI ATE: {ate_ci}")

df_ATE.at[2, 'ATE'] = ate
df_ATE.at[2, '95% CI'] = ate_ci
print(df_ATE)


#%%

X = data_std_visceral[['rMisery', 'uMisery', 'rainfall', 'temperature',
                       'DANE_normalized', 'DANE_year_normalized']].to_numpy()


plt.style.use('ggplot')

def build_grid(X, col_idx, n_points=100):
    col      = X[:, col_idx]
    mn, mx   = col.min(), col.max()
    delta    = (mx - mn) / n_points
    grid_col = np.arange(mn, mx + delta - 1e-9, delta)
    n        = len(grid_col)
    parts    = []
    for j in range(X.shape[1]):
        parts.append(grid_col if j == col_idx
                     else np.full(n, np.mean(X[:, j])))
    return np.column_stack(parts)


def plot_cate(X, col_idx, estimator, x_label, panel_label, filename,
              y_label='Effect of HFP on excess visceral leishmaniasis cases'):

    grid         = build_grid(X, col_idx)
    cate         = estimator.effect(grid).ravel()
    lower, upper = estimator.effect_interval(grid, alpha=0.05)
    lower        = lower.ravel()
    upper        = upper.ravel()
    x_plot       = grid[:, col_idx].ravel()

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#EBEBEB')
    ax.grid(True, color='white', linewidth=0.8, linestyle='-', zorder=0)
    ax.set_axisbelow(True)

    ax.fill_between(x_plot, lower, upper,
                    alpha=0.2, color='steelblue', zorder=2)
    ax.plot(x_plot, cate,
            color='steelblue', linewidth=1.8, zorder=3)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.0, zorder=4)

    y_range = max(np.max(np.abs(upper)), np.max(np.abs(lower)),
                  np.max(np.abs(cate)))

    if y_range < 1e-10:
        ax.set_ylim(-0.05, 0.05)
        print(f"[{panel_label}] AVISO: CATE de '{x_label}' ≈ 0 "
              f"(y_range = {y_range:.2e}). "
              f"Variable no es modificador del efecto.")
    else:
        y_pad = (np.max(upper) - np.min(lower)) * 0.10
        ax.set_ylim(np.min(lower) - y_pad, np.max(upper) + y_pad)

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f'{val:.4f}')
    )

    ax.set_xlabel(x_label,    fontsize=12, labelpad=6)
    ax.set_ylabel(y_label,    fontsize=12, labelpad=6)
    ax.set_title(panel_label, fontsize=14, loc='center', pad=8)
    ax.tick_params(axis='both', labelsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# ── a — Rural misery   (col 0) ────────────────────────────────────────
plot_cate(X, col_idx=0, estimator=estimator,
          x_label='Rural misery (SD)',
          panel_label='a',
          filename='cate_rmisery_visceral leishmaniasis.png')

# ── b — Urban misery   (col 1) ────────────────────────────────────────
plot_cate(X, col_idx=1, estimator=estimator,
          x_label='Urban misery (SD)',
          panel_label='b',
          filename='cate_umisery_visceral leishmaniasis.png')

# ── c — Rainfall       (col 2) ────────────────────────────────────────
plot_cate(X, col_idx=2, estimator=estimator,
          x_label='Rainfall (SD)',
          panel_label='c',
          filename='cate_rainfall_visceral leishmaniasis.png')

# ── d — Temperature    (col 3) ────────────────────────────────────────
plot_cate(X, col_idx=3, estimator=estimator,
          x_label='Temperature (SD)',
          panel_label='d',
          filename='cate_temperature_visceral leishmaniasis.png')



#%%
#with random
random_visceral = model_visceral.refute_estimate(identified_estimand_visceral, causal_estimate_visceral,
                                         method_name="random_common_cause", random_state=123, num_simulations=50)
print(random_visceral)

#with subset
subset_visceral  = model_visceral.refute_estimate(identified_estimand_visceral, causal_estimate_visceral,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=50)
print(subset_visceral)

#with bootstrap
bootstrap_visceral  = model_visceral.refute_estimate(identified_estimand_visceral, causal_estimate_visceral,
                                             method_name="bootstrap_refuter", random_state=123, num_simulations=50)
print(bootstrap_visceral)

#with placebo
placebo_visceral  = model_visceral.refute_estimate(identified_estimand_visceral, causal_estimate_visceral,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=50)
print(placebo_visceral)



#%%

#Figure ATE

# labels
labs = ['Malaria',
        'Dengue',
        'Visceral leishmaniasis']

df_labs = pd.DataFrame({'Labels': labs})

# ATE
ATE = df_ATE['ATE'].astype(float)

df_ci = df_ATE['95% CI'].apply(lambda x: (x[0][0], x[1][0])).apply(pd.Series)
df_ci.columns = ['Lower', 'Upper']

Lower = df_ci['Lower']
Upper = df_ci['Upper']

df_plot = pd.concat([df_labs, ATE, Lower, Upper], axis=1)

labels = df_plot['Labels'].values
ate = df_plot['ATE'].values
lower = df_plot['Lower'].values
upper = df_plot['Upper'].values

y_pos = np.arange(len(labels))
xerr = [ate - lower, upper - ate]

plt.figure(figsize=(12,7))

plt.errorbar(
    ate,
    y_pos,
    xerr=xerr,
    fmt='s',
    capsize=4,
    color="blue"
)

plt.axvline(x=0, linestyle='--', color="red")

plt.yticks(y_pos, labels, fontsize=14)
plt.xticks(fontsize=14)

plt.xlabel("ATE", fontsize=18)
plt.title("", fontsize=20)

plt.xlim(-0.1, 0.1)

plt.tight_layout()
plt.show()

