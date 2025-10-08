
# -*- coding: utf-8 -*-
"""

@author: juand
"""


###############################################################################
# Code for the paper:
#   "Anthropogenic Landscapes and Vector-Borne Disease Dynamics: Unveiling the Comple
#    Interplay between Human Footprint and Disease Transmission in Colombia" Juan D. Gutiérrez, Wendy L. Quintero-García, Yanyu Xiao, F. DeWolfe Miller, Diego F. Cuadros
#    Last modification = 30/10/2024
#    e-mail = juandavidgutier@gmail.com, jdgutierrez@udes.edu.co
###############################################################################

# Before running the code, I suggest creating a virtual environment and installing the modules econml and dowhy, as follows:
# pip install econml==0.15.0
# pin install dowhy==0.11.1
# you need to install also the modules: pandas, zepid, xgboost, plotnine and matplotlib

# importing required libraries
import os, warnings, random
import dowhy
import econml
import pandas as pd
import numpy as np
import econml
from econml.dml import DML
from sklearn.preprocessing import PolynomialFeatures
from zepid.graphics import EffectMeasurePlot
import numpy as np, scipy.stats as st
import scipy.stats as stats
from zepid.causal.causalgraph import DirectedAcyclicGraph
from zepid.graphics import EffectMeasurePlot
import numpy as np, scipy.stats as st
from sklearn.linear_model import LassoCV
from econml.dml import SparseLinearDML
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text
from xgboost import XGBRegressor
from econml.dml import DML, SparseLinearDML
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dowhy import CausalModel




# Set seeds to make the results more reproducible
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.2f' % x)


#%%#
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

######################################################################################################################################
#malaria
data = pd.read_csv("D:/clases/UDES/artículo huella humana/ml/malaria_final.csv", encoding='latin-1') # Modify the path if necessary
data['excess'] = (data['sir'] > 1).astype(int)
data['rainfall'] = data['rainfall'] *1000
data['temperature'] = data['temperature'] -273.15

data_malaria = data[[
                 'excess', 'HFP',
                 'Coca', 'Forest', 'Mining', 'Fire', 
                 'Deforest', 'uMisery', 'rMisery', 
                 'rainfall', 'temperature'
                 ]]

variables_normalizaded = ['HFP', 'rMisery', 'uMisery', 'rainfall', 'temperature']

for var in variables_normalizaded:
    std = data_malaria[var].std()
    mean = data_malaria[var].mean()
    median = data_malaria[var].median()
    print(f"{var}: std = {std:.6f}, mean = {mean:.6f}, median = {median:.6f}")

# HFP with z-score
scaler = StandardScaler()
data_malaria['HFP'] = scaler.fit_transform(data_malaria[['HFP']])
data_malaria['rMisery'] = scaler.fit_transform(data_malaria[['rMisery']])
data_malaria['uMisery'] = scaler.fit_transform(data_malaria[['uMisery']])
data_malaria['rainfall'] = scaler.fit_transform(data_malaria[['rainfall']])
data_malaria['temperature'] = scaler.fit_transform(data_malaria[['temperature']])

data_malaria = data_malaria.dropna()

#%%

# Convert columns to binary
columns_convert = ['Coca', 'Forest', 'Mining', 'Fire', 'Deforest']
for col in columns_convert:
    median = data_malaria[col].median()
    data_malaria[col] = (data_malaria[col] > median).astype(int)

#%%
Y = data_malaria['excess'].to_numpy() 
T = data_malaria['HFP'].to_numpy()
W = data_malaria[['Coca', 'Forest', 'Mining', 'Fire', 'Deforest', 'rMisery', 'uMisery', 'rainfall', 'temperature']].to_numpy()
X = data_malaria[['rMisery', 'uMisery', 'rainfall', 'temperature']].to_numpy()

# Split data
X_train, X_test, T_train, T_test, Y_train, Y_test, W_train, W_test = train_test_split(
            X, T, Y, W, test_size=0.2, random_state=123)


#%%
## Ignore warnings
warnings.filterwarnings('ignore') 

reg1 = lambda: XGBRegressor(n_estimators=2500, random_state=123, eta=0.0001, max_depth=10, reg_lambda=1.5, alpha=0.01)


#Estimation of ATE
estimate_malaria = SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(),
                                   discrete_treatment=False, cv=5, random_state=123, max_iter=30000)

# fit the model
estimate_malaria.fit(Y=Y, T=T, X=X, W=W, inference='auto')

# predict effect for each sample X
estimate_malaria.effect(X)

# ate
ate_malaria = estimate_malaria.ate(X) 
print(ate_malaria)

# confidence interval of ate
ci_malaria = estimate_malaria.ate_interval(X) 
print(ci_malaria)

# Set values in the df_ATE
df_ATE.at[0, 'ATE'] = round(ate_malaria, 5)
df_ATE.at[0, '95% CI'] = ci_malaria
print(df_ATE)


#%%
#CATE
#range of X for rural misery
# Find the maximum and minimum values of rural misery
min_X0 = np.min(X[:, 0]) 
max_X0 = np.max(X[:, 0])
delta = (max_X0 - min_X0) / 100
X0_grid = np.arange(min_X0, max_X0 + delta - 0.001, delta)

# Means of other variables in X
X1_mean = np.mean(X[:, 1])   
X2_mean = np.mean(X[:, 2])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X0_grid = np.column_stack([
    X0_grid,  
    np.full_like(X0_grid, X1_mean),     
    np.full_like(X0_grid, X2_mean),
    np.full_like(X0_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_malaria.effect(X0_grid)
hte_lower2_cons, hte_upper2_cons = estimate_malaria.effect_interval(X0_grid)

# Reshape para plotting
X0_grid_plot = X0_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X0_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE rural misery- 
cate_rmisery= (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Rural misery (sd)', y='Effect of HFP on excess malaria cases',
           title='a')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_rmisery)

#range of X for urban misery
# Find the maximum and minimum values of rural misery
min_X1 = np.min(X[:, 1]) 
max_X1 = np.max(X[:, 1])
delta = (max_X1 - min_X1) / 100
X1_grid = np.arange(min_X1, max_X1 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X2_mean = np.mean(X[:, 2])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X1_grid = np.column_stack([
    X1_grid,  
    np.full_like(X1_grid, X0_mean),     
    np.full_like(X1_grid, X2_mean),
    np.full_like(X1_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_malaria.effect(X1_grid)
hte_lower2_cons, hte_upper2_cons = estimate_malaria.effect_interval(X1_grid)

# Reshape para plotting
X1_grid_plot = X1_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X1_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE urban misery- 
cate_umisery = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Urban misery (sd)', y='Effect of HFP on excess malaria cases',
           title='b')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_umisery)

#range of X for rainfall
# Find the maximum and minimum values of rural misery
min_X2 = np.min(X[:, 2]) 
max_X2 = np.max(X[:, 2])
delta = (max_X0 - min_X0) / 100
X2_grid = np.arange(min_X2, max_X2 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X1_mean = np.mean(X[:, 1])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X2_grid = np.column_stack([
    X2_grid,  
    np.full_like(X2_grid, X0_mean),     
    np.full_like(X2_grid, X1_mean),
    np.full_like(X2_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_malaria.effect(X2_grid)
hte_lower2_cons, hte_upper2_cons = estimate_malaria.effect_interval(X2_grid)

# Reshape para plotting
X2_grid_plot = X2_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X2_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE Rainfall- 
cate_rainfall = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Rainfall (sd)', y='Effect of HFP on excess malaria cases',
           title='c')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_rainfall)

#range of X for Temperature
# Find the maximum and minimum values of rural misery
min_X3 = np.min(X[:, 3]) 
max_X3 = np.max(X[:, 3])
delta = (max_X0 - min_X0) / 100
X3_grid = np.arange(min_X3, max_X3 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X1_mean = np.mean(X[:, 1])
X2_mean = np.mean(X[:, 2])    

# Matrix of X
X3_grid = np.column_stack([
    X3_grid,  
    np.full_like(X3_grid, X0_mean),     
    np.full_like(X3_grid, X1_mean),
    np.full_like(X3_grid, X2_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_malaria.effect(X3_grid)
hte_lower2_cons, hte_upper2_cons = estimate_malaria.effect_interval(X3_grid)

# Reshape para plotting
X3_grid_plot = X3_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X3_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE Temperature- 
cate_temperature = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Temperature (sd)', y='Effect of HFP on excess malaria cases',
           title='d')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_temperature)


#%%

# HFP as binary
median_HFP = data_malaria['HFP'].median()
data_malaria['HFP'] = (data_malaria['HFP'] > median_HFP).astype(int)


#Causal mechanism
model_malaria = CausalModel(
        data = data_malaria,
        treatment=['HFP'],
        outcome=['excess'],
        graph= """graph[directed 1 
                    node[id "HFP" label "HFP"]
                    node[id "excess" label "excess"]
                    node[id "Forest" label "Forest"]
                    node[id "Deforest" label "Deforest"]
                    node[id "Coca" label "Coca"]
                    node[id "Mining" label "Mining"]
                    node[id "Fire" label "Fire"]
                    node[id "rMisery" label "rMisery"]
                    node[id "uMisery" label "uMisery"]
                    node[id "rainfall" label "rainfall"]
                    node[id "temperature" label "temperature"]
                    

                    edge[source "rainfall" target "temperature"]
                    edge[source "rainfall" target "Forest"]
                    edge[source "temperature" target "Forest"]
                    edge[source "rainfall" target "HFP"]
                    edge[source "temperature" target "HFP"]
                    edge[source "rainfall" target "excess"]
                    edge[source "temperature" target "excess"]
                    

                    edge[source "Forest" target "HFP"]    
                    edge[source "Forest" target "excess"]


                    edge[source "Forest" target "Deforest"]
                    edge[source "rMisery" target "Deforest"]
                    edge[source "uMisery" target "Deforest"]
                    edge[source "Deforest" target "HFP"]
                    edge[source "Deforest" target "excess"]
                    

                    edge[source "Forest" target "Coca"]
                    edge[source "rMisery" target "Coca"]
                    edge[source "uMisery" target "Coca"]
                    edge[source "Coca" target "HFP"]
                    edge[source "Coca" target "excess"]


                    edge[source "Forest" target "Mining"]
                    edge[source "rMisery" target "Mining"]
                    edge[source "uMisery" target "Mining"]
                    edge[source "Mining" target "HFP"]
                    edge[source "Mining" target "excess"]
                    

                    edge[source "Forest" target "Fire"]
                    edge[source "Deforest" target "Fire"]
                    edge[source "Coca" target "Fire"]
                    edge[source "rMisery" target "Fire"]
                    edge[source "uMisery" target "Fire"]
                    edge[source "Fire" target "HFP"]
                    edge[source "Fire" target "excess"]
    

                    edge[source "rMisery" target "HFP"]
                    edge[source "rMisery" target "excess"]
                    edge[source "uMisery" target "HFP"]
                    edge[source "uMisery" target "excess"]


                    edge[source "HFP" target "excess"]
                    
                    ]"""
                    )
    
    
#%% 

# Identifying effects
identified_estimand_malaria = model_malaria.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_malaria)

#%%

# Model with DoWhy
estimate_malaria = model_malaria.estimate_effect(
    identified_estimand_malaria,
    effect_modifiers=['rMisery', 'uMisery', 'rainfall', 'temperature'],
    method_name="backdoor.econml.dml.SparseLinearDML",
    confidence_intervals=True,
    method_params={
        "init_params": {
            "featurizer": PolynomialFeatures(degree=3, include_bias=False),
            "model_y":reg1(),
            "model_t":reg1(),
            "discrete_treatment":False,
            "max_iter": 30000,
            "cv": 5,
            "random_state": 123
            },
        }
)

# ATE with DoWhy
ate_malaria_DoWhy = estimate_malaria.value
print(ate_malaria_DoWhy)
   
#%%
# Refutations

random_malaria = model_malaria.refute_estimate(
    identified_estimand_malaria,
    estimate_malaria,
    method_name="random_common_cause",
    random_state=123,
    num_simulations=50,
    )
print(random_malaria)

subset_malaria = model_malaria.refute_estimate(
    identified_estimand_malaria,
    estimate_malaria,
    subset_fraction=0.1,
    method_name="data_subset_refuter",
    random_state=123,
    num_simulations=50,
    )
print(subset_malaria)

dummy_malaria_results = model_malaria.refute_estimate(
        identified_estimand_malaria,
        estimate_malaria,
        method_name="dummy_outcome_refuter",
        random_state=123,
        num_simulations=50
    )
print(dummy_malaria_results[0])


placebo_malaria = model_malaria.refute_estimate(
    identified_estimand_malaria,
    estimate_malaria,
    method_name="placebo_treatment_refuter",
    random_state=123,
    num_simulations=50,
    )
print(placebo_malaria)




#%%




######################################################################################################################################
#_dengue
#import data
data = pd.read_csv("D:/clases/UDES/artículo huella humana/ml/dengue_final.csv", encoding='latin-1') # Modify the path if necessary

data['excess'] = (data['sir'] > 1).astype(int)
data['rainfall'] = data['rainfall'] *1000
data['temperature'] = data['temperature'] -273.15

data_dengue = data[[
                 'excess', 'HFP',
                 'House', 'Services', 'Overcrowding',
                 'Urban', 'Ethnic', 'uMisery', 'rMisery',
                 'rainfall', 'temperature'
                 ]]

variables_normalizaded = ['HFP', 'rMisery', 'uMisery', 'rainfall', 'temperature']

for var in variables_normalizaded:
    std = data_dengue[var].std()
    mean = data_dengue[var].mean()
    median = data_dengue[var].median()
    print(f"{var}: std = {std:.6f}, mean = {mean:.6f}, median = {median:.6f}")

# HFP with z-score
scaler = StandardScaler()
data_dengue['HFP'] = scaler.fit_transform(data_dengue[['HFP']])
data_dengue['rMisery'] = scaler.fit_transform(data_dengue[['rMisery']])
data_dengue['uMisery'] = scaler.fit_transform(data_dengue[['uMisery']])
data_dengue['rainfall'] = scaler.fit_transform(data_dengue[['rainfall']])
data_dengue['temperature'] = scaler.fit_transform(data_dengue[['temperature']])

data_dengue = data_dengue.dropna()

#%%

# Convert columns to binary
columns_convert = ['House', 'Services', 'Overcrowding', 'Urban', 'Ethnic']
for col in columns_convert:
    median = data_dengue[col].median()
    data_dengue[col] = (data_dengue[col] > median).astype(int) 

#%%
Y = data_dengue['excess'].to_numpy() 
T = data_dengue['HFP'].to_numpy()
W = data_dengue[['House', 'Services', 'Overcrowding', 'Urban', 'Ethnic', 
                 'rMisery', 'uMisery',
                 'rainfall', 'temperature']].to_numpy()
X = data_dengue[['rMisery', 'uMisery', 'rainfall', 'temperature']].to_numpy()

# Split data
X_train, X_test, T_train, T_test, Y_train, Y_test, W_train, W_test = train_test_split(
            X, T, Y, W, test_size=0.2, random_state=123)


#%%
## Ignore warnings
warnings.filterwarnings('ignore') 


#Estimation of ATE
estimate_dengue = SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(),
                                   discrete_treatment=False, cv=5, random_state=123, max_iter=30000)

# fit the model
estimate_dengue.fit(Y=Y, T=T, X=X, W=W, inference='auto')

# predict effect for each sample X
estimate_dengue.effect(X)

# ate
ate_dengue = estimate_dengue.ate(X) 
print(ate_dengue)

# confidence interval of ate
ci_dengue = estimate_dengue.ate_interval(X) 
print(ci_dengue)

# Set values in the df_ATE
df_ATE.at[1, 'ATE'] = round(ate_dengue, 5)
df_ATE.at[1, '95% CI'] = ci_dengue
print(df_ATE)

#%%
#CATE
#range of X for rural misery
# Find the maximum and minimum values of rural misery
min_X0 = np.min(X[:, 0]) 
max_X0 = np.max(X[:, 0])
delta = (max_X0 - min_X0) / 100
X0_grid = np.arange(min_X0, max_X0 + delta - 0.001, delta)

# Means of other variables in X
X1_mean = np.mean(X[:, 1])   
X2_mean = np.mean(X[:, 2])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X0_grid = np.column_stack([
    X0_grid,  
    np.full_like(X0_grid, X1_mean),     
    np.full_like(X0_grid, X2_mean),
    np.full_like(X0_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_dengue.effect(X0_grid)
hte_lower2_cons, hte_upper2_cons = estimate_dengue.effect_interval(X0_grid)

# Reshape para plotting
X0_grid_plot = X0_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X0_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE rural misery- 
cate_rmisery= (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Rural misery (sd)', y='Effect of HFP on excess dengue cases',
           title='a')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_rmisery)

#range of X for urban misery
# Find the maximum and minimum values of rural misery
min_X1 = np.min(X[:, 1]) 
max_X1 = np.max(X[:, 1])
delta = (max_X1 - min_X1) / 100
X1_grid = np.arange(min_X1, max_X1 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X2_mean = np.mean(X[:, 2])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X1_grid = np.column_stack([
    X1_grid,  
    np.full_like(X1_grid, X0_mean),     
    np.full_like(X1_grid, X2_mean),
    np.full_like(X1_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_dengue.effect(X1_grid)
hte_lower2_cons, hte_upper2_cons = estimate_dengue.effect_interval(X1_grid)

# Reshape para plotting
X1_grid_plot = X1_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X1_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE urban misery- 
cate_umisery = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Urban misery (sd)', y='Effect of HFP on excess dengue cases',
           title='b')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_umisery)

#range of X for rainfall
# Find the maximum and minimum values of rural misery
min_X2 = np.min(X[:, 2]) 
max_X2 = np.max(X[:, 2])
delta = (max_X0 - min_X0) / 100
X2_grid = np.arange(min_X2, max_X2 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X1_mean = np.mean(X[:, 1])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X2_grid = np.column_stack([
    X2_grid,  
    np.full_like(X2_grid, X0_mean),     
    np.full_like(X2_grid, X1_mean),
    np.full_like(X2_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_dengue.effect(X2_grid)
hte_lower2_cons, hte_upper2_cons = estimate_dengue.effect_interval(X2_grid)

# Reshape para plotting
X2_grid_plot = X2_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X2_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE Rainfall- 
cate_rainfall = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Rainfall (sd)', y='Effect of HFP on excess dengue cases',
           title='c')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_rainfall)

#range of X for Temperature
# Find the maximum and minimum values of rural misery
min_X3 = np.min(X[:, 3]) 
max_X3 = np.max(X[:, 3])
delta = (max_X0 - min_X0) / 100
X3_grid = np.arange(min_X3, max_X3 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X1_mean = np.mean(X[:, 1])
X2_mean = np.mean(X[:, 2])    

# Matrix of X
X3_grid = np.column_stack([
    X3_grid,  
    np.full_like(X3_grid, X0_mean),     
    np.full_like(X3_grid, X1_mean),
    np.full_like(X3_grid, X2_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_dengue.effect(X3_grid)
hte_lower2_cons, hte_upper2_cons = estimate_dengue.effect_interval(X3_grid)

# Reshape para plotting
X3_grid_plot = X3_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X3_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE Temperature- 
cate_temperature = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Temperature (sd)', y='Effect of HFP on excess dengue cases',
           title='d')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_temperature)




#%%

# HFP as binary
median_HFP = data_dengue['HFP'].median()
data_dengue['HFP'] = (data_dengue['HFP'] > median_HFP).astype(int)

#Causal mechanism
model_dengue = CausalModel(
        data = data_dengue,
        treatment=['HFP'],
        outcome=['excess'],
        graph= """graph[directed 1 
                    node[id "HFP" label "HFP"]
                    node[id "excess" label "excess"]
                    node[id "House" label "House"]
                    node[id "Urban" label "Urban"]
                    node[id "Overcrowding" label "Overcrowding"]
                    node[id "Ethnic" label "Ethnic"]
                    node[id "Services" label "Services"]
                    node[id "rMisery" label "rMisery"]
                    node[id "uMisery" label "uMisery"]
                    node[id "rainfall" label "rainfall"]
                    node[id "temperature" label "temperature"]
                    

                    edge[source "rainfall" target "temperature"]
                    edge[source "rainfall" target "HFP"]
                    edge[source "temperature" target "HFP"]
                    edge[source "rainfall" target "excess"]
                    edge[source "temperature" target "excess"]
                    
                    edge[source "House" target "HFP"]    
                    edge[source "House" target "excess"]
                    edge[source "House" target "Ethnic"]
                    edge[source "House" target "Services"]
                    edge[source "House" target "Overcrowding"]
                    edge[source "House" target "rMisery"]
                    edge[source "House" target "uMisery"]
                    


                    edge[source "Urban" target "rMisery"]
                    edge[source "Urban" target "uMisery"]
                    edge[source "Urban" target "HFP"]
                    edge[source "Urban" target "excess"]
                    edge[source "Urban" target "Services"]
                    


                    edge[source "uMisery" target "HFP"]
                    edge[source "uMisery" target "excess"]
                    
                    edge[source "rMisery" target "HFP"]
                    edge[source "rMisery" target "excess"]
                    
                    
                    edge[source "Overcrowding" target "HFP"]
                    edge[source "Overcrowding" target "excess"]


                    edge[source "Ethnic" target "HFP"]
                    edge[source "Ethnic" target "excess"]
                    


                    edge[source "Services" target "Overcrowding"]
                    edge[source "Services" target "HFP"]
                    edge[source "Services" target "excess"]
                    

                    edge[source "HFP" target "excess"]
                    
                    ]"""
                    )
    
    
#%% 

# Identifying effects
identified_estimand_dengue = model_dengue.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_dengue)

#%%

# Model with DoWhy
estimate_dengue = model_dengue.estimate_effect(
    identified_estimand_dengue,
    effect_modifiers=['rMisery', 'uMisery', 'rainfall', 'temperature'],
    method_name="backdoor.econml.dml.SparseLinearDML",
    confidence_intervals=True,
    method_params={
        "init_params": {
            "featurizer": PolynomialFeatures(degree=3, include_bias=False),
            "model_y":reg1(),
            "model_t":reg1(),
            "discrete_treatment":False,
            "max_iter": 30000,
            "cv": 5,
            "random_state": 123
            },
        }
)

# ATE with DoWhy
ate_dengue_DoWhy = estimate_dengue.value
print(ate_dengue_DoWhy)

#%%
# Refutations

random_dengue = model_dengue.refute_estimate(
    identified_estimand_dengue,
    estimate_dengue,
    method_name="random_common_cause",
    random_state=123,
    num_simulations=50,
    )
print(random_dengue)

subset_dengue = model_dengue.refute_estimate(
    identified_estimand_dengue,
    estimate_dengue,
    subset_fraction=0.1,
    method_name="data_subset_refuter",
    random_state=123,
    num_simulations=50,
    )
print(subset_dengue)

dummy_dengue_results = model_dengue.refute_estimate(
        identified_estimand_dengue,
        estimate_dengue,
        method_name="dummy_outcome_refuter",
        random_state=123,
        num_simulations=50
    )
print(dummy_dengue_results[0])


placebo_dengue = model_dengue.refute_estimate(
    identified_estimand_dengue,
    estimate_dengue,
    method_name="placebo_treatment_refuter",
    random_state=123,
    num_simulations=50,
    )
print(placebo_dengue)

#%%

######################################################################################################################################
#_visceral
#import data
data = pd.read_csv("D:/clases/UDES/artículo huella humana/ml/visceral_final.csv", encoding='latin-1') # Modify the path if necessary

data['excess'] = (data['sir'] > 1).astype(int)
data['rainfall'] = data['rainfall'] *1000
data['temperature'] = data['temperature'] -273.15

data_visceral = data[[
                 'excess', 'HFP',
                 'Coca', 'Forest', 'Mining', 'Fire', 'Deforest',
                 'House', 'Services', 'Overcrowding',
                 'Urban', 'Ethnic', 'uMisery', 'rMisery',
                 'rainfall', 'temperature'
                 ]]

variables_normalizaded = ['HFP', 'rMisery', 'uMisery', 'rainfall', 'temperature']

for var in variables_normalizaded:
    std = data_visceral[var].std()
    mean = data_visceral[var].mean()
    median = data_visceral[var].median()
    print(f"{var}: std = {std:.6f}, mean = {mean:.6f}, median = {median:.6f}")

# HFP with z-score
scaler = StandardScaler()
data_visceral['HFP'] = scaler.fit_transform(data_visceral[['HFP']])
data_visceral['rMisery'] = scaler.fit_transform(data_visceral[['rMisery']])
data_visceral['uMisery'] = scaler.fit_transform(data_visceral[['uMisery']])
data_visceral['rainfall'] = scaler.fit_transform(data_visceral[['rainfall']])
data_visceral['temperature'] = scaler.fit_transform(data_visceral[['temperature']])

data_visceral = data_visceral.dropna()

#%%

# Convert columns to binary
columns_convert = ['Coca', 'Forest', 'Mining', 'Fire', 'Deforest', 'House', 'Services', 'Overcrowding', 'Urban', 'Ethnic']
for col in columns_convert:
    median = data_visceral[col].median()
    data_visceral[col] = (data_visceral[col] > median).astype(int) 

#%%
Y = data_visceral['excess'].to_numpy() 
T = data_visceral['HFP'].to_numpy()
W = data_visceral[['House', 'Services', 'Overcrowding', 'Urban', 'Ethnic',
                   'Coca', 'Forest', 'Mining', 'Fire', 'Deforest', 
                   'rainfall', 'temperature',
                   'rMisery', 'uMisery']].to_numpy()
X = data_visceral[['rMisery', 'uMisery', 'rainfall', 'temperature']].to_numpy()

# Split data
X_train, X_test, T_train, T_test, Y_train, Y_test, W_train, W_test = train_test_split(
            X, T, Y, W, test_size=0.2, random_state=123)


#%%
## Ignore warnings
warnings.filterwarnings('ignore') 

#Estimation of ATE
estimate_visceral = SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(),
                                   discrete_treatment=False, cv=5, random_state=123, max_iter=30000)

# fit the model
estimate_visceral.fit(Y=Y, T=T, X=X, W=W, inference='auto')

# predict effect for each sample X
estimate_visceral.effect(X)

# ate
ate_visceral = estimate_visceral.ate(X) 
print(ate_visceral)

# confidence interval of ate
ci_visceral = estimate_visceral.ate_interval(X) 
print(ci_visceral)

# Set values in the df_ATE
df_ATE.at[2, 'ATE'] = round(ate_visceral, 5)
df_ATE.at[2, '95% CI'] = ci_visceral
print(df_ATE)

#%%
#CATE
#range of X for rural misery
# Find the maximum and minimum values of rural misery
min_X0 = np.min(X[:, 0]) 
max_X0 = np.max(X[:, 0])
delta = (max_X0 - min_X0) / 100
X0_grid = np.arange(min_X0, max_X0 + delta - 0.001, delta)

# Means of other variables in X
X1_mean = np.mean(X[:, 1])   
X2_mean = np.mean(X[:, 2])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X0_grid = np.column_stack([
    X0_grid,  
    np.full_like(X0_grid, X1_mean),     
    np.full_like(X0_grid, X2_mean),
    np.full_like(X0_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_visceral.effect(X0_grid)
hte_lower2_cons, hte_upper2_cons = estimate_visceral.effect_interval(X0_grid)

# Reshape para plotting
X0_grid_plot = X0_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X0_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE rural misery- 
cate_rmisery= (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Rural misery (sd)', y='Effect of HFP on excess visceral leishmaniasis cases',
           title='a')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_rmisery)

#range of X for urban misery
# Find the maximum and minimum values of rural misery
min_X1 = np.min(X[:, 1]) 
max_X1 = np.max(X[:, 1])
delta = (max_X1 - min_X1) / 100
X1_grid = np.arange(min_X1, max_X1 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X2_mean = np.mean(X[:, 2])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X1_grid = np.column_stack([
    X1_grid,  
    np.full_like(X1_grid, X0_mean),     
    np.full_like(X1_grid, X2_mean),
    np.full_like(X1_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_visceral.effect(X1_grid)
hte_lower2_cons, hte_upper2_cons = estimate_visceral.effect_interval(X1_grid)

# Reshape para plotting
X1_grid_plot = X1_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X1_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE urban misery- 
cate_umisery = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Urban misery (sd)', y='Effect of HFP on excess visceral leishmaniasis cases',
           title='b')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_umisery)

#range of X for rainfall
# Find the maximum and minimum values of rural misery
min_X2 = np.min(X[:, 2]) 
max_X2 = np.max(X[:, 2])
delta = (max_X0 - min_X0) / 100
X2_grid = np.arange(min_X2, max_X2 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X1_mean = np.mean(X[:, 1])
X3_mean = np.mean(X[:, 3])    

# Matrix of X
X2_grid = np.column_stack([
    X2_grid,  
    np.full_like(X2_grid, X0_mean),     
    np.full_like(X2_grid, X1_mean),
    np.full_like(X2_grid, X3_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_visceral.effect(X2_grid)
hte_lower2_cons, hte_upper2_cons = estimate_visceral.effect_interval(X2_grid)

# Reshape para plotting
X2_grid_plot = X2_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X2_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE Rainfall- 
cate_rainfall = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Rainfall (sd)', y='Effect of HFP on excess visceral leishmaniasis cases',
           title='c')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_rainfall)

#range of X for Temperature
# Find the maximum and minimum values of rural misery
min_X3 = np.min(X[:, 3]) 
max_X3 = np.max(X[:, 3])
delta = (max_X0 - min_X0) / 100
X3_grid = np.arange(min_X3, max_X3 + delta - 0.001, delta)

# Means of other variables in X
X0_mean = np.mean(X[:, 0])   
X1_mean = np.mean(X[:, 1])
X2_mean = np.mean(X[:, 2])    

# Matrix of X
X3_grid = np.column_stack([
    X3_grid,  
    np.full_like(X3_grid, X0_mean),     
    np.full_like(X3_grid, X1_mean),
    np.full_like(X3_grid, X2_mean)  
])

# Conditional marginal effect
treatment_cont_marg = estimate_visceral.effect(X3_grid)
hte_lower2_cons, hte_upper2_cons = estimate_visceral.effect_interval(X3_grid)

# Reshape para plotting
X3_grid_plot = X3_grid[:, 0]
treatment_cont_marg_plot = treatment_cont_marg

# DataFrame for plotting
plot_data = pd.DataFrame({
    'X_test': X3_grid_plot,
    'treatment_cont_marg': treatment_cont_marg,
    'hte_lower2_cons': hte_lower2_cons,
    'hte_upper2_cons': hte_upper2_cons
})

# Figure CATE Temperature- 
cate_temperature = (
    ggplot(plot_data)
    + aes(x='X_test', y='treatment_cont_marg')
    + geom_line(color='blue', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Temperature (sd)', y='Effect of HFP on excess visceral leishmaniais cases',
           title='d')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(plot_title=element_text(hjust=0.5, size=12),
            axis_title_x=element_text(size=10),
            axis_title_y=element_text(size=10))
)
print(cate_temperature)

#%%

# HFP as binary
median_HFP = data_visceral['HFP'].median()
data_visceral['HFP'] = (data_visceral['HFP'] > median_HFP).astype(int)

#Causal mechanism
model_visceral = CausalModel(
        data = data_visceral,
        treatment=['HFP'],
        outcome=['excess'],
        graph= """graph[directed 1 
                    node[id "HFP" label "HFP"]
                    node[id "excess" label "excess"]
                    node[id "House" label "House"]
                    node[id "Urban" label "Urban"]
                    node[id "Overcrowding" label "Overcrowding"]
                    node[id "Ethnic" label "Ethnic"]
                    node[id "Services" label "Services"]
                    
                    node[id "Forest" label "Forest"]
                    node[id "Deforest" label "Deforest"]
                    node[id "Coca" label "Coca"]
                    node[id "Mining" label "Mining"]
                    node[id "Fire" label "Fire"]
                    
                    node[id "rMisery" label "rMisery"]
                    node[id "uMisery" label "uMisery"]
                    node[id "rainfall" label "rainfall"]
                    node[id "temperature" label "temperature"]
                    
                    edge[source "rainfall" target "temperature"]
                    edge[source "rainfall" target "Forest"]
                    edge[source "temperature" target "Forest"]
                    edge[source "rainfall" target "HFP"]
                    edge[source "temperature" target "HFP"]
                    edge[source "rainfall" target "excess"]
                    edge[source "temperature" target "excess"]
                    

                    edge[source "Forest" target "HFP"]    
                    edge[source "Forest" target "excess"]


                    edge[source "Forest" target "Deforest"]
                    edge[source "rMisery" target "Deforest"]
                    edge[source "uMisery" target "Deforest"]
                    edge[source "Deforest" target "HFP"]
                    edge[source "Deforest" target "excess"]
                    

                    edge[source "Forest" target "Coca"]
                    edge[source "rMisery" target "Coca"]
                    edge[source "uMisery" target "Coca"]
                    edge[source "Coca" target "HFP"]
                    edge[source "Coca" target "excess"]


                    edge[source "Forest" target "Mining"]
                    edge[source "rMisery" target "Mining"]
                    edge[source "uMisery" target "Mining"]
                    edge[source "Mining" target "HFP"]
                    edge[source "Mining" target "excess"]
                    

                    edge[source "Forest" target "Fire"]
                    edge[source "Deforest" target "Fire"]
                    edge[source "Coca" target "Fire"]
                    edge[source "rMisery" target "Fire"]
                    edge[source "uMisery" target "Fire"]
                    edge[source "Fire" target "HFP"]
                    edge[source "Fire" target "excess"]
    



                    
                    edge[source "House" target "HFP"]    
                    edge[source "House" target "excess"]
                    edge[source "House" target "Ethnic"]
                    edge[source "House" target "Services"]
                    edge[source "House" target "Overcrowding"]
                    edge[source "House" target "rMisery"]
                    edge[source "House" target "uMisery"]
                    


                    edge[source "Urban" target "rMisery"]
                    edge[source "Urban" target "uMisery"]
                    edge[source "Urban" target "HFP"]
                    edge[source "Urban" target "excess"]
                    edge[source "Urban" target "Services"]
                    


                    edge[source "rMisery" target "HFP"]
                    edge[source "rMisery" target "excess"]
                    
                    edge[source "uMisery" target "HFP"]
                    edge[source "uMisery" target "excess"]
                    
                    
                    edge[source "Overcrowding" target "HFP"]
                    edge[source "Overcrowding" target "excess"]


                    edge[source "Ethnic" target "HFP"]
                    edge[source "Ethnic" target "excess"]
                    


                    edge[source "Services" target "Overcrowding"]
                    edge[source "Services" target "HFP"]
                    edge[source "Services" target "excess"]
                    

                    edge[source "HFP" target "excess"]
                    
                    ]"""
                    )
    
    
#%% 

# Identifying effects
identified_estimand_visceral = model_visceral.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_visceral)

#%%

# Model with DoWhy
estimate_visceral = model_visceral.estimate_effect(
    identified_estimand_visceral,
    effect_modifiers=['rMisery', 'uMisery', 'rainfall', 'temperature'],
    method_name="backdoor.econml.dml.SparseLinearDML",
    confidence_intervals=True,
    method_params={
        "init_params": {
            "featurizer": PolynomialFeatures(degree=3, include_bias=False),
            "model_y":reg1(),
            "model_t":reg1(),
            "discrete_treatment":False,
            "max_iter": 30000,
            "cv": 5,
            "random_state": 123
            },
        }
)

# ATE with DoWhy
ate_visceral_DoWhy = estimate_visceral.value
print(ate_visceral_DoWhy)

#%%
# Refutations

random_visceral = model_visceral.refute_estimate(
    identified_estimand_visceral,
    estimate_visceral,
    method_name="random_common_cause",
    random_state=123,
    num_simulations=50,
    )
print(random_visceral)

subset_visceral = model_visceral.refute_estimate(
    identified_estimand_visceral,
    estimate_visceral,
    subset_fraction=0.1,
    method_name="data_subset_refuter",
    random_state=123,
    num_simulations=50,
    )
print(subset_visceral)

dummy_visceral_results = model_visceral.refute_estimate(
        identified_estimand_visceral,
        estimate_visceral,
        method_name="dummy_outcome_refuter",
        random_state=123,
        num_simulations=50
    )
print(dummy_visceral_results[0])


placebo_visceral = model_visceral.refute_estimate(
    identified_estimand_visceral,
    estimate_visceral,
    method_name="placebo_treatment_refuter",
    random_state=123,
    num_simulations=50,
    )
print(placebo_visceral)

#%%


#Figure 2
labs = ['Malaria',
        'Dengue',
        'Visceral leishmaniasis']

df_labs = pd.DataFrame({'Labels': labs})

print(df_ATE)

# Convert to separate DataFrame
ATE_05 = df_ATE[['ATE']].round(5)
print(ATE_05)

# Convert tuples in the '95% CI' column to separate DataFrame
df_ci = df_ATE['95% CI'].apply(pd.Series)

# Rename columns in df_ci
df_ci.columns = ['Lower', 'Upper']

# Create two separate DataFrames for Lower and Upper
Lower = df_ci[['Lower']].copy()
print(Lower)
Upper = df_ci[['Upper']].copy()
print(Upper)


df_plot = pd.concat([df_labs.reset_index(drop=True), ATE_05, Lower, Upper], axis=1)
print(df_plot)

p = EffectMeasurePlot(label=df_plot.Labels, effect_measure=df_plot.ATE, lcl=df_plot.Lower, ucl=df_plot.Upper)
p.labels(center=0)
p.colors(pointcolor='r' , pointshape="s", linecolor='b')
p.labels(effectmeasure='ATE')  
p.plot(figsize=(10, 5), t_adjuster=0.12, max_value=0.1, min_value=-0.2)
plt.tight_layout()
plt.show()











