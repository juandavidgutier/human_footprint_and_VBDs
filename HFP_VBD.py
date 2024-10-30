
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
data = pd.read_csv("D:/malaria_final.csv", encoding='latin-1') # Modify the path if necessary
data['excess_cases'] = (data['sir'] > 1).astype(int)

# HFP as binary
median_HFP = data['HFP'].median()
print(median_HFP)
data['HFP'] = (data['HFP'] > median_HFP).astype(int)

data_malaria = data[[
                 'excess_cases', 'HFP',
                 'Coca', 'Forest', 'Mining', 'Fire', 
                 'Deforest', 'uMisery', 'rMisery'
                 ]]

#%%

#z-score
data_malaria.Coca = stats.zscore(data_malaria.Coca, nan_policy='omit') 
data_malaria.Forest = stats.zscore(data_malaria.Forest, nan_policy='omit')
data_malaria.Mining = stats.zscore(data_malaria.Mining, nan_policy='omit') 
data_malaria.Fire = stats.zscore(data_malaria.Fire, nan_policy='omit') 
data_malaria.Deforest = stats.zscore(data_malaria.Deforest, nan_policy='omit')

data_malaria = data_malaria.dropna()

#%%
Y = data_malaria.excess_cases.to_numpy() 
T = data_malaria.HFP.to_numpy()
W = data_malaria[['Coca', 'Forest', 'Mining', 'Fire', 'Deforest', 'rMisery', 'uMisery']].to_numpy().reshape(-1, 7)
X = data_malaria[['rMisery', 'uMisery']].to_numpy().reshape(-1, 2)

## Ignore warnings
warnings.filterwarnings('ignore') 

reg1 = lambda: XGBRegressor(n_estimators=2000, random_state=123)

#%%
#Estimation of ATE
estimate_malaria = SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(),
                                   discrete_treatment=True, cv=3, random_state=123)

estimate_malaria = estimate_malaria.dowhy

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
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)


# Find the maximum and minimum values of urban misery
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

est2_malaria =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(), 
                                cv=3, random_state=123)

est2_malaria.fit(Y=Y, T=T, X=X, W=W, inference="auto")

treatment_effects2 = est2_malaria.const_marginal_effect(X_test)
te_lower2_cons, te_upper2_cons = est2_malaria.const_marginal_effect_interval(X_test)

X_test = X_test[:, 0].ravel()
treatment_effects2 = treatment_effects2.ravel()

# Reshape to 1-dimensional arrays
te_lower2_cons = te_lower2_cons.ravel()
te_upper2_cons = te_upper2_cons.ravel()

#Figure 3A
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  #+ geom_point() 
  + geom_line()
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='Rural misery', y='Effect of HFP on excess malaria cases')
  + geom_hline(yintercept = 0, linetype = "dotted", color="red")
  + ggtitle("A")
  + theme(panel_background=element_rect(fill='lightblue'),  
          axis_text_x=element_text(size=12), 
          axis_text_y=element_text(size=12),
          axis_title_x=element_text(size=15),
          axis_title_y=element_text(size=15),
          plot_title=element_text(size=18, hjust=0.5))
)


#%%
#range of X for urban misery
# Find the maximum and minimum values of urban misery
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)


# Find the maximum and minimum values of urban misery
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

X_test = X_test[:, 1].ravel()
treatment_effects2 = treatment_effects2.ravel()

# Reshape to 1-dimensional arrays
te_lower2_cons = te_lower2_cons.ravel()
te_upper2_cons = te_upper2_cons.ravel()


#Figure 3B
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  #+ geom_point() 
  + geom_line()
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='Urban misery', y='Effect of HFP on excess malaria cases')
  + geom_hline(yintercept = 0, linetype = "dotted", color="red")
  + ggtitle("B")
  + theme(panel_background=element_rect(fill='lightblue'),  
          axis_text_x=element_text(size=12), 
          axis_text_y=element_text(size=12),
          axis_title_x=element_text(size=15),
          axis_title_y=element_text(size=15),
          plot_title=element_text(size=18, hjust=0.5))
)

#%%
#Refute tests
#with random common cause
random_malaria = estimate_malaria.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_malaria)

#with replace a random subset of the data
subset_malaria = estimate_malaria.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=10)
print(subset_malaria)

#with replace a dummy outcome
dummy_malaria = estimate_malaria.refute_estimate(method_name="dummy_outcome_refuter", random_state=123, num_simulations=10)
print(dummy_malaria[0])

#with placebo 
placebo_malaria = estimate_malaria.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=10)
print(placebo_malaria)

#%%

######################################################################################################################################
#_dengue
#import data
data = pd.read_csv("D:/dengue_final.csv", encoding='latin-1') # Modify the path if necessary
data['excess_cases'] = (data['sir'] > 1).astype(int)

# HFP as binary
median_HFP = data['HFP'].median()
print(median_HFP)
data['HFP'] = (data['HFP'] > median_HFP).astype(int)

data_dengue = data[[
                 'excess_cases', 'HFP',
                 'House', 'Services', 'Overcrowding',
                 'Urban', 'Ethnic', 'uMisery', 'rMisery'
                 ]]

#%%

#z-score
data_dengue.House = stats.zscore(data_dengue.House, nan_policy='omit') 
data_dengue.Services = stats.zscore(data_dengue.Services, nan_policy='omit')
data_dengue.Overcrowding = stats.zscore(data_dengue.Overcrowding, nan_policy='omit') 
data_dengue.Urban = stats.zscore(data_dengue.Urban, nan_policy='omit') 
data_dengue.Ethnic = stats.zscore(data_dengue.Ethnic, nan_policy='omit')

data_dengue = data_dengue.dropna()

#%%
Y = data_dengue.excess_cases.to_numpy() 
T = data_dengue.HFP.to_numpy()
W = data_dengue[['House', 'Services', 'Overcrowding', 'Urban', 'Ethnic', 'rMisery', 'uMisery']].to_numpy().reshape(-1, 7)
X = data_dengue[['rMisery', 'uMisery']].to_numpy().reshape(-1, 2)

#%%
#Estimation of ATE
estimate_dengue = SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(),
                                   discrete_treatment=True, cv=3, random_state=123)

estimate_dengue = estimate_dengue.dowhy

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
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)


# Find the maximum and minimum values of urban misery
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

est2_dengue =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(), 
                                cv=3, random_state=123)

est2_dengue.fit(Y=Y, T=T, X=X, W=W, inference="auto")

treatment_effects2 = est2_dengue.const_marginal_effect(X_test)
te_lower2_cons, te_upper2_cons = est2_dengue.const_marginal_effect_interval(X_test)

X_test = X_test[:, 0].ravel()
treatment_effects2 = treatment_effects2.ravel()

# Reshape to 1-dimensional arrays
te_lower2_cons = te_lower2_cons.ravel()
te_upper2_cons = te_upper2_cons.ravel()

#Figure 3C
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  #+ geom_point() 
  + geom_line()
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='Rural misery', y='Effect of HFP on excess dengue cases')
  + geom_hline(yintercept = 0, linetype = "dotted", color="red")
  + ggtitle("C")
  + theme(panel_background=element_rect(fill='lightblue'),  
          axis_text_x=element_text(size=12), 
          axis_text_y=element_text(size=12),
          axis_title_x=element_text(size=15),
          axis_title_y=element_text(size=15),
          plot_title=element_text(size=18, hjust=0.5))
)

#%%

#range of X for urban misery
# Find the maximum and minimum values of urban misery
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)

# Find the maximum and minimum values of urban misery
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

X_test = X_test[:, 1].ravel()
treatment_effects2 = treatment_effects2.ravel()

# Reshape to 1-dimensional arrays
te_lower2_cons = te_lower2_cons.ravel()
te_upper2_cons = te_upper2_cons.ravel()


#Figure 3D
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  #+ geom_point() 
  + geom_line()
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='Urban misery', y='Effect of HFP on excess dengue cases')
  + geom_hline(yintercept = 0, linetype = "dotted", color="red")
  + ggtitle("D")
  + theme(panel_background=element_rect(fill='lightblue'),  
          axis_text_x=element_text(size=12), 
          axis_text_y=element_text(size=12),
          axis_title_x=element_text(size=15),
          axis_title_y=element_text(size=15),
          plot_title=element_text(size=18, hjust=0.5))
)


#%%
#Refute tests
#with random common cause
random_dengue = estimate_dengue.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_dengue)

#with replace a random subset of the data
subset_dengue = estimate_dengue.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=10)
print(subset_dengue)

#with replace a dummy outcome
dummy_dengue = estimate_dengue.refute_estimate(method_name="dummy_outcome_refuter", random_state=123, num_simulations=10)
print(dummy_dengue[0])

#with placebo 
placebo_dengue = estimate_dengue.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=10)
print(placebo_dengue)



#%%

######################################################################################################################################
#_visceral
#import data
data = pd.read_csv("D:/visceral_final.csv", encoding='latin-1') # Modify the path if necessary
data['excess_cases'] = (data['sir'] > 1).astype(int)

# HFP as binary
median_HFP = data['HFP'].median()
print(median_HFP)
data['HFP'] = (data['HFP'] > median_HFP).astype(int)

data_visceral = data[[
                 'excess_cases', 'HFP',
                 'House', 'Services', 'Overcrowding', 'Urban', 'Ethnic', 'uMisery',
                 'Coca', 'Forest', 'Mining', 'Fire', 'Deforest', 'rMisery'
                  ]]

#%%

#z-score
data_visceral.House = stats.zscore(data_visceral.House, nan_policy='omit') 
data_visceral.Overcrowding = stats.zscore(data_visceral.Overcrowding, nan_policy='omit') 
data_visceral.Urban = stats.zscore(data_visceral.Urban, nan_policy='omit') 
data_visceral.Ethnic = stats.zscore(data_visceral.Ethnic, nan_policy='omit')
data_visceral.Services = stats.zscore(data_visceral.Services, nan_policy='omit')
data_visceral.Coca = stats.zscore(data_visceral.Coca, nan_policy='omit') 
data_visceral.Forest = stats.zscore(data_visceral.Forest, nan_policy='omit')
data_visceral.Mining = stats.zscore(data_visceral.Mining, nan_policy='omit') 
data_visceral.Fire = stats.zscore(data_visceral.Fire, nan_policy='omit') 
data_visceral.Deforest = stats.zscore(data_visceral.Deforest, nan_policy='omit')

data_visceral = data_visceral.dropna()

#%%

Y = data_visceral.excess_cases.to_numpy()  
T = data_visceral.HFP.to_numpy()
W = data_visceral[['House', 'Services', 'Overcrowding', 'Urban', 'Ethnic', 'uMisery',
                   'Coca', 'Forest', 'Mining', 'Fire', 'Deforest', 'rMisery']].to_numpy().reshape(-1, 12)
X = data_visceral[['rMisery', 'uMisery']].to_numpy().reshape(-1, 2)

#%%

#Estimation of ATE
estimate_visceral = SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(),
                                   discrete_treatment=True, cv=3, random_state=123)

estimate_visceral = estimate_visceral.dowhy

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
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)


# Find the maximum and minimum values of urban misery
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

est2_visceral =  SparseLinearDML(featurizer=PolynomialFeatures(degree=3, include_bias=False), model_y=reg1(), model_t=reg1(), 
                                cv=3, random_state=123)

est2_visceral.fit(Y=Y, T=T, X=X, W=W, inference="auto")

treatment_effects2 = est2_visceral.const_marginal_effect(X_test)
te_lower2_cons, te_upper2_cons = est2_visceral.const_marginal_effect_interval(X_test)

X_test = X_test[:, 0].ravel()
treatment_effects2 = treatment_effects2.ravel()

# Reshape to 1-dimensional arrays
te_lower2_cons = te_lower2_cons.ravel()
te_upper2_cons = te_upper2_cons.ravel()

#Figure 3E
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  #+ geom_point() 
  + geom_line()
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='Rural misery', y='Effect of HFP on excess visceral leishmaniasis cases')
  + geom_hline(yintercept = 0, linetype = "dotted", color="red")
  + ggtitle("E")
  + theme(panel_background=element_rect(fill='lightblue'),  
          axis_text_x=element_text(size=12), 
          axis_text_y=element_text(size=12),
          axis_title_x=element_text(size=15),
          axis_title_y=element_text(size=15),
          plot_title=element_text(size=18, hjust=0.5))
)



#%%

#range of X for urban misery
# Find the maximum and minimum values of urban misery
max_value0 = max(X[:, 0])
min_value0 = min(X[:, 0])
min_X0 = min_value0
max_X0 = max_value0
delta0 = (max_X0 - min_X0) / 100
X_test0 = np.arange(min_X0, max_X0 + delta0 - 0.001, delta0).reshape(-1, 1)


# Find the maximum and minimum values of urban misery
max_value1 = max(X[:, 1])
min_value1 = min(X[:, 1])
min_X1 = min_value1
max_X1 = max_value1
delta1 = (max_X1 - min_X1) / 100
X_test1 = np.arange(min_X1, max_X1 + delta1 - 0.001, delta1).reshape(-1, 1)

X_test = np.concatenate((X_test0, X_test1), axis=1)

X_test = X_test[:, 1].ravel()
treatment_effects2 = treatment_effects2.ravel()

# Reshape to 1-dimensional arrays
te_lower2_cons = te_lower2_cons.ravel()
te_upper2_cons = te_upper2_cons.ravel()


#Figure 3F
(
ggplot(aes(x=X_test.flatten(), y=treatment_effects2)) 
  #+ geom_point() 
  + geom_line()
  + geom_ribbon(aes(ymin = te_lower2_cons, ymax = te_upper2_cons), alpha = .1)
  + labs(x='Urban misery', y='Effect of HFP on excess visceral leishmaniasis cases')
  + geom_hline(yintercept = 0, linetype = "dotted", color="red")
  + ggtitle("F")
  + theme(panel_background=element_rect(fill='lightblue'),  
          axis_text_x=element_text(size=12), 
          axis_text_y=element_text(size=12),
          axis_title_x=element_text(size=15),
          axis_title_y=element_text(size=15),
          plot_title=element_text(size=18, hjust=0.5))
)

#%%
#Refute tests
#with random common cause
random_visceral = estimate_visceral.refute_estimate(method_name="random_common_cause", random_state=123, num_simulations=10)
print(random_visceral)

#with replace a random subset of the data
subset_visceral = estimate_visceral.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=10)
print(subset_visceral)

#with replace a dummy outcome
dummy_visceral = estimate_visceral.refute_estimate(method_name="dummy_outcome_refuter", random_state=123, num_simulations=10)
print(dummy_visceral[0])

#with placebo 
placebo_visceral = estimate_visceral.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=10)
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
p.plot(figsize=(10, 5), t_adjuster=0.12, max_value=2, min_value=-1)
plt.tight_layout()
plt.show()











