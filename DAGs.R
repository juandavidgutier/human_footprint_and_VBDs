###############################################################################
# Code for the paper:
#   "Anthropogenic Landscapes and Vector-Borne Disease Dynamics: Unveiling the Comple
#    Interplay between Human Footprint and Disease Transmission in Colombia?" Gutiérrez, Quintero-García, Xiao, Cuadros
#    Last modification = 19/08/2024
#    e-mail = juandavidgutier@gmail.com, jdgutierrez@udes.edu.co
###############################################################################


library(ggdag)
library(dagitty)
library(lavaan)
library(CondIndTests)
library(dplyr)
library(GGally)
library(tidyr)
library(MKdescr)

################################################################################
# For malaria
data <- read.csv("D:/malaria_final.csv") # Modify the path if necessary

data_malaria <- select(data, excess, HFP, Coca, Forest, Mining, Fire, Deforest, Misery, rainfall, temperature)

#sd units
data_malaria$excess <- zscore(data_malaria$excess, na.rm = TRUE)
data_malaria$HFP <- zscore(data_malaria$HFP, na.rm = TRUE)
data_malaria$Coca <- zscore(data_malaria$Coca, na.rm = TRUE)
data_malaria$Forest <- zscore(data_malaria$Forest, na.rm = TRUE)
data_malaria$Mining <- zscore(data_malaria$Mining, na.rm = TRUE)
data_malaria$Fire <- zscore(data_malaria$Fire, na.rm = TRUE)
data_malaria$Deforest <- zscore(data_malaria$Deforest, na.rm = TRUE)
data_malaria$Misery <- zscore(data_malaria$Misery, na.rm = TRUE)

#DAG 
dag <- dagitty('dag {
excess [pos="0, 0.5"]
HFP  [pos="-0.2, 0.5"]

Forest [pos="-1.6, 1.1"]
Deforest [pos="-0.20, -0.40"]
Coca [pos="-0.30, 1.2"]
Mining [pos="-1.1, -1.4"]
Fire [pos="-1.4, -0.90"]
rainfall [pos="-2.4, 1.6"]
temperature [pos="-2.0, 1.6"]
uMisery [pos="-2.0, -1.6"]
rMisery [pos="-2.0, -2.0"]

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

}')  

plot(dag)

#check whether any correlations are perfect (i.e., collinearity)
myCov <- cov(data_malaria)
round(myCov, 2)

myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)

#if not, check for multicollinearity (i.e., is one variable a linear combination of 2+ variables?)
det(myCov) < 0
#or
any(eigen(myCov)$values < 0)

#cond. independences
impliedConditionalIndependencies(dag)
corr <- lavCor(data_malaria)

summary(corr)

#plot
localTests(dag, sample.cov=corr, sample.nobs=nrow(data_malaria))
plotLocalTestResults(localTests(dag, sample.cov=corr, sample.nobs=nrow(data_malaria)), xlim=c(-1,1))
# Notice there is not conditional independences 



##############################################################################
# For dengue
data <- read.csv("D:/dengue_final.csv") # Modify the path if necessary

data_dengue <- select(data, excess, HFP, Services, House, Overcrowding, Urban, Ethnic, Misery, rainfall, temperature)

#sd units
data_dengue$excess <- zscore(data_dengue$excess, na.rm = TRUE)
data_dengue$HFP <- zscore(data_dengue$HFP, na.rm = TRUE)
data_dengue$Services <- zscore(data_dengue$Services, na.rm = TRUE)
data_dengue$House <- zscore(data_dengue$House, na.rm = TRUE)
data_dengue$Overcrowding <- zscore(data_dengue$Overcrowding, na.rm = TRUE)
data_dengue$Urban <- zscore(data_dengue$Urban, na.rm = TRUE)
data_dengue$Ethnic <- zscore(data_dengue$Ethnic, na.rm = TRUE)
data_dengue$Misery <- zscore(data_dengue$Misery, na.rm = TRUE)

#DAG 
dag <- dagitty('dag {
excess [pos="0, 0.5"]
HFP  [pos="-0.2, 0.5"]

House [pos="-1.6, 1.1"]
Ethnic [pos="-0.4, -0.40"]
Services [pos="-0.30, 1.2"]
Overcrowding [pos="-1.1, -1.4"]
Misery [pos="-1.1, 1.30"]
Urban [pos="-1.4, -0.90"]
rainfall [pos="-2.4, 1.6"]
temperature [pos="-2.0, 1.6"]

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
Misery -> HFP
Misery -> excess

House -> Services
House -> Overcrowding
House -> Ethnic
House -> Misery

Ethnic -> Overcrowding
Ethnic -> Services
Ethnic -> Urban
Ethnic -> Misery

Urban -> Services
Urban -> Misery

Services -> Overcrowding
Services -> Misery

Overcrowding -> Misery
Overcrowding -> Urban

HFP -> excess
}')  
plot(dag)

#check whether any correlations are perfect (i.e., collinearity)
myCov <- cov(data_dengue)
round(myCov, 2)

myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)

#if not, check for multicollinearity (i.e., is one variable a linear combination of 2+ variables?)
det(myCov) < 0
#or
any(eigen(myCov)$values < 0)

#cond. independences
impliedConditionalIndependencies(dag)
corr <- lavCor(data_dengue)

summary(corr)

#plot
localTests(dag, sample.cov=corr, sample.nobs=nrow(data_dengue))
plotLocalTestResults(localTests(dag, sample.cov=corr, sample.nobs=nrow(data_dengue)), xlim=c(-1,1))
#Notice there is not conditional independences



##############################################################################
# For visceral leishmaniasis
data <- read.csv("D:/visceral_final.csv") # Modify the path if necessary

data_visceral <- select(data, excess, HFP, Coca, Forest, Mining, Fire, Deforest, rMisery,
                        Services, House, Overcrowding, Urban, Ethnic, uMisery, rainfall, temperature)

#sd units
data_visceral$excess <- zscore(data_visceral$excess, na.rm = TRUE)
data_visceral$HFP <- zscore(data_visceral$HFP, na.rm = TRUE)
data_visceral$Coca <- zscore(data_visceral$Coca, na.rm = TRUE)
data_visceral$Forest <- zscore(data_visceral$Forest, na.rm = TRUE)
data_visceral$Mining <- zscore(data_visceral$Mining, na.rm = TRUE)
data_visceral$Fire <- zscore(data_visceral$Fire, na.rm = TRUE)
data_visceral$Deforest <- zscore(data_visceral$Deforest, na.rm = TRUE)
data_visceral$rMisery <- zscore(data_visceral$rMisery, na.rm = TRUE)
data_visceral$excess <- zscore(data_visceral$excess, na.rm = TRUE)
data_visceral$HFP <- zscore(data_visceral$HFP, na.rm = TRUE)
data_visceral$Services <- zscore(data_visceral$Services, na.rm = TRUE)
data_visceral$House <- zscore(data_visceral$House, na.rm = TRUE)
data_visceral$Overcrowding <- zscore(data_visceral$Overcrowding, na.rm = TRUE)
data_visceral$Urban <- zscore(data_visceral$Urban, na.rm = TRUE)
data_visceral$Ethnic <- zscore(data_visceral$Ethnic, na.rm = TRUE)
data_visceral$uMisery <- zscore(data_visceral$uMisery, na.rm = TRUE)

#DAG 
dag <- dagitty('dag {
excess [pos="0, 0.5"]
HFP  [pos="-0.2, 0.5"]

Forest [pos="-1.6, 1.1"]
Deforest [pos="-0.20, -0.40"]
Coca [pos="-0.30, 1.2"]
Mining [pos="-1.1, -1.4"]
rMisery [pos="-1.1, 1.30"]
Fire [pos="-1.4, -0.90"]
House [pos="-1.8, 2.1"]
Ethnic [pos="-0.4, -0.40"]
Services [pos="-0.9, 1.8"]
Overcrowding [pos="-1.3, -1.7"]
uMisery [pos="-1.5, 1.8"]
Urban [pos="-1.8, -1.70"]
rainfall [pos="-2.4, 1.6"]
temperature [pos="-2.0, 1.6"]

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
Deforest -> Forest

Fire -> Mining

Coca -> Fire
Coca -> Mining

rMisery -> Deforest
rMisery -> Fire
rMisery -> Coca
rMisery -> Mining
rMisery -> Ethnic

Mining -> Deforest
Mining -> Forest

Deforest -> Urban
Coca -> uMisery
uMisery -> rMisery 
rMisery -> Services
Mining -> Urban
Forest -> Services
Overcrowding -> Coca
Mining -> Services
Fire -> uMisery
Deforest -> uMisery
Mining -> uMisery

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

Services -> Overcrowding
Services -> uMisery

Overcrowding -> uMisery
Overcrowding -> Urban

HFP -> excess
}')  
plot(dag)

#check whether any correlations are perfect (i.e., collinearity)
myCov <- cov(data_visceral)
round(myCov, 2)

myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)

#if not, check for multicollinearity (i.e., is one variable a linear combination of 2+ variables?)
det(myCov) < 0
#or
any(eigen(myCov)$values < 0)

#cond. independences
impliedConditionalIndependencies(dag)
corr <- lavCor(data_visceral)

summary(corr)

#plot
localTests(dag, sample.cov=corr, sample.nobs=nrow(data_visceral))
plotLocalTestResults(localTests(dag, sample.cov=corr, sample.nobs=nrow(data_visceral)), xlim=c(-1,1))
#Notice there is not conditional independences

