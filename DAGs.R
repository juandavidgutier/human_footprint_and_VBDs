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

data_malaria <- select(data, sir, HFP, Coca, Forest, Mining, Fire, Deforest, Misery)

#sd units
data_malaria$sir <- zscore(data_malaria$sir, na.rm = TRUE)
data_malaria$HFP <- zscore(data_malaria$HFP, na.rm = TRUE)
data_malaria$Coca <- zscore(data_malaria$Coca, na.rm = TRUE)
data_malaria$Forest <- zscore(data_malaria$Forest, na.rm = TRUE)
data_malaria$Mining <- zscore(data_malaria$Mining, na.rm = TRUE)
data_malaria$Fire <- zscore(data_malaria$Fire, na.rm = TRUE)
data_malaria$Deforest <- zscore(data_malaria$Deforest, na.rm = TRUE)
data_malaria$Misery <- zscore(data_malaria$Misery, na.rm = TRUE)

#DAG 
dag <- dagitty('dag {
sir [pos="0, 0.5"]
HFP  [pos="-0.2, 0.5"]

Forest [pos="-1.6, 1.1"]
Deforest [pos="-0.20, -0.40"]
Coca [pos="-0.30, 1.2"]
Mining [pos="-1.1, -1.4"]
Misery [pos="-1.1, 1.30"]
Fire [pos="-1.4, -0.90"]

Forest -> HFP
Forest -> sir
Deforest -> HFP
Deforest -> sir
Coca -> HFP
Coca -> sir
Mining -> HFP
Mining -> sir
Fire -> HFP
Fire -> sir
Misery -> HFP
Misery -> sir

Forest -> Deforest
Forest -> Fire
Forest -> Coca
Forest -> Misery

Deforest -> Fire
Deforest -> Coca
Deforest -> Forest

Fire -> Mining

Coca -> Fire
Coca -> Mining

Misery -> Deforest
Misery -> Fire
Misery -> Coca
Misery -> Mining

Mining -> Deforest
Mining -> Forest

HFP -> sir

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

#identification
simple_dag <- dagify(
  sir ~  HFP + Coca + Forest + Mining + Fire +  Deforest + Misery,
  HFP ~ Coca + Forest + Mining + Fire + Deforest + Misery, 
  Coca ~ Forest + Deforest + Misery, 
  Forest ~ Mining + Deforest,
  Mining ~ Misery + Coca + Fire,
  Fire ~ Forest + Deforest +  Misery + Coca,
  Deforest ~ Forest + Misery + Mining,
  Misery ~ Forest,
  exposure = "HFP",
  outcome = "sir",
  coords = list(x = c(HFP=2, sir=2, Forest=3.5, Coca=-1.12, Mining=3.2, Fire=-1.0, Deforest=1.2, Misery=-0.8),
                y = c(HFP=1.8, sir=1, Forest=3.5, Coca=2.0, Mining=2.2, Fire=3.3, Deforest=3.2, Misery=2.5))
    )

# theme_dag() coloca la trama en un fondo blanco sin etiquetas en los ejes
ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()

#adjusting
adjustmentSets(simple_dag,  type = "minimal")
## {z_miseria, indixes}

ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()



##############################################################################
# For dengue
data <- read.csv("D:/dengue_final.csv") # Modify the path if necessary

data_dengue <- select(data, sir, HFP, Services, House, Overcrowding, Urban, Ethnic, Misery)

#sd units
data_dengue$sir <- zscore(data_dengue$sir, na.rm = TRUE)
data_dengue$HFP <- zscore(data_dengue$HFP, na.rm = TRUE)
data_dengue$Services <- zscore(data_dengue$Services, na.rm = TRUE)
data_dengue$House <- zscore(data_dengue$House, na.rm = TRUE)
data_dengue$Overcrowding <- zscore(data_dengue$Overcrowding, na.rm = TRUE)
data_dengue$Urban <- zscore(data_dengue$Urban, na.rm = TRUE)
data_dengue$Ethnic <- zscore(data_dengue$Ethnic, na.rm = TRUE)
data_dengue$Misery <- zscore(data_dengue$Misery, na.rm = TRUE)

#DAG 
dag <- dagitty('dag {
sir [pos="0, 0.5"]
HFP  [pos="-0.2, 0.5"]

House [pos="-1.6, 1.1"]
Ethnic [pos="-0.4, -0.40"]
Services [pos="-0.30, 1.2"]
Overcrowding [pos="-1.1, -1.4"]
Misery [pos="-1.1, 1.30"]
Urban [pos="-1.4, -0.90"]

House -> HFP
House -> sir
Ethnic -> HFP
Ethnic -> sir
Services -> HFP
Services -> sir
Overcrowding -> HFP
Overcrowding -> sir
Urban -> HFP
Urban -> sir
Misery -> HFP
Misery -> sir

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

HFP -> sir
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

#identification
simple_dag <- dagify(
  sir ~  HFP + Services + House + Overcrowding + Urban +  Ethnic + Misery,
  HFP ~ Services + House + Overcrowding + Urban + Ethnic + Misery, 
  Services ~ House + Ethnic + Urban, 
  Misery ~ House + Services + Ethnic + Overcrowding + Urban,
  Overcrowding ~  House  + Urban,
  Urban ~ House + Ethnic +  Misery + Services,
  Ethnic ~ House,
  exposure = "HFP",
  outcome = "sir",
  coords = list(x = c(HFP=2, sir=2, House=3.5, Services=-1.12, Overcrowding=3.2, Urban=-1.0, Ethnic=1.2, Misery=-0.8),
                y = c(HFP=1.8, sir=1, House=3.5, Services=2.0, Overcrowding=2.2, Urban=3.3, Ethnic=3.2, Misery=2.5))
)

# theme_dag() coloca la trama en un fondo blanco sin etiquetas en los ejes
ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()

#adjusting
adjustmentSets(simple_dag,  type = "minimal")
## {z_miseria, indixes}

ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()


##############################################################################
# For visceral leishmaniasis
data <- read.csv("D:/visceral_final.csv") # Modify the path if necessary

data_visceral <- select(data, sir, HFP, Coca, Forest, Mining, Fire, Deforest, rMisery,
                        Services, House, Overcrowding, Urban, Ethnic, uMisery)

#sd units
data_visceral$sir <- zscore(data_visceral$sir, na.rm = TRUE)
data_visceral$HFP <- zscore(data_visceral$HFP, na.rm = TRUE)
data_visceral$Coca <- zscore(data_visceral$Coca, na.rm = TRUE)
data_visceral$Forest <- zscore(data_visceral$Forest, na.rm = TRUE)
data_visceral$Mining <- zscore(data_visceral$Mining, na.rm = TRUE)
data_visceral$Fire <- zscore(data_visceral$Fire, na.rm = TRUE)
data_visceral$Deforest <- zscore(data_visceral$Deforest, na.rm = TRUE)
data_visceral$rMisery <- zscore(data_visceral$rMisery, na.rm = TRUE)
data_visceral$sir <- zscore(data_visceral$sir, na.rm = TRUE)
data_visceral$HFP <- zscore(data_visceral$HFP, na.rm = TRUE)
data_visceral$Services <- zscore(data_visceral$Services, na.rm = TRUE)
data_visceral$House <- zscore(data_visceral$House, na.rm = TRUE)
data_visceral$Overcrowding <- zscore(data_visceral$Overcrowding, na.rm = TRUE)
data_visceral$Urban <- zscore(data_visceral$Urban, na.rm = TRUE)
data_visceral$Ethnic <- zscore(data_visceral$Ethnic, na.rm = TRUE)
data_visceral$uMisery <- zscore(data_visceral$uMisery, na.rm = TRUE)

#DAG 
dag <- dagitty('dag {
sir [pos="0, 0.5"]
HFP  [pos="-0.2, 0.5"]

Forest [pos="-1.6, 1.1"]
Deforest [pos="-0.20, -0.40"]
Coca [pos="-0.30, 1.2"]
Mining [pos="-1.1, -1.4"]
rMisery [pos="-1.1, 1.30"]
Fire [pos="-1.4, -0.90"]
House [pos="-2.6, 2.1"]
Ethnic [pos="-0.4, -0.40"]
Services [pos="-0.9, 1.8"]
Overcrowding [pos="-1.3, -1.7"]
uMisery [pos="-1.5, 1.8"]
Urban [pos="-1.8, -1.70"]

Forest -> HFP
Forest -> sir
Deforest -> HFP
Deforest -> sir
Coca -> HFP
Coca -> sir
Mining -> HFP
Mining -> sir
Fire -> HFP
Fire -> sir
rMisery -> HFP
rMisery -> sir

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
House -> sir
Ethnic -> HFP
Ethnic -> sir
Services -> HFP
Services -> sir
Overcrowding -> HFP
Overcrowding -> sir
Urban -> HFP
Urban -> sir
uMisery -> HFP
uMisery -> sir

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

HFP -> sir
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

#identification
simple_dag <- dagify(
  sir ~  HFP + Services + House + Overcrowding + Urban +  Ethnic + uMisery + Coca + Forest + Mining + Fire +  Deforest + rMisery,
  HFP ~ Services + House + Overcrowding + Urban +  Ethnic + uMisery + Coca + Forest + Mining + Fire +  Deforest + rMisery, 
  Services ~ House + Ethnic + Urban + rMisery + Mining, 
  uMisery ~ House + Services + Ethnic + Overcrowding + Urban + Deforest + rMisery + Coca + Fire + Mining,
  Overcrowding ~  House  + Urban,
  Urban ~ House + Ethnic +  uMisery + Services + Mining,
  Ethnic ~ House + rMisery,
  Coca ~ Forest + Deforest + rMisery + Overcrowding, 
  Forest ~ Mining + Ethnic + Deforest,
  Mining ~ rMisery + Coca + Fire,
  Fire ~ Forest + Deforest +  rMisery + Coca,
  Deforest ~ Forest + rMisery + Mining,
  rMisery ~ Forest + Ethnic,
  exposure = "HFP",
  outcome = "sir",
  coords = list(x = c(HFP=2, sir=2, House=3.5, Services=-1.12, Overcrowding=3.2, Urban=-1.0, Ethnic=1.2, uMisery=-0.8,
                      Forest=4.0, Coca=-1.62, Mining=3.7, Fire=-1.5, Deforest=1.7, rMisery=-1.3),
                y = c(HFP=1.8, sir=1, House=3.5, Services=2.0, Overcrowding=2.2, Urban=3.3, Ethnic=3.2, uMisery=2.5,
                      Forest=4.0, Coca=2.5, Mining=2.8, Fire=3.8, Deforest=3.7, rMisery=3.0))
)

# theme_dag() coloca la trama en un fondo blanco sin etiquetas en los ejes
ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()

#adjusting
adjustmentSets(simple_dag,  type = "minimal")
## {z_miseria, indixes}

ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()






