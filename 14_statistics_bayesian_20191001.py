"""
TITLE: "Bayesian + MCMC"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
The results of the script will be similar to the ones obtained by using bootstrapping
+ OLS technique. Anyway, it is always nice and reassuiring to confirm twice the same results.
    The script is already parallelized intrisically by using 'arch' Python library. 
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'

"""

###############################################################################
## IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import argparse
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time


## SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.4.0-143-generic').
RELEASE = platform.release()

if RELEASE == '5.0.0-32-generic': # Linux laptop
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/recommendation_engine/raw_data')
   BASE_DIR_OUTPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/recommendation_engine/outputs')
  
else:
   BASE_DIR_INPUT = ('/home/ubuntu/raw_data')
   BASE_DIR_OUTPUT = ('/home/ubuntu/outputs')

   
###############################################################################
## PARAMETERS TO BE SET!!!
input_file_name_1 = ('video_more_than_once_small.csv') 
parallel = True # whenever starting the script from the Linux bash, uncomment such variable

# setting PyMC3 parameters
#draws = 100000 # ideal: 440000 draws
draws = 1000
chains = 2
tune = 900
#tune = (draws*90)/100 # ideal: 90 % burn-in
cores = 1


###############################################################################
# set whether use parallel computing (parallel = True) or 
# single-core computing (parallel = False).
 # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parallel",  dest = 'parallel', action = 'store_true',
	help="# enable multi-core computation")
ap.add_argument("-no-p", "--no-parallel",  dest = 'parallel', action = 'store_false',
	help="# disable multi-core computation")
args = vars(ap.parse_args())

# grab the "parallel" statment and store it in a convenience variable
# you have to set it from the command line
parallel = args["parallel"] 

# start clocking time
start_time = time.time()


###############################################################################
## LOADING DATA-SET 

#loading the .csv file 
X = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_1]), header = 0) 
# summary_stats
#summary_stats = X.describe()
  
# standardized strictly positive values (necessary step, otherwise PyMC3 throws errors)
#X.loc[:, ['total_views', 'vid_duration']] = PowerTransformer(method='box-cox').fit_transform\
#(X.loc[:, ['total_views', 'vid_duration']]) 

# standardized categorical variable (e.g. 'liked', 'disliked' etc.)
X.loc[:, ['liked', 'disliked', 'favorite', 'full_screen', 'total_views', 'vid_duration' ]] = StandardScaler().fit_transform\
(X.loc[:, ['liked', 'disliked', 'favorite', 'full_screen', 'total_views', 'vid_duration']]) 
#
# standardized categorical variable (e.g. 'liked', 'disliked' etc.)
#X.loc[:, ['liked', 'disliked', 'favorite', 'full_screen' ]] = MinMaxScaler().fit_transform\
#(X.loc[:, ['liked', 'disliked', 'favorite', 'full_screen']]) 

    
###############################################################################
## BAYESIAN + MCMC LINEAR REGRESSION  

# linear regression (OLS) formula
# liked + full_screen
formula = 'RE ~ liked ' 

# split data into train and test sets [STANDARD CROSS-VALIDATION DOES NOT WORK
# WITHIN THE BAYESIAN FRAMEWORK. PLEASE USE Widely-applicable Information Criterion (WAIC)
INSTEAD]
#X_train, X_test= train_test_split(
#        X, test_size = 0.3, random_state = None,
#        shuffle = True, stratify = None)

# change name accordingly
X_train = X

# Context for the model
with pm.Model() as normal_model:
    
    # The prior for the data likelihood is a Normal Distribution
    family = pm.glm.families.Normal()
    
    # Creating the model requires a formula and data (and optionally a family)
    pm.GLM.from_formula(formula, data = X_train, family = family)
    
    # Perform Markov Chain Monte Carlo sampling letting PyMC3 choose the algorithm
    normal_trace = pm.sample(draws = draws, chains = chains, tune = tune, cores = cores)
    
# show traces
#pm.traceplot(normal_trace)    

# show traces    
#pm.plot_posterior(normal_trace)  

# summary statistics
summary_statistics_trace = pm.summary(normal_trace)  

#ppc = pm.sample_posterior_predictive(normal_trace, model = normal_model, samples=100)  

 
# shows execution time
print( time.time() - start_time, "seconds")
    
    
