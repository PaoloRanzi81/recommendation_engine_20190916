"""
TITLE: "Outliers detection and their elimination by Isolation Forest"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

Outlier detection has been run following the sequence: Principal Component Analysis
(PCA) + Independent Component Analysis (ICA) + Isolation Forest (IF).
The combination of PCA + ICA seem to improve clustering/classification results by
reducing noise. It does not work for regression, though. This paper has been here implemented: 
    
Gultepe, E., & Makrehchi, M. (2018). Improving clustering performance using 
independent component analysis and unsupervised feature learning. 
Human-centric Computing and Information Sciences, 8(1), 25.

DESCRIPTION: 
The script is parallelized by using 'joblib' Python library. Please set 'RELEASE' to 
your local system specifics if you would like to use the script by a single-core mode.
By default the script works by a multi-core/multi-threaded fashion. 
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'
    - there are the PCA + ICA + IF parameters which should be modified according to 
    the specific data-set's needs when calling these very functions; 

"""


###############################################################################
## IMPORTING LIBRARIES
# import required Python libraries
import platform
import os
import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from multiprocessing import cpu_count
import argparse
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import FastICA
from sklearn.ensemble import IsolationForest
import time

                   
###############################################################################
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
input_file_name_1 = ('0.csv')
#input_file_name_2 = ('iiiii_relative_engagement.csv')
#output_file_name_1 = ('video_more_than_once_whole.csv')
output_file_name_2 = ('re_scores_clean.csv')
parallel = True # whenever starting the script from the Linux bash, uncomment such variable


###############################################################################
# set whether use parallel computing (parallel = True) or 
# single-core computing (parallel = False).
 # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parallel",  dest='parallel', action='store_true',
	help="# enable multi-core computation")
ap.add_argument("-no-p", "--no-parallel",  dest='parallel', action='store_false',
	help="# disable multi-core computation")
args = vars(ap.parse_args())

# grab the "parallel" statment and store it in a convenience variable
# you have to set it from the command line
parallel = args["parallel"] 

# start clocking time
start_time = time.time()

"""
###############################################################################
## DOWNSCALING FROM MULTI-THREAD TO SINGLE-THREAD PARALLELIZATION (ONLY WHEN DEBUGGING)
# reduce the number of threads used by each CPU by intervening on OpenBLAS. 
# In order to avoid multi-threading (thus sucking all server's CPU power) run 
# the following command before importing Numpy. Lastly set n_jobs to a small 
# number (thus freeing up resources, but slowing down computation).
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
#os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
#np.show_config()
"""

###############################################################################
## LOADING
#loading the .csv files
re_scores_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_1]), header = 0, 
    dtype = {'session_id': 'str', 'event.type': 'str'})
    
# rename    
re_scores_tmp.rename(columns={ 'vid': 'video_id'}, inplace = True)

# fill NaN with '0'
re_scores_tmp.fillna(0, inplace = True)

# copy original DataFrame (it will become useful later)
original_df = re_scores_tmp.copy()

# drop columns
re_scores_tmp.drop(columns = ['user_id', 'session_id', 'video_id', 
                              'start', 'end', 'event.ts', 'event.vidTime' ], inplace = True)

# It seems Gradient Boosting is NOT able to handle One-Hot Encoding    
le = LabelEncoder()
re_scores_tmp['event.type'] = le.fit_transform(re_scores_tmp['event.type'])


###############################################################################
## FUNCTION: GETTING RID OF OUTLIERS BY USING PCA + ICA + IF
def deleting_outliers(chunks_df_tmp):    
    ###############################################################################
    ## KERNEL PRINCIPAL COMPONENT ANALYSIS (PCA)
    # initialize Kernel PCA parameters: 
    n_components = 2
    kernel = 'linear'
    #kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine', 'precomputed']
    gamma = 1/4
    degree = 1
    coef0 = 1
    kernel_params = None
    alpha = 1.0
    fit_inverse_transform = False
    eigen_solver = 'dense'
    #eigen_solver = ['auto', 'arpack','dense']
    tol = 0
    max_iter = 30000
    #max_iter = 3
    remove_zero_eig = True
    random_state = None
    copy_X = True
    n_jobs = int(round((cpu_count() - 1), 0))
    
    # compute Kernel PCA
    kernel_pca = KernelPCA(n_components = n_components, 
              kernel = kernel, 
              gamma = gamma, 
              degree = degree, 
              coef0 = coef0, 
              kernel_params = kernel_params,
              alpha = alpha,
              fit_inverse_transform  = fit_inverse_transform ,
              eigen_solver = eigen_solver,
              tol = tol,
              max_iter  = max_iter,
              remove_zero_eig  = remove_zero_eig,
              random_state = random_state, 
              copy_X  = copy_X ,
              n_jobs = n_jobs,)
    
    # fit model
    #Y =[]
    Y_pca = kernel_pca.fit_transform(chunks_df_tmp)
    
     
    ###############################################################################
    ## INDEPENDENT COMPONENT ANALYSIS (ICA) 
    # initialize FastICA parameters: 
    n_components = 2
    algorithm  = 'parallel'
    #algorithm  = ['parallel', 'deflation']
    whiten = True
    fun = 'logcosh'
    #fun  = ['logcosh', 'exp', 'cube']
    fun_args = {'alpha' : 1.0}
    max_iter = 30000
    #max_iter = 3
    tol = 1e-06
    w_init = None
    random_state = None
    
    
    # initialize ICA model
    fast_ICA = FastICA(n_components = n_components, 
              algorithm = algorithm, 
              whiten = whiten, 
              fun = fun, 
              fun_args = fun_args, 
              max_iter = max_iter,
              tol = tol, 
              w_init = w_init,
              random_state = random_state)
    
    
    # fit model
    #Y =[]
    Y_ica = fast_ICA.fit_transform(Y_pca)
    #Y = fast_ICA.fit_transform(chunks_df_tmp)    
    
    
    ###############################################################################
    ## 2.8. ISOLATION FOREST (IF) 
    # initialize Isolation Forest parameters: 
    n_estimators = 10000
    #n_estimators = 2
    max_samples = 1.0
    #max_samples = 'auto'
    #contamination = 0.9
    contamination = 'auto'
    max_features = 1.0
    bootstrap = True
    n_jobs = int(round((cpu_count() - 1), 0))
    behaviour ='new' 
    random_state = None
    verbose = 0 
    warm_start = True    
      
    
    # initialize classifier
    clf = IsolationForest(n_estimators = n_estimators, 
                          max_samples = max_samples,
                          contamination = contamination,
                          max_features = max_features, 
                          bootstrap = bootstrap,
                          n_jobs = n_jobs,
                          behaviour = behaviour,                                      
                          random_state = random_state,
                          verbose = verbose, 
                          warm_start = warm_start)
    
    # fit 
    clf.fit(Y_ica)
    
    # label outliers by '-1' value
    y_pred = pd.DataFrame(clf.predict(Y_ica))    
    
    # rename column
    y_pred = y_pred.rename(columns={0: 'outliers'}, inplace = False)
    
    # set index according to pre-chopped DataFrame
    y_pred.set_index(chunks_df_tmp.index, inplace = True) 

    # drop outliers 
    chunks_df_tmp_cleaned = chunks_df_tmp.drop(index = y_pred.loc[y_pred['outliers'] == -1].index)
    
    return (chunks_df_tmp_cleaned)


###############################################################################
## SPLIT DATA IN CHUNKS (GOAL: AVOID MEMORY ERROR ISSUES)
# in order to try to put all the same video_ids within the same chuck. Otherwise
# we would have problem in summing the variables' values. Indeed, we will have
# lots of duplicates in the final aggregated DataFrame.
# Since memory issue probelm by PCA, it is important to split the data in chunk 
# of ~ 15000 rows  

# test pipeline on a small sample    
#re_scores_tmp = re_scores_tmp.loc[0:100,:].copy()       

# for huge DataFrames only
if re_scores_tmp.shape[0] > 15000: 
    
    # sort values in order to decrease data loss by splitting
    tot_tmp = re_scores_tmp.copy()
    
    # divide in chunks of 15000 rows each
    chunks_df = np.array_split(re_scores_tmp, int(round((re_scores_tmp.shape[0]/15000), 0)))
    
    # initialize DataFrame
    cleaned_df_tmp = pd.DataFrame()
    
    for chunk in range(0, len(chunks_df)):
        
        # initialize chunk
        chunks_df_tmp = pd.DataFrame()
        chunks_df_tmp = chunks_df[chunk] 
        
        # call function
        chunks_df_tmp_cleaned = deleting_outliers(chunks_df_tmp)
        
        # reconstruct original DataFrame by appending     
        cleaned_df_tmp  = cleaned_df_tmp.append(chunks_df_tmp_cleaned)
          
    # drop outliers
    cleaned_df = original_df.iloc[cleaned_df_tmp.index, :]
        
    # save DataFrame as .csv file
    cleaned_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2]), index= False) 

else:
    
    # copy 
    chunks_df_tmp = re_scores_tmp.copy()
    
    # call function
    chunks_df_tmp = deleting_outliers(chunks_df_tmp)
    
    # drop outliers
    cleaned_df = original_df.iloc[chunks_df_tmp.index, :]
    
    # save DataFrame as .csv file
    cleaned_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2]), index= False) 


# shows execution time
print( time.time() - start_time, "seconds")





"""    
###############################################################################
## HDBSCAN (NOT MANDATORY)
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=criteria.shape[0],
                            metric = 'euclidean',
                            p = None, 
                            cluster_selection_method = 'eom', 
                            core_dist_n_jobs = int(round((cpu_count() - 1), 0)), 
                            approx_min_span_tree  = False).fit(Y_ica)

# plot outliers
#sns.distplot(clusterer.outlier_scores_[np.isfinite(clusterer.outlier_scores_)], rug=True)

# delete outliers from original dataset
threshold = pd.Series(clusterer.outlier_scores_).quantile(0.95)
outliers = np.where(clusterer.outlier_scores_ > threshold)[0]

# drop outliers 
merged_df_small = merged_df_small_tmp.drop(index = outliers)

# reset index
merged_df_small_tmp.reset_index(drop=True, inplace=True)

# re-run kernel PCA on the new dataset
Y_pca =[]
Y_pca = kernel_pca.fit_transform(merged_df_small_tmp)

    
# re-run ICA on the new dataset    
# fit model
Y_ica =[]
Y_ica = fast_ICA.fit_transform(Y_pca)    
""" 

"""
# BRUTE FORCE METHOD: delete outliers (2.5 % at each tail) about 'vid_duration' 
threshold_high = merged_df_small_tmp.loc[:, 'vid_duration'].quantile(0.95)
threshold_low = merged_df_small_tmp.loc[:, 'vid_duration'].quantile(0.05)
merged_df_small_tmp = merged_df_small_tmp.loc[(merged_df_small_tmp.loc[:, 'vid_duration'] > threshold_low) & (merged_df_small_tmp.loc[:, 'vid_duration'] < threshold_high)]
"""  
