"""
TITLE: "Gradient boosting regressor grid search "
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION:
    When setting analysis = 'grid_search' it will run a grid search on specific 
hyperparameters' ranges (please select the hyperparamenters and their parameters
to be tried). Once the best hyperparameters have been found, use analysis = 'bootstrapping'. 
Wiht 'bootstrapping' a 30 runs will be run in order to see how the cross-validation
oscillates. Than the median of the 30 runs' scores will be takes. Such a median is 
the real cross-validation score.  
    The script is parallelized by using 'joblib' Python library. Please set 'RELEASE' to 
your local system specifics if you would like to use the script by a single-core mode.
By default the script works by a multi-core/multi-threaded fashion. 
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'

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
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import  KFold
#from sklearn.metrics import f1_score, matthews_corrcoef, log_loss 
from sklearn.metrics import mean_absolute_error 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

                   
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
input_file_name_1 = ('iiiii_relative_engagement_quantiles.csv')
input_file_name_2 = ('0.csv')
output_file_name_1 = ('cv_results.csv')
output_file_name_2 = ('best_scores.csv')
parallel = True # whenever starting the script from the Linux bash, uncomment such variable
analysis = 'bootstrapping' # whenever starting the script from the Linux bash, uncomment such variable
## WARNING: set also 're_scores_tmp' variable approriately; 


###############################################################################
# set whether use parallel computing (parallel = True) or 
# single-core computing (parallel = False).
 # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--parallel",  dest='parallel', action='store_true',
	help="# enable multi-core computation")
ap.add_argument("-no-p", "--no-parallel",  dest='parallel', action='store_false',
	help="# disable multi-core computation")
ap.add_argument("-a", "--analysis", type=str, default='bootstrapping',
	help="# type analysis's name") 
args = vars(ap.parse_args())

# grab the analysis you want to run. You have to write down the analysis name
# in the command line as done for all 'argparse' arguments.
analysis = args["analysis"] 

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
re_scores = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_1]), header = 0)
   
clean_df = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_2]), header = 0)

# drop columns
re_scores.drop(columns = ['median_sec_watched', 'vid_duration', 'total_views',
       'full_screen', 'average_watch_percentage', 'vid_duration_standardized',
       'quantile_rank'], inplace = True)
    
# rename
if 'vid' in clean_df.columns:
    clean_df.rename(columns={'vid': 'video_id'}, inplace = True)       

# fill NaN with '0'
clean_df.fillna(0, inplace = True)
    
# merge 'RE' scores
merged_df = pd.merge(clean_df, re_scores, how ='inner', 
                     on = 'video_id')  

# fix names
re_scores_tmp = merged_df.copy()

# save merged_df DataFrame as .csv file
#re_scores_tmp.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
#                                     ('set_' + input_file_name_2)]), index= False)
 
# pop out 'RE'
y = re_scores_tmp.pop('RE')    


###############################################################################
## PREPARING DATA FOR ML
# It seems Gradient Boosting is NOT able to handle One-Hot Encoding    
le = LabelEncoder()
re_scores_tmp['user_id'] = le.fit_transform(re_scores_tmp['user_id'])
re_scores_tmp['session_id'] = le.fit_transform(re_scores_tmp['session_id'])
re_scores_tmp['event.type'] = le.fit_transform(re_scores_tmp['event.type'])
re_scores_tmp['video_id'] = le.fit_transform(re_scores_tmp['video_id'])

# OneHotEncoding for 'video_id' vector
#t = pd.get_dummies(re_scores_tmp.loc[:, 'video_id'])

# create Numpy array to be fed to sklearn [not necessary anymore. scikit-learn
# not is accepting Pandas DataFrames, as well.]
#X = re_scores_tmp.to_numpy()

# copy DataFrame
X = re_scores_tmp.copy()


##############################################################################
## GRADIENT BOOSTING WITH BOOTSTRAPPING (when the range of optimal 
## parameters has been already found)
if analysis == 'bootstrapping':
    # configure bootstrap
    n_iterations = 30
    best_score_df = pd.DataFrame()
    
    # set grid search's parameters
    model = GradientBoostingRegressor()
    param_grid = {'loss' : ['lad'],
                  #'learning_rate' : np.linspace(0.1, 1, 10, dtype = float),
                  'learning_rate' : [0.2],                  
                  'n_estimators' : [5000],
                  #'n_estimators' : [50],
                  'subsample' : [0.9],
                  #'subsample' : np.linspace(0.1, 1, 10, dtype = float),
                  'criterion' : ['friedman_mse'],
                  #'min_samples_split' : np.linspace(2, 30, 5, dtype = int),
                  'min_samples_split' : [16], # insignificant difference BTW...?!?
                  #'min_samples_leaf' : [1, 2, 3, 4],
                  'min_samples_leaf' : [3],
                  #'min_weight_fraction_leaf' : np.linspace(0.1, 0.4, 4, dtype = float),
                  'min_weight_fraction_leaf' : [0.1], 
                  #'max_depth' : np.linspace(1, 50, 5, dtype = int),
                  'max_depth' : [50], # insignificant difference BTW...?!?
                  #'min_impurity_decrease' : np.linspace(0.1, 0.6, 6, dtype = float),
                  'min_impurity_decrease' : [0.6], 
                  #'min_impurity_split' : [None],
                  'init' : [None],
                  'random_state' : [None],
                  'max_features' : ['log2'],
                  #'max_features' : ['auto', 'log2' ],
                  'verbose' : [0],
                  'max_leaf_nodes' : [None],
                  'warm_start' : [True],
                  #'warm_start' : [False, True],
                  #'presort ' : ['auto'],
                  #'presort ' : ['auto', True],
                  'validation_fraction' : [0.1],
                  'n_iter_no_change' : [None],
                  'tol' : [0.0001],
                  }

    # function for computing grid search    
    def grid_search(model, param_grid, X, y):    
        
        # grid search
        if RELEASE == '5.0.0-31-generic': # Linux laptop
            # set cross-validation  
            cv = KFold(n_splits = int(round((cpu_count() - 1), 0)), shuffle = True,                         
                                    random_state = None)

            # split data-set without 'stratify' option
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                                random_state=None)
            # single-core computation
            grid = GridSearchCV(estimator = model, param_grid = param_grid,
             iid = True, cv = cv, n_jobs = 1, refit='mean_absolute_error')
        
        else:
            # set cross-validation  
            # WARNING: for parellelizing code use as many n_splits as cpus 
            cv = KFold(n_splits = int(round((cpu_count() - 1), 0)), shuffle = True,                         
                                    random_state = None)

            # split data-set without 'stratify' option
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                                random_state=None)
            # use multi-cores available    
            grid = GridSearchCV(estimator = model, param_grid = param_grid,
                     iid = True, cv = cv, n_jobs = int(round((cpu_count() - 1), 0)), 
                     refit='mean_absolute_error')
   
        # fit model
        grid_result = grid.fit(X_train, y_train)

        return (grid_result)
    
        
    # running either parallel or single-core computation. 
    if parallel:
    	# execute configs in parallel
        executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        tasks = (delayed(grid_search)(model, param_grid, X, 
                 y) for i in range(n_iterations))
        output = executor(tasks)
                
    else:
        output = [grid_search(model, param_grid, X, 
                              y) for i in range(n_iterations)]        
 
    # append output from 'joblib' in lists and DataFrames    
    stats = []
    best_score = []
    best_parameters = []
    results = pd.DataFrame() 
    
    # collect and save all CV results for plotting
    for counter in range(0,len(output)):
        
        # collect cross-validation results (e.g. multiple metrics etc.)
        results_1 = pd.DataFrame.from_dict(output[counter].cv_results_)
        results = results.append(results_1, ignore_index=True) 
        
        # collect best_scores    
        best_score_series = pd.Series(np.round(output[counter].best_score_, 3))
        best_score_df = best_score_df.append(best_score_series, 
                                             ignore_index = True, sort = False)
        best_parameters.append(output[counter].best_params_)

    # print
    print('Number iterations: %.0f' % (len(output)))


##############################################################################
## GRADIENT BOOSTING GRID SEARCH (when the range of optimal 
## parameters has to be found)
if analysis == 'grid_search': 
    # configure bootstrap
    n_iterations = 30
    #n_size = int(len(data) * 0.50)
    stats = []
    best_score_df = pd.DataFrame()
    best_parameters_df = pd.DataFrame()
    #best_score = []
    best_parameters = []
    results = pd.DataFrame()
    
    
    for i in range(n_iterations):
    
        # split data-set with 'stratify' option
        #X_train, X_test, y_train, y_test = train_test_split(
        #        X, y, test_size=0.2, random_state=None, stratify = y)
        
        ## split data-set without 'stratify' option
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size = 0.3, random_state=None)    

        # set grid search's parameters
        model = GradientBoostingRegressor()
        param_grid = {'loss' : ['lad'],
                      #'learning_rate' : np.linspace(0.1, 1, 10, dtype = float),
                      'learning_rate' : [0.2],                  
                      'n_estimators' : [5000],
                      #'n_estimators' : [50],
                      'subsample' : [0.9],
                      #'subsample' : np.linspace(0.1, 1, 10, dtype = float),
                      'criterion' : ['friedman_mse'],
                      #'min_samples_split' : np.linspace(2, 30, 5, dtype = int),
                      'min_samples_split' : [16], # insignificant difference BTW...?!?
                      #'min_samples_leaf' : [1, 2, 3, 4],
                      'min_samples_leaf' : [3],
                      #'min_weight_fraction_leaf' : np.linspace(0.1, 0.4, 4, dtype = float),
                      'min_weight_fraction_leaf' : [0.1], 
                      #'max_depth' : np.linspace(1, 50, 5, dtype = int),
                      'max_depth' : [50], # insignificant difference BTW...?!?
                      #'min_impurity_decrease' : np.linspace(0.1, 0.6, 6, dtype = float),
                      'min_impurity_decrease' : [0.6], 
                      #'min_impurity_split' : [None],
                      'init' : [None],
                      'random_state' : [None],
                      'max_features' : ['log2'],
                      #'max_features' : ['auto', 'log2' ],
                      'verbose' : [0],
                      'max_leaf_nodes' : [None],
                      'warm_start' : [True],
                      #'warm_start' : [False, True],
                      #'presort ' : ['auto'],
                      #'presort ' : ['auto', True],
                      'validation_fraction' : [0.1],
                      'n_iter_no_change' : [None],
                      'tol' : [0.0001],
                      }
           
        # set cross-validation  
        cv = KFold(n_splits = int(round((cpu_count() - 1), 0)), shuffle = True,                         
                                    random_state = None)

        scoring = mean_absolute_error
  
        # NOT WORKING: in order to avoid the warning the following could be implemented 
        # 
        #scoring = f1_score(y_test, y_pred, average='weighted', 
        #                 labels=np.unique(y_pred))
        
        # grid search
        if RELEASE == '5.0.0-31-generic': # Linux laptop
            # single-core computation
        	grid = GridSearchCV(estimator = model, param_grid = param_grid,
                 iid = True, cv = cv, n_jobs = 1,
                 refit='mean_absolute_error')
    
        else:
            # use multi-cores available    
            grid = GridSearchCV(estimator = model, param_grid = param_grid,
                         iid = True, cv = cv,
                         n_jobs = int(round((cpu_count() - 1), 0)), 
                         refit='mean_absolute_error')
        
        # fit model
        grid_result = grid.fit(X_train, y_train) 

        # collect and save all CV results for plotting
        results_tmp = pd.DataFrame.from_dict(grid_result.cv_results_)
        results = results.append(results_tmp, ignore_index = True) 
    
        # collect best_scores   
        best_score_series = pd.Series(np.round(grid_result.best_score_, 3))
        best_score_df = best_score_df.append(best_score_series, ignore_index = True, sort = False)
        best_parameters.append(grid_result.best_params_)
        #best_parameters_df_tmp = pd.DataFrame.from_dict(grid_result.best_params_, orient = 'index', columns = ['scores']).reset_index()
        #best_parameters_df =  best_parameters_df_tmp.pivot(values = 'scores', columns='index')
    
        # print
        print(grid_result.best_score_)
        print(grid_result.best_params_)
        print('Number iterations %.0f' % (i))


###############################################################################
## GENERATE OUTPUT (.csv files and .jpg pictures)

#feature_importances=grid_search.best_estimator_.feature_importances_
#cvres=grid_search.cv_results_   
#for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):    
#    print(np.sqrt(-mean_score), params)
    
   
###############################################################################
## SAVE BEST SCORES IN A PANDAS DATAFRAME AND PLOT THEIR BOOTSTRAPPING 
## DISTRIBUTION 

# save cleaned DataFrame as .csv file
results.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_1]), index= False) 

# rename
best_score_df.rename(columns={0: 'best_score'}, inplace = True)    

# concatenate results
#best_score_df = pd.DataFrame(best_score, columns=['best_score'])
best_parameters_df = pd.DataFrame(best_parameters)
summary_table = pd.concat([best_score_df, best_parameters_df], axis = 1)

# save cleaned DataFrame as .csv file
summary_table.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_2]), index= False)
    
# in case you want to load the .csv with the best scores
#summary_table = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
#                                     output_file_name_2]))    
    
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(summary_table.loc[:,'best_score'], p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(summary_table.loc[:,'best_score'], p))
median = np.median(summary_table.loc[:,'best_score'])
median_parameters = summary_table.loc[summary_table['best_score'] == (round(median, 2))]
print('%.1f confidence interval %.4f and %.4f' % (alpha*100, lower, 
                                                      upper))
print('Median %.4f' % (median))
print('Below best score (median) and parameters ')
print(median_parameters)

    
"""
# TEST: plot scores and save plot
date = str(datetime.datetime.now())
sns_plot = sns.distplot(best_score_df, bins = 30)
#sns_plot = sns.distplot(best_score_df, bins = (len(best_score_df)/100))
fig = sns_plot.get_figure()
fig.savefig(os.path.sep.join([BASE_DIR, date[0:10]+ "_" + date[11:16]+".jpg"])) 
 """ 

# shows execution time
print( time.time() - start_time, "seconds")







