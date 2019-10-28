"""
TITLE: "Bootstrapping + Ordinary Least Squares (OLS)"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
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
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from arch.bootstrap import IIDBootstrap
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
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

# start clocking time
start_time = time.time()


###############################################################################
## LOADING DATA-SET 

#loading the .csv file 
raw_data_frame = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_1]), header = 0) 
  
# delete 'videos_id' already present in the first DataFrame
#mask = relative_engagement.loc[:, ['video_id']].ne(video_id_list)


"""  
# standardized strictly positive values
raw_data_frame.loc[:, ['total_views', 'vid_duration']] = PowerTransformer(method='box-cox',).fit_transform\
(raw_data_frame.loc[:, ['total_views', 'vid_duration']]) 
"""
"""
# standardized categorical variable (e.g. 'liked', 'disliked' etc.)
raw_data_frame.loc[:, ['liked', 'disliked', 'favorite', 'full_screen' ]] = MinMaxScaler().fit_transform\
(raw_data_frame.loc[:, ['liked', 'disliked', 'favorite', 'full_screen']]) 
"""


###############################################################################
## SIMPLE DATA-SET DIAGNOSTICS      
# check summary statistics
#summary_stats = raw_data_frame.describe()


###############################################################################
## BOOTSTRAPPED OLS 

# call statsmodels OLS
def ols_stats(input_data):
    return smf.ols('RE ~ liked', data = input_data).fit().pvalues
    #return smf.ols('RE ~ liked', data = input_data).fit().tvalues
    #return smf.ols('RE ~ liked', data = input_data).fit().params


# run bootstrapping by arch
input_data = raw_data_frame
bs = IIDBootstrap(input_data)
ci = pd.DataFrame(data = (bs.conf_int(ols_stats, 10000, method = 'basic', tail = 'two')), 
                          columns = ['intercept_ci', 'first_variable_ci'], 
                          index = ['lower', 'upper'])
mean_ci = ci.mean(axis = 0)
ci = ci.append(mean_ci, ignore_index = True)
ci = ci.T.copy()
print(ci)
print(mean_ci, "summary_statistics: mean_ci")


# computing p-values manually from t-values
#pval = stats.t.sf(np.abs(4.4), n-1)*2  # two-sided pvalue = Prob(abs(t)>tt)
#print 't-statistic = %6.3f pvalue = %6.4f' % (tt, pval)
 
# shows execution time
print( time.time() - start_time, "seconds")

