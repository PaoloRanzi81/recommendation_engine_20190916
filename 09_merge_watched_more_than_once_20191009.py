"""
TITLE: "Merge session_id and video_id when video_ids watched more than once
within the same session_id"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
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
input_file_name_1 = ('list_of_double_watched_video_id.csv')
input_file_name_2 = ('iiiii_relative_engagement_quantiles.csv')
output_file_name_1 = ('video_more_than_once_whole.csv')
output_file_name_2 = ('video_more_than_once_small.csv')
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


###############################################################################
## LOADING
#loading the .csv files
double_watched_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_1]), header = 0, 
    dtype ={'vid' : str})
   
# rename column    
double_watched_tmp = double_watched_tmp.rename(columns={ 'vid': 'video_id'}, 
                                         inplace = False)    
    
# drop duplicate column
double_watched_tmp.drop(columns = ['session_id'], inplace = True)

# unique video_id
double_watched = pd.Series(pd.unique(double_watched_tmp.loc[ : , 'video_id']))

# rename column    
double_watched = double_watched.rename('video_id', inplace = False)


# test pipeline on a small sample    
#double_watched = double_watched.loc[:100].copy() 

# load
relative_engagement = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_2]), header = 0, 
    dtype ={'video_id' : str})


# drop duplicate column
if 'quantiles' in relative_engagement.columns: 
    relative_engagement.drop(columns = ['vid_duration_standardized', 'quantiles' ], inplace = True)
else: 
    relative_engagement.drop(columns = ['vid_duration_standardized'], inplace = True)    



# merge 'median_sec_watched' with 
merged_df = pd.merge(double_watched, relative_engagement, how ='inner', 
                     on = 'video_id') 

# save DataFrame as .csv file
merged_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_1]), index= False)    

# sub-setting: taking only 'total_views' > 30  
merged_df_small_tmp = merged_df.loc[(merged_df['total_views'] > 30)]

# re-set index
merged_df_small_tmp.reset_index(drop = True, inplace = True)


