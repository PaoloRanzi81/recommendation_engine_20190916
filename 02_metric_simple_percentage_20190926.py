"""
TITLE: "Loading data and prepare data for reccomendation engine project"
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
# number of .csv files to be processed. The pattern for calling them is:
# '0.csv', '1.csv', '2.csv' ... '900.csv'.
number_csv_files = 900 
# output csv file of the clean data
output_file_name = ('i_stacked_simple_percentage.csv')
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
parallel = True # switch-off when using Linux command-line

# start clocking time
start_time = time.time()

###############################################################################
## FUNCTION
# function for manipulating .csv files in a loop
def manipulating_csv(BASE_DIR_INPUT, i):
    """Helper function:
    
    Arguments (INPUTS):
        BASE_DIR_INPUT {string} -- absolute path to the location where the csv files
                             are stored;
        i {integer} -- counter needed for building file name. Example: "5.csv";
    OUTPUTS: 
        'video_id_and_duration' -- a single Pandas DataFrame with all pieces of 
        information squeezed out from one single .CSV file; 
    """
    ## LOADING
    # set file name for input
    file_name = str(i) + '.csv'
    
    # check whether the file exists
    #if os.path.isfile(os.path.sep.join([BASE_DIR_INPUT, file_name])): 
       
    #loading the .csv file with raw data already flattend (not deeply nested JSON)
    raw_data_frame = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         file_name]), header = 0)
       
    ## PRE-PROCESSING
    # data cleaning and data manipulation
        
    # drop duplicates
    raw_data_frame = raw_data_frame.drop_duplicates().copy()
              
    # test pipeline on a small sample    
    #raw_data_frame = raw_data_frame.loc[0:1000,:].copy()   
    
    ## SIMPLE METRIC (percentage of watched video with respect to total 
    # video duration
    session_id_tmp = pd.DataFrame()
    session_id_df = pd.DataFrame()
    user_id_df = pd.DataFrame()
    liked_df = pd.DataFrame()
    disliked_df = pd.DataFrame()
    favorite_df = pd.DataFrame()
    full_screen_df = pd.DataFrame()
    video_id_df = pd.DataFrame()
    vid_duration_df = pd.DataFrame()
    time_on_page_df = pd.DataFrame()
    for session_id in pd.unique(raw_data_frame.loc[:, 'session_id']):
        session_id_tmp = raw_data_frame.loc[raw_data_frame['session_id'] == session_id].reset_index()
        session_id_tmp.reset_index(drop = True, inplace = True)
        for video_id in pd.unique(session_id_tmp.loc[:, 'vid']):
            session_id_series = pd.Series(str(session_id))
            session_id_df = session_id_df.append(session_id_series, ignore_index = True, sort = False)
            video_id_series = pd.Series(str(video_id))
            video_id_df = video_id_df.append(video_id_series, ignore_index = True, sort = False)
            vid_duration_tmp = session_id_tmp.loc[session_id_tmp['vid'] == video_id].reset_index()
            vid_duration = pd.Series(vid_duration_tmp.loc[0,'vid_duration'])
            vid_duration_df = vid_duration_df.append(vid_duration, ignore_index = True, sort = False)
            user_id_tmp = pd.Series(vid_duration_tmp.loc[0,'user_id'])
            user_id_df = user_id_df.append(user_id_tmp, ignore_index = True, sort = False)
            
            # check the existence of 'liked', 'disliked' and 'favorite' columns
            if 'liked' in raw_data_frame.columns: 
                liked_tmp = pd.Series(vid_duration_tmp.loc[0,'liked'])
                liked_df = liked_df.append(liked_tmp, ignore_index = True, sort = False)
                disliked_tmp = pd.Series(vid_duration_tmp.loc[0,'disliked'])
                disliked_df = disliked_df.append(disliked_tmp, ignore_index = True, sort = False)
                favorite_tmp = pd.Series(vid_duration_tmp.loc[0,'favorite'])
                favorite_df = favorite_df.append(favorite_tmp, ignore_index = True, sort = False)
                
            full_screen_tmp = pd.Series(vid_duration_tmp.loc[0,'full_screen'])
            full_screen_df = full_screen_df.append(full_screen_tmp, ignore_index = True, sort = False)
            time_on_page_series = pd.Series(vid_duration_tmp.loc[0,'time_on_page'])
            time_on_page_df = time_on_page_df.append(time_on_page_series, ignore_index = True, sort = False)

    # rename columns of DataFrames
    user_id_df = user_id_df.rename(columns={0: 'user_id'}, inplace = False)
    session_id_df = session_id_df.rename(columns={0: 'session_id'}, inplace = False)
    video_id_df = video_id_df.rename(columns={0: 'video_id'}, inplace = False)
    vid_duration_df = vid_duration_df.rename(columns={0: 'vid_duration'}, 
                                             inplace = False)
    time_on_page_df = time_on_page_df.rename(columns={0: 'time_on_page'}, 
                                             inplace = False)
    
    # check the existence of 'liked', 'disliked' and 'favorite' columns
    if 'liked' in raw_data_frame.columns: 
        liked_df = liked_df.rename(columns={0: 'liked'}, 
                                                 inplace = False)
        disliked_df = disliked_df.rename(columns={0: 'disliked'}, 
                                                 inplace = False)
        favorite_df = favorite_df.rename(columns={0: 'favorite'}, 
                                                 inplace = False)
    full_screen_df = full_screen_df.rename(columns={0: 'full_screen'}, 
                                             inplace = False)
    
    # check the existence of 'liked', 'disliked' and 'favorite' columns
    if 'liked' in raw_data_frame.columns:
        # concatenate across columns the two DataFrames
        video_id_and_duration = pd.concat([user_id_df, session_id_df, video_id_df, 
                                           vid_duration_df, time_on_page_df, 
                                           liked_df, disliked_df, favorite_df, 
                                           full_screen_df], axis = 1, sort = False)
    else:
        video_id_and_duration = pd.concat([user_id_df, session_id_df, video_id_df, 
                                           vid_duration_df, time_on_page_df, 
                                           full_screen_df], axis = 1, sort = False)
        
        
    # function's output
    return (video_id_and_duration)


###############################################################################
## FOR LOOP FOR LOADING + MANIPULATING .CSV FILES
# initialize DataFrame that will be use for appending outputs coming from 
# the function
video_stacked = pd.DataFrame()

# How many .csv files you want to retrieve
fnames = range(0, (number_csv_files))

# initialize a "for loop"
if parallel:

		# execute configs in parallel
		executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        
        # second 'for loop' (learning_rate)    
		tasks = (delayed(manipulating_csv)(BASE_DIR_INPUT, i) for i in fnames)
		output = executor(tasks)

        
else:
        # classical list comprehension 
		output = [manipulating_csv(BASE_DIR_INPUT, i) for i in fnames]



# append DataFrame (the key contents from each .CSV file are stacked in 
# one single DataFrame)
video_stacked = video_stacked.append(
        output, ignore_index = True)

# save 'video_stacked'
video_stacked.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name]), index= False)

# shows execution time
print( time.time() - start_time, "seconds")


