"""
TITLE: "Detecting videos watched 2+ and compare with results with other employees"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

The videos watched 2+ are considered the best ones, the gold-standard to 
compare everythingelse against. 
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
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/recommendation_engine/watched_2_plus')
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
input_file_name_1 = ('list_of_double_watched_video_id.csv')
output_file_name_1 = ('victor_2_plus_watched.csv')
output_file_name_2 = ('merged_victor_paolo.csv')   


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
parallel = True

# start clocking time
start_time = time.time()


###############################################################################
## FUNCTION
# function for manipulating .csv files in a loop
def manipulating_csv(BASE_DIR, i):
    """Helper function:
    
    Arguments (INPUTS):
        BASE_DIR {string} -- absolute path to the location where the csv files
                             are stored;
        file_name {string} -- remote file name. Example: "5.csv";
    OUTPUTS: 
        'video_id_and_duration' -- a single Pandas DataFrame with all pieces of 
        information squeezed out from one single .CSV file; 
    """
    ## LOADING
    # set file name for input
    file_name = str(i) + '_watched_twice.csv'
    
    #loading the .csv file with raw data already flattend (not deeply nested JSON)
    raw_data_frame = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         file_name]), header = 0)
    
    # test pipeline on a small sample    
    #raw_data_frame = raw_data_frame.loc[0:1000,:].copy() 
    
    # initialize DataFrames  
    session_id_tmp = pd.DataFrame()
    session_id_df = pd.DataFrame()
    video_id_df = pd.DataFrame()
    
    
    for session_id in pd.unique(raw_data_frame.loc[:, 'session_id']):
        session_id_tmp = raw_data_frame.loc[raw_data_frame['session_id'] == session_id].reset_index()
        session_id_tmp.reset_index(drop = True, inplace = True)
        for video_id in pd.unique(session_id_tmp.loc[:, 'vid']):
            session_id_series = pd.Series(str(session_id))
            session_id_df = session_id_df.append(session_id_series, ignore_index = True)
            video_id_series = pd.Series(str(video_id))
            video_id_df = video_id_df.append(video_id_series, ignore_index = True)
    
    # rename columns of DataFrames
    session_id_df = session_id_df.rename(columns={0: 'session_id'}, inplace = False)
    video_id_df = video_id_df.rename(columns={0: 'video_id'}, inplace = False)
   
    
    # concatenate across columns the two DataFrames
    video_id_and_duration = pd.concat([session_id_df, video_id_df], axis = 1)   


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
		executor = Parallel(n_jobs= int(round((cpu_count() - 1),0)), 
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
        output, ignore_index=True)

# save 'video_stacked'
video_stacked.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                     output_file_name_1]), index= False)

# loading the .csv files
double_watched_tmp = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_1]), header = 0, 
    dtype ={'vid' : str})
   
# rename column    
double_watched_tmp = double_watched_tmp.rename(columns={ 'vid': 'video_id'}, 
                                         inplace = False)    
    
# drop duplicate column
#double_watched_tmp.drop(columns = ['session_id'], inplace = True)

# unique video_id
#double_watched = pd.Series(pd.unique(double_watched_tmp.loc[ : , 'video_id']))

# rename column    
#double_watched = double_watched.rename('video_id', inplace = False)

# merge 'median_sec_watched' with 
merged_df = pd.merge(video_stacked, double_watched_tmp, how ='inner', 
                     on = 'video_id') 

# save DataFrame as .csv file
merged_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2]), index= False)       
    
# shows execution time
print( time.time() - start_time, "seconds")


