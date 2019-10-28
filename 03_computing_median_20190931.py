"""
TITLE: "Aggregate raw data into a tabular format" 
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
The script does:
- count total viewers for each 'video_id';  
- count 'liked', 'disliked', 'favorite' and 'full_screen';  
- compute medians for 'time_on_page'; 

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
input_file_name = ('i_stacked_simple_percentage.csv')
output_file_name = ('ii_simple_median_standardized.csv')
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
## LOADING
#loading the .csv files
manipulated_data_frame = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name]), header = 0)  

# test pipeline on a small sample    
#manipulated_data_frame = manipulated_data_frame.loc[0:1000, :].copy() 
    
# check summary statistics
#summary_stats = manipulated_data_frame.describe()

# check unique -not repeated- labels/item in the data-set
#summary_stats_unique = manipulated_data_frame.nunique()

    
###############################################################################
## SPLIT DATA IN CHUNKS (GOAL: IMPROVE COMPUTATIONAL SPEED)
# in order to try to put all the same video_ids within the same chunk. Otherwise
# we would have problem in summing the variables' values. Indeed, we will have
# lots of duplicates in the final aggregated DataFrame. 
manipulated_sorted_by_video_id = manipulated_data_frame.sort_values(by=['video_id']) 

# divide in chunks. Each chunk will be computed by a specific CPU 
chunks_df = np.array_split(manipulated_sorted_by_video_id , int(round((cpu_count() - 1), 0)))

    
###############################################################################
## FUNCTION: AGGREGATING DATA
def aggregating_data(chunk, total_views_sum_df, video_id_df, 
                     vid_duration_df, liked_df, disliked_df, favorite_df, full_screen_df):
    
    # initialize chunk
    chunks_df_tmp = pd.DataFrame()
    chunks_df_tmp = chunks_df[chunk]

    
    for video_id in pd.unique(chunks_df_tmp.loc[:, 'video_id']):    
        
        # sub-setting DataFrame
        total_views_tmp = chunks_df_tmp[chunks_df_tmp['video_id'] == video_id]
        
        # drop old index and reset a new one
        total_views_tmp.reset_index(drop = True, inplace = True)
        
        # check whether there are duplicated session_id
        if (pd.unique(total_views_tmp.loc[:, 'session_id']).shape[0] != total_views_tmp.shape[0]):
            print("Duplicated session_id:" + str(total_views_tmp.loc[:, 'session_id']))
        else: 
            total_views_sum_series = pd.Series(total_views_tmp.loc[:, 'session_id'].count())
            total_views_sum_df = total_views_sum_df.append(total_views_sum_series, ignore_index = True, sort = False)
    
            # sub-setting DataFrame
            subset_tmp = chunks_df_tmp.loc[chunks_df_tmp['video_id'] == video_id]
            subset_tmp.reset_index(drop = True, inplace = True)
            
            # aggregate 'video_id'
            video_id_series = pd.Series(str(video_id))
            video_id_df = video_id_df.append(video_id_series, ignore_index = True, sort = False)
            
            # aggregate 'vid_duration'
            vid_duration_series = pd.Series(pd.unique(subset_tmp.loc[:, 'vid_duration']))
            vid_duration_df = vid_duration_df.append(vid_duration_series, ignore_index = True, sort = False)
            
            # check the existence of 'liked', 'disliked' and 'favorite' columns
            if 'liked' in manipulated_data_frame.columns: 
                # count number of 'liked'
                liked_series = pd.Series(int(subset_tmp.loc[:, 'liked'].sum()))
                liked_df = liked_df.append(liked_series, ignore_index = True, sort = False)
                
                # count number of 'disliked'
                disliked_series = pd.Series(int(subset_tmp.loc[:, 'disliked'].sum()))
                disliked_df = disliked_df.append(disliked_series, ignore_index = True, sort = False)
                
                # count number of 'favorite'
                favorite_series = pd.Series(int(subset_tmp.loc[:, 'favorite'].sum()))
                favorite_df = favorite_df.append(favorite_series, ignore_index = True, sort = False)
            
            # count number of 'full_screen'
            full_screen_series = pd.Series(int(subset_tmp.loc[:, 'full_screen'].sum()))
            full_screen_df = full_screen_df.append(full_screen_series, ignore_index = True, sort = False)
            
            
    # rename columns of DataFrames
    video_id_df = video_id_df.rename(columns={0: 'video_id'}, inplace = False)
    total_views_sum_df = total_views_sum_df.rename(columns={0: 'total_views'}, inplace = False)
    vid_duration_df = vid_duration_df.rename(columns={0: 'vid_duration'}, inplace = False)
    # check the existence of 'liked', 'disliked' and 'favorite' columns
    if 'liked' in manipulated_data_frame.columns: 
        liked_df = liked_df.rename(columns={0: 'liked'}, inplace = False)
        disliked_df = disliked_df.rename(columns={0: 'disliked'}, inplace = False)
        favorite_df = favorite_df.rename(columns={0: 'favorite'}, inplace = False)
        
    full_screen_df = full_screen_df.rename(columns={0: 'full_screen'}, inplace = False)
    
    # check the existence of 'liked', 'disliked' and 'favorite' columns
    if 'liked' in manipulated_data_frame.columns:
        # concatenate across columns the two DataFrames
        video_id_and_views_tmp = pd.concat([video_id_df, vid_duration_df, total_views_sum_df, 
                                        liked_df, disliked_df, favorite_df, full_screen_df], 
        axis = 1, sort = False)
    else: 
        # concatenate across columns the two DataFrames
        video_id_and_views_tmp = pd.concat([video_id_df, vid_duration_df, total_views_sum_df, 
                                         full_screen_df], axis = 1, sort = False)

    # function's output
    return (video_id_and_views_tmp)

    
###############################################################################           
## FOR LOOP FOR LOADING + MANIPULATING .CSV FILES
# initialize DataFrame that will be use for appending outputs coming from 
# the function 

       
# initialize DataFrames
total_views_sum_df = pd.DataFrame()
video_id_df = pd.DataFrame()
vid_duration_df = pd.DataFrame()
liked_df = pd.DataFrame()
disliked_df = pd.DataFrame()
favorite_df = pd.DataFrame()
full_screen_df = pd.DataFrame()
aggregated_df = pd.DataFrame()  


# initialize a "for loop"
if parallel:

		# execute configs in parallel
		executor = Parallel(n_jobs= int(round((cpu_count()-1),0)), 
                                        backend='loky')
        
        # second 'for loop' (learning_rate)    
		tasks = (delayed(aggregating_data)(chunk, total_views_sum_df, video_id_df, 
                     vid_duration_df, liked_df, disliked_df, favorite_df, 
                     full_screen_df) for chunk in range(0, len(chunks_df)))
		output = executor(tasks)
        
else:
        # classical list comprehension 
		output = [aggregating_data(chunk, manipulated_data_frame, total_views_sum_df, video_id_df, 
                     vid_duration_df, liked_df, disliked_df, favorite_df, 
                     full_screen_df) for chunk in range(0, len(chunks_df))]
   

###############################################################################           
## APPENDING/CONCATENATING DATAFRAMES    
# append 'video_id_watched_df'
aggregated_df = aggregated_df.append(output, ignore_index = True, sort = False)

# check presence duplicates
#dupli_test = aggregated_df.nunique()

# save 'video_id_and_views'
aggregated_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name]), index= False)    

# shows execution time
print( time.time() - start_time, "seconds")


