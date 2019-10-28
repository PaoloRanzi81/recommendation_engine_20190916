"""
TITLE: "Detecting video_ids which have been watched more than once within the same 
session"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION:   
    The videos watched 2+ are considered the best ones, the gold-standard to 
compare everythingelse against. 
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
from sklearn.preprocessing import PowerTransformer
import time


###############################################################################
## SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.15.0-1051-aws').
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
# '0.csv', '1.csv', '2.csv' ... '900.csv'. WARNING: use half of the data (max 450 csv.)
# otherwise it throws a memory error!!
number_csv_files = 900 
output_file_name = ('list_of_double_watched_video_id.csv')
parallel = True # whenever starting the script from the Linux bash, uncomment such variable


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
## FUNCTION: COMPUTING SECONDS OF VIDEO ACTUALLY WATCHED BY EACH USER
def double_watch_session(BASE_DIR_INPUT, i):
    """Helper function:
    
    Arguments (INPUTS):
        BBASE_DIR_INPUT {string} -- absolute path to the location where the csv files
                             are stored;
        i {integer} -- counter needed for building file name. Example: "5.csv";
    OUTPUTS: 
        'session_id_double_df' -- a single Pandas DataFrame with all pieces of 
        information squeezed out from one single .CSV file; 
    """

    ## LOADING
    # set file name for input
    file_name = str(i) + '.csv'
    
    #loading the .csv file with raw data already flattend (not deeply nested JSON)
    raw_data_frame = pd.read_csv(os.path.sep.join([BASE_DIR_INPUT, 
                                         file_name]), header = 0)

   
    ## PRE-PROCESSING 
    # eliminate duplicates
    # drop duplicates
    raw_data_frame = raw_data_frame.drop_duplicates().copy() 
       
    # test pipeline on a small sample    
    #raw_data_frame = raw_data_frame.loc[:1000,:].copy()      

    # identify session_ids which watched the same video_id > than once within 
    # the same session_id
    user_id_tmp = pd.DataFrame()
    session_id_tmp = pd.DataFrame()
    session_id_double_df = pd.DataFrame()
    
    for user_id in pd.unique(raw_data_frame.loc[:, 'user_id']):
        user_id_tmp = raw_data_frame.loc[raw_data_frame['user_id'] == user_id].reset_index(drop = True)
        for session_id in pd.unique(user_id_tmp.loc[:, 'session_id']):
            session_id_tmp = user_id_tmp.loc[user_id_tmp['session_id'] == session_id].reset_index(drop = True)
            
            # number of video_id watched in each session_id
            #video_id_list = pd.Series(session_id_tmp.loc[:, 'vid'].nunique())
            video_id_list = pd.Series(pd.unique(session_id_tmp.loc[:, 'vid']).shape[0])

            # number of positions in each session_id   
            position_list = pd.Series(pd.unique(session_id_tmp.loc[:, 'pos_in_session']).shape[0])
            
            # detect double watching within the same session_id
            if position_list.gt(video_id_list).bool():
                
                # append all session_ids which watched the same video_id > than once
                session_id_double_df = session_id_double_df.append(session_id_tmp, ignore_index = True, sort = False) 
                
                # drop useless columns
                session_id_double_df.drop(columns = ['user_id', 'vid_duration', 'start', 'end',
                                                     'time_on_page', 'liked', 'disliked', 
                                                     'favorite', 'full_screen', 'repeated', 
                                                     'f_repeated', 'repeated_is_last', 
                                                     'f_repeated_is_last', 'event.type', 
                                                     'event.ts', 'event.vidTime'], inplace = True)
            else:
                pass
            
    # function's output
    return (session_id_double_df)
                
      
###############################################################################           
## FOR LOOP FOR LOADING + MANIPULATING .CSV FILES
# initialize DataFrame that will be use for appending outputs coming from 
# the function
session_id_double_watch_df = pd.DataFrame()

# How many .csv files you want to retrieve
fnames = range(0, (number_csv_files))

# initialize a "for loop"
if parallel:

		# execute configs in parallel
		executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        
        # second 'for loop' (learning_rate)    
		tasks = (delayed(double_watch_session)(BASE_DIR_INPUT, i) for i in fnames)
		output = executor(tasks)

        
else:
        # classical list comprehension 
		output = [double_watch_session(BASE_DIR_INPUT, i) for i in fnames]


###############################################################################           
## APPENDING/CONCATENATING DATAFRAMES
# append 'video_id_watched'
session_id_double_watch_df = session_id_double_watch_df.append(output, ignore_index = True, sort = False)

# drop duplicates
#session_id_double_watch_df_1 = session_id_double_watch_df.drop_duplicates().copy() 

# drop NaN
session_id_double_watch_df = session_id_double_watch_df.dropna()

# print status
print('First step done!')


###############################################################################
## SPLIT DATA IN CHUNKS (GOAL: IMPROVE COMPUTATIONAL SPEED)
# in order to try to put all the same video_ids within the same chuck. Otherwise
# we would have problem in summing the variables' values. Indeed, we will have
# lots of duplicates in the final aggregated DataFrame. 
double_watched_sorted_by_session_id = session_id_double_watch_df.sort_values(by = ['session_id'])

# divide in chunks. Each chunk will be computed by a specific CPU 
chunks_df = np.array_split(double_watched_sorted_by_session_id , int(round((cpu_count() - 1), 0)))


###############################################################################
## FUNCTION: IDENTIFY AND LIST video_id WATCHED > THAN ONCE WITHIN THE SAME session_id
def indentify_more_than_once(chunk):
    
    """Helper function:
    
    Arguments (INPUTS):
        'chunk' {integer} -- the chunk number that has to be delivered to 1 CPU;
    OUTPUTS: 
        'video_id_last_watched_df' -- a single Pandas DataFrame with all pieces of 
        information squeezed out from 'session_id_double_watch_df' DataFrame; 
    """
    
    # initializing DataFrame
    video_id_last_watched_df = pd.DataFrame()
    
    # initialize chunk
    chunks_df_tmp = pd.DataFrame()
    chunks_df_tmp = chunks_df[chunk]
    
    for session_id_list in pd.unique(chunks_df_tmp.loc[:, 'session_id']):
        session_id_list_tmp = chunks_df_tmp.loc[chunks_df_tmp['session_id'] == session_id_list].reset_index(drop = True)
        for video_id_list in pd.unique(session_id_list_tmp.loc[:, 'vid']):
            video_id_list_tmp = session_id_list_tmp.loc[session_id_list_tmp['vid'] == video_id_list].reset_index(drop = True)
            
            # getting rid of duplicates 
            video_id_list_tmp = video_id_list_tmp.drop_duplicates().copy() 
            
            # select only the 'video_id' which have been watched again within the same 'session_id'
            if (video_id_list_tmp.shape[0] == 1): 
                pass
            else: 
                
                # identify the video_id which have been watched > than once within a single session_id
                video_id_last_watched = video_id_list_tmp.loc[:, ['session_id','vid']].drop_duplicates().copy()

                # append all video_ids which have been watched > than once within a single session_id
                video_id_last_watched_df = video_id_last_watched_df.append(video_id_last_watched, ignore_index = True, sort = False) 
        
    # rename columns of DataFrames
    video_id_last_watched_df = video_id_last_watched_df.rename(columns={0: 'video_id'}, inplace = False)
   
    # function's output
    return (video_id_last_watched_df)


###############################################################################           
## FOR LOOP FOR LOADING + MANIPULATING FILES

# initialize a "for loop"
if parallel:

		# execute configs in parallel
		executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        
        # second 'for loop' (learning_rate)    
		tasks = (delayed(indentify_more_than_once)(chunk) for chunk in range(0, len(chunks_df)))
		output = executor(tasks)

        
else:
        # classical list comprehension 
		output = [indentify_more_than_once(chunk) for chunk in range(0, len(chunks_df))]
    
    
###############################################################################           
## APPENDING/CONCATENATING DATAFRAMES
        
#initialize DataFrame
video_id_list_df = pd.DataFrame()         
        
# append 'video_id_watched_df'
video_id_list_df = video_id_list_df.append(output, ignore_index = True, sort = False)

# drop NaN
video_id_list_df = video_id_list_df.dropna()

# save 'video_id_watched_df'
video_id_list_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name]), index= False)    


# shows execution time
print( time.time() - start_time, "seconds")
             

           

   
   

 
                
        

        


