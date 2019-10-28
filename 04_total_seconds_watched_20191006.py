"""
TITLE: "Computing total seconds watched. Total seconds watched for each video_id
is an intermediate step necessary for computing relative engagement metric (Wu et al. 2018)"
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
The script is parallelized by using 'joblib' Python library. Please set 'RELEASE' to 
your local system specifics if you would like to use the script by a single-core mode.
By default the script works by a multi-core/multi-threaded fashion. 
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
    - 'PARAMETERS TO BE SET!!!'

REFERENCES:
Wu, S., Rizoiu, M. A., & Xie, L. (2018, June). 
Beyond views: Measuring and predicting engagement in online videos. 
In Twelfth International AAAI Conference on Web and Social Media.    

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
number_csv_files = 1 
output_file_name = ('iii_video_id_seconds_watched.csv')
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
## FUNCTION: COMPUTING SECONDS OF VIDEO ACTUALLY WATCHED BY EACH USER
def total_seconds_watched(BASE_DIR_INPUT, i):
    """Helper function:
    
    Arguments (INPUTS):
        BASE_DIR_INPUT {string} -- absolute path to the location where the .csv files
                             are stored;
        i -- file_name counter. Example: "5.csv";
    OUTPUTS: 
        'video_id_watched_tmp' -- a single Pandas DataFrame with all pieces of 
        information squeezed out/aggregated from one single .CSV file; 
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

    # computing seconds of video actually watched
    user_id_tmp = pd.DataFrame()
    session_id_tmp = pd.DataFrame()
    position_tmp = pd.DataFrame()
    video_id_watched_tmp = pd.DataFrame()
    
    # loop through
    for user_id in pd.unique(raw_data_frame.loc[:, 'user_id']):
        user_id_tmp = raw_data_frame.loc[raw_data_frame['user_id'] == user_id].reset_index(drop = True)
        for session_id in pd.unique(user_id_tmp.loc[:, 'session_id']):
            session_id_tmp = user_id_tmp.loc[user_id_tmp['session_id'] == session_id].reset_index(drop = True)
            
            # determining whether the same video_id has been watched more than once
            # number of video_id watched in each session_id        
            video_id_list = pd.Series(session_id_tmp.loc[:, 'vid'].nunique())

            # number of positions in each session_id   
            position_list = pd.Series(pd.unique(session_id_tmp.loc[:, 'pos_in_session']).shape[0])
            
            # detect double watching within the same session_id
            if position_list.gt(video_id_list).bool():
                
                # initialize DataFrame
                rebuilding = pd.DataFrame()  
                
                for video_id_list_counter in pd.unique(session_id_tmp.loc[:, 'vid']):
                    video_id_list_tmp = session_id_tmp.loc[session_id_tmp['vid'] == video_id_list_counter].reset_index(drop = True)
         
                    # double-check: select only the 'video_id' which have been watched again within the same 'session_id'
                    if (video_id_list_tmp.loc[:, 'vid'].nunique() == 1):
                        rebuilding = rebuilding.append(video_id_list_tmp, ignore_index = True, sort = False)
                    else: 
                        position_series = pd.Series(pd.unique(video_id_list_tmp.loc[:, 'pos_in_session']))
                        video_id_tmp = video_id_list_tmp.loc[video_id_list_tmp['pos_in_session'] == position_series.iloc[-1]].reset_index(drop = True).copy() 
                        rebuilding = rebuilding.append(video_id_tmp, ignore_index = True, sort = False)

                # rebuild 'session_id_tmp'
                session_id_tmp = rebuilding.copy()
   
            else: 
                pass
            
            # after if-else condition above the loop continues 
            for position in pd.unique(session_id_tmp.loc[:, 'pos_in_session']):

                segment_length_df = pd.DataFrame()
                position_tmp = session_id_tmp.loc[session_id_tmp['pos_in_session'] == position].reset_index(drop = True)
                
                # WARNING: the loop is discarding on purpose users which have only 
                # 1 instance (i.e. 1 row) for each specific video_id
                for c, event in enumerate(position_tmp.loc[:, 'event.type']):
                    if event == 'play':
                        start_segment = position_tmp.loc[c, 'event.ts']
                    elif event == 'pause' and (c == 0):
                        pass    
                    elif event == 'pause':
                        end_segment = pd.DataFrame()
                        end_segment = position_tmp.loc[(c), 'event.ts']
                        segment_length = pd.Series(end_segment - start_segment)
                        segment_length_df = segment_length_df.append(segment_length, ignore_index = True, sort = False) 
                    elif event == 'ended':
                        end_segment = pd.DataFrame()
                        end_segment = position_tmp.loc[(c), 'event.ts']
                        segment_length = pd.Series(end_segment - start_segment)
                        segment_length_df = segment_length_df.append(segment_length, ignore_index = True, sort = False) 
                    elif  (event == 'seeking') and ((c + 1) == position_tmp.shape[0]):
                        end_segment = pd.DataFrame()
                        end_segment = position_tmp.loc[(c), 'end']
                        segment_length = pd.Series(end_segment - start_segment)
                        segment_length_df = segment_length_df.append(segment_length, ignore_index = True, sort = False) 
                    elif event == 'seeking':
                        pass
                    else:
                        segment_length_df = pd.DataFrame()
    
                # calculate total seconds watched per video_id
                if segment_length_df.empty:
                    pass
                else: 
                    seconds_watched = pd.Series(segment_length_df.sum())
                    video_id_series = pd.Series(pd.unique(session_id_tmp.loc[session_id_tmp['pos_in_session'] == position].vid))

                    # concatenate across columns the two DataFrames
                    video_id_watched = pd.concat([video_id_series, seconds_watched], 
                    axis = 1, sort = False)
                    
                    # rename columns
                    video_id_watched = video_id_watched.rename(columns={0: 'video_id', 
                                                                        1: 'tot_seconds'}, 
                                                     inplace = False)
                    
                    # append 'video_id_watched'
                    video_id_watched_tmp = video_id_watched_tmp.append(video_id_watched, ignore_index = True, sort = False)


    # function's output
    return (video_id_watched_tmp)
                
 
###############################################################################           
## FOR LOOP FOR LOADING + MANIPULATING .CSV FILES
# initialize DataFrame that will be use for appending outputs coming from 
# the function
video_id_watched_df = pd.DataFrame()

# How many .csv files you want to retrieve
fnames = range(0, (number_csv_files))

# initialize a "for loop"
if parallel:

		# execute configs in parallel
		executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        
        # second 'for loop' (learning_rate)    
		tasks = (delayed(total_seconds_watched)(BASE_DIR_INPUT, i) for i in fnames)
		output = executor(tasks)

        
else:
        # classical list comprehension 
		output = [total_seconds_watched(BASE_DIR_INPUT, i) for i in fnames]


###############################################################################           
## APPENDING/CONCATENATING DATAFRAMES
# append 'video_id_watched'
video_id_watched_df = video_id_watched_df.append(output, ignore_index = True, sort = False)

# drop duplicates
#video_id_watched_df = video_id_watched_df.drop_duplicates().copy() 

# drop NaN
video_id_watched_df = video_id_watched_df.dropna()

# sub-setting: taking only 'tot_seconds' > 0  
video_id_watched_df  = video_id_watched_df.loc[(video_id_watched_df['tot_seconds'] > 0)]

# save 'video_id_watched'
video_id_watched_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name]), index= False)    
 
# shows execution time
print( time.time() - start_time, "seconds")
             

           

   
   

 
                
        

        


