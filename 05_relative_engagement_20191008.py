"""
TITLE: "Computing relative engagement metric (Wu et al. 2018)"
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
import statistics as stats
from statsmodels.distributions.empirical_distribution import ECDF
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
input_file_name_1 = ('iii_video_id_seconds_watched.csv')
input_file_name_2 = ('ii_simple_median_standardized.csv')
output_file_name_1 = ('iiii_median_seconds_watched_tmp.csv')
output_file_name_2 = ('iiiii_relative_engagement_quantiles.csv')
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
### INPUT TO BE CHANGED SOON!!!!
total_seconds_watched = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_1]), header = 0, 
    dtype ={'video_id' : str})

# test pipeline on a small sample    
#total_seconds_watched = total_seconds_watched.loc[0:1000, :].copy() 
    
# drop NaN
total_seconds_watched = total_seconds_watched.dropna()  

### INPUT TO BE CHANGED SOON!!!!
total_views = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_2]), header = 0, 
    dtype ={'video_id' : str})
    
# drop NaN
total_views = total_views.dropna() 

# check types 
#total_seconds_watched.info()
#total_views.info()


###############################################################################
## SPLIT DATA IN CHUNKS (GOAL: IMPROVE COMPUTATIONAL SPEED)
# in order to try to put all the same video_ids within the same chuck. Otherwise
# we would have problem in summing the variables' values. Indeed, we will have
# lots of duplicates in the final aggregated DataFrame. 
tot_tmp = total_seconds_watched.sort_values(by = ['video_id'])
total_seconds_watched_sorted_by_video_id = tot_tmp.loc[tot_tmp['tot_seconds'] > 0]

# divide in chunks. Each chunk will be computed by a specific CPU 
chunks_df = np.array_split(total_seconds_watched_sorted_by_video_id , int(round((cpu_count() - 1), 0)))


###############################################################################
## FUNCTION: COMPUTING MEDIAN WATCHED TIME FOR EACH 'video_id'
# computing 'median_watch_time' by calculating median of all session_ids who 
# watched a specific 'video_id'
def median_watched_time(chunk):
    
    # initializing DataFrame
    video_id_watched_df = pd.DataFrame()
    
    # initialize chunk
    chunks_df_tmp = pd.DataFrame()
    chunks_df_tmp = chunks_df[chunk]
    
    for video_id in pd.unique(chunks_df_tmp.loc[:, 'video_id']):
        
        # sub-setting
        video_id_tmp = chunks_df_tmp.loc[chunks_df_tmp['video_id'] == str(video_id)].reset_index(drop = True)
        
        # build 'video_id' column
        video_id_series = pd.Series(str(video_id))
            
        # compute median of total views for each video
        median_single_series = pd.Series(stats.median_grouped(video_id_tmp.loc[:, 'tot_seconds']))
    
        # concatenate across columns the two DataFrames
        video_id_watched = pd.concat([video_id_series, median_single_series], axis = 1, sort = False)
        
        # rename columns
        video_id_watched = video_id_watched.rename(columns={0: 'video_id', 
                                                            1: 'median_sec_watched'}, 
                                         inplace = False)
        
        # append DataFrame
        video_id_watched_df = video_id_watched_df.append(video_id_watched, ignore_index = True, sort = False)


    # function's output
    return (video_id_watched_df)
    
    
###############################################################################           
## FOR LOOP FOR LOADING + MANIPULATING FILES

# initialize a "for loop"
if parallel:

		# execute configs in parallel
		executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        
        # second 'for loop' (learning_rate)    
		tasks = (delayed(median_watched_time)(chunk) for chunk in range(0, len(chunks_df)))
		output = executor(tasks)

        
else:
        # classical list comprehension 
		output = [median_watched_time(chunk) for chunk in range(0, len(chunks_df))]
    
    
###############################################################################           
## APPENDING/CONCATENATING DATAFRAMES
        
#initialize DataFrame
video_id_median_watched_df = pd.DataFrame()         
        
# append 'video_id_watched_df'
video_id_median_watched_df = video_id_median_watched_df.append(output, ignore_index = True, sort = False)

# drop duplicates
video_id_median_watched_df = video_id_median_watched_df.drop_duplicates().copy() 

# drop NaN
video_id_median_watched_df = video_id_median_watched_df.dropna()

# drop '0' values
video_id_median_watched_df = video_id_median_watched_df.loc[video_id_median_watched_df['median_sec_watched'] > 0]

# save 'video_id_watched_df'
video_id_median_watched_df.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_1]), index= False)    
 

###############################################################################           
## MERGING TWO DATAFRAMES                  
# merge 'video_id' + 'vid_duration' + 'total_views'  
#merged_df_tmp = pd.merge(total_views, vid_duration_df, how ='inner', 
#                     on = 'video_id')

# drop duplicates
#merged_df_tmp = merged_df_tmp.drop_duplicates().copy()

# merge 'median_sec_watched' with 
merged_df = pd.merge(video_id_median_watched_df, total_views, how ='inner', 
                     on = 'video_id') 


###############################################################################
## COMPUTING RELATIVE ENGAGEMENT    
# computing 'average_watch_percentage' 
merged_df['average_watch_percentage']  = merged_df.loc[:, 'median_sec_watched']/merged_df['vid_duration']

# clean DataFrame from outliers/ambiguos data. E.g. if the median duration is > vid_duration it means
# the user was likely idling instead of actually watching the movie   
mask = merged_df.loc[:, 'median_sec_watched'].le(merged_df.loc[:, 'vid_duration'])
merged_df = merged_df.loc[mask, :].copy()

# computing standardized 'vid_duration'
# standardized strictly positive values
merged_df['vid_duration_standardized'] = merged_df.loc[:, ['vid_duration']]  

# standardiz 'vid_duration
merged_df.loc[:, ['vid_duration_standardized']] = PowerTransformer(method='box-cox').fit_transform\
(merged_df.loc[:, ['vid_duration_standardized']]) 


# dividing the whole standardize/not standardized 'vid_duration' variable into 1000 bins. 
# Each bin represents an interval of video duration. All videos with the same length fall within the same bin. 
merged_df['quantile_rank'] = pd.qcut(merged_df['vid_duration_standardized'], 1000, labels = False, 
         duplicates = 'drop')

# computing log instead of 1000 quantile ranks. Please switch-off standardization
# of 'vid_duration' when using such the 'log'. Natural log or log10 appear to 
# generate identical RE scores. Overall, it seems to work well, 
# but it has lots of videos_is scoring '1'. Excessively positive metrics? Anyway, 
# the paper Wu et al. 2018 used log10 instead of 1000 quantiles. But by using quantiles
# seems to work better. 
#merged_df['quantile_rank'] = round(merged_df['vid_duration'].apply(np.log10), 2)
#merged_df['quantile_rank'] = merged_df['vid_duration'].apply(np.log10)


###############################################################################
## SPLIT DATA IN CHUNKS (GOAL: IMPROVE COMPUTATIONAL SPEED)
# in order to try to put all the same video_ids within the same chuck. Otherwise
# we would have problem in summing the variables' values. Indeed, we will have
# lots of duplicates in the final aggregated DataFrame. 
merged_tmp = merged_df.sort_values(by = ['quantile_rank'])

# divide in chunks. Each chunk will be computed by a specific CPU 
chunks_df = np.array_split(merged_tmp, int(round((cpu_count() - 1), 0)))


###############################################################################
## FUNCTION: COMPUTING PERCENTILES FOR EACH BIN OF 'vid_duration_standardized'
#def percentiles_computation(quantile_rank):
def percentiles_computation(chunk):

    # initialize Pandas DataFrame 
    percentiles_series_append = pd.DataFrame()
            
    # initialize chunk
    chunks_df_tmp = pd.DataFrame()
    chunks_df_tmp = chunks_df[chunk]
    
    for quantile_rank in pd.unique(chunks_df_tmp.loc[:, 'quantile_rank']):

        # sub-setting
        percentiles_tmp = chunks_df_tmp.loc[chunks_df_tmp['quantile_rank'] == quantile_rank ].reset_index(drop = True)
        
        # compute percentiles of the sub-set accorting to sub-set's size
        ecdf_values = []
        
        # prepare Numpy array
        numpy_array = percentiles_tmp.loc[:, ['average_watch_percentage']].to_numpy()
        numpy_vector = numpy_array.flatten()
        
        # compute percentiles (by using Empirical Cumulative Density Function method)
        ecdf_values = ECDF(numpy_vector)
    
        # build Pandas Series with percentiles
        percentiles_series_tmp = pd.Series(ecdf_values(numpy_vector))
        percentiles_series = percentiles_series_tmp.round(decimals = 2)
    
        # build Pandas Series with video_ids
        video_id_series = percentiles_tmp.loc[:, ['video_id']]
        
        # build Pandas Series with quantiles
        quantile_series_tmp = pd.Series(quantile_rank)
        quantile_series = quantile_series_tmp.repeat(repeats = percentiles_tmp.shape[0]).reset_index(drop = True) 
            
        # concatenate across columns the two DataFrames
        video_id_percentiles = pd.concat([video_id_series, quantile_series, percentiles_series], axis = 1, sort = False)
        
        # rename columns
        video_id_percentiles = video_id_percentiles.rename(columns={0: 'quantiles', 
                                                                    1: 'percentiles'}, 
                                         inplace = False)
        
        # append Series
        percentiles_series_append = percentiles_series_append.append(video_id_percentiles, ignore_index = True, sort = False)
    
    
    # function's output
    return (percentiles_series_append)


###############################################################################           
## FOR LOOP FOR LOADING + MANIPULATING FILES

# the function
percentiles_df = pd.DataFrame()

if parallel:

		# execute configs in parallel
		executor = Parallel(n_jobs= int(round((cpu_count() - 1), 0)), 
                                        backend='loky')
        
        # second 'for loop' (learning_rate)    
		tasks = (delayed(percentiles_computation)(quantile_rank) for quantile_rank in range(0, len(chunks_df)))
		output = executor(tasks)

        
else:
        # classical list comprehension 
		output = [percentiles_computation(quantile_rank) for quantile_rank in range(0, len(chunks_df))]


###############################################################################           
## APPENDING/CONCATENATING DATAFRAMES
# append 'video_id_watched'
percentiles_df = percentiles_df.append(output, ignore_index = True, sort = False)        


# merge 'median_sec_watched' with 
merged_df_percentile = pd.merge(merged_df, percentiles_df, how ='inner', 
                     on = 'video_id') 

# drop duplicate column
merged_df_percentile.drop(columns = ['quantiles'], inplace = True)

# rename 'percetiles' to 'RE' (Relative Engagement)
merged_df_percentile = merged_df_percentile.rename(columns={'percentiles': 'RE'}, 
                                         inplace = False)

# save DataFrame as .csv file
merged_df_percentile.to_csv(os.path.sep.join([BASE_DIR_OUTPUT, output_file_name_2]), index= False)    
 

# shows execution time
print(time.time() - start_time, "seconds")



        


