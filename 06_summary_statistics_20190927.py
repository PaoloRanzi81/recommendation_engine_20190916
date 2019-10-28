"""
TITLE: "Summary statistics. Exploratory Data Analyis (EDA) by using Empirical
Density Cumulative Function (ECDF)"
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
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from mlxtend.plotting import ecdf 


## SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.4.0-143-generic').
RELEASE = platform.release()

if RELEASE == '5.0.0-32-generic': # Linux laptop
   BASE_DIR_INPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/recommendation_engine/raw_data')
   BASE_DIR_OUTPUT = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/recommendation_engine/outputs')
  
else:
   BASE_DIR = '/home/Lan/paolo_scripts/exp_seasonality/seac_data'

   
###############################################################################
## PARAMETERS TO BE SET!!!
input_file_name_1 = ('video_more_than_once_small_best.csv')
input_file_name_2 = ('video_more_than_once_small.csv')


###############################################################################
## LOADING DATA-SET 
#loading the .csv file with raw data already flattend (not deeply nested JSON)
df_A = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_1]), header = 0)  

# rename
df_A.rename(columns={'percentiles': 'RE'}, 
                             inplace = True)

#loading the .csv file with raw data already flattend (not deeply nested JSON)
df_B = pd.read_csv(os.path.sep.join([BASE_DIR_OUTPUT, 
                                         input_file_name_2]), header = 0)

# rename    
df_B.rename(columns={'percentiles': 'RE'}, 
                             inplace = True)
 
""" 
# standardized strictly positive values
raw_data_frame.loc[:, ['total_views', 'vid_duration']] = PowerTransformer(method='box-cox',).fit_transform\
(raw_data_frame.loc[:, ['total_views', 'vid_duration']]) 

# standardized categorical variable (e.g. 'liked', 'disliked' etc.)
raw_data_frame.loc[:, ['liked', 'disliked', 'favorite', 'full_screen' ]] = MinMaxScaler().fit_transform\
(raw_data_frame.loc[:, ['liked', 'disliked', 'favorite', 'full_screen']]) 
#sample.loc[:, ['liked', 'disliked', 'favorite', 'total_views', 'vid_duration', 'full_screen']] = MinMaxScaler().fit_transform\
#(sample.loc[:, ['liked', 'disliked', 'favorite', 'total_views', 'vid_duration','full_screen']]) 

"""     
    
###############################################################################
## SIMPLE DATA-SET DIAGNOSTICS      
# check summary statistics
#summary_stats = df_A.describe()

# check unique -not repeated- labels/item in the data-set
#summary_stats_unique = raw_data_frame.nunique().T

# check data types
#data_frame_data_types = raw_data_frame.info()


###############################################################################
## EXPLORATORY DATA  ANALYSIS (EDA): Empirical Cumulative Density Function (ECDF) 


# close picture in order to avoid overwriting with previous pics
plt.clf()

# set ECDF    
ax, _, _ = ecdf(x = df_A.loc[:, 'RE'], 
                x_label='relative engagement', ecdf_color = 'green')
ax, _, _ = ecdf(x = df_B.loc[:, 'RE'], ecdf_color = 'red')
#ax, _, _ = ecdf(x = test_c, ecdf_color = 'yellow')
#ax, _, _ = ecdf(x = merged_df_percentile_2.loc[:, 'percentiles'], ecdf_color = 'red')
#ax, _, _ = ecdf(x = merged_df_percentile_3.loc[:, 'percentiles'], ecdf_color = 'blue')
#ax, _, _ = ecdf(x = merged_df_percentile.loc[:, 'percentiles'], ecdf_color = 'yellow')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles,
          labels = ['test_a', 'test_b'], 
          #labels = ['RE_100_quantiles', 'RE_1000_quantiles', 'RE_log', 'RE_log_10'], 
          framealpha=0.3, scatterpoints=1, loc='upper left')
plt.title('RE and video_ids watched 2+ times')

# save plot [NOT WORKING with Sebastian Raschka's ECDF!]
#date = str(datetime.datetime.now())
#plt.savefig(os.path.sep.join([BASE_DIR, 
#                          date[0:10]+ "_" + date[11:len(date)]+".jpg"]))



###############################################################################
## EXPLORATORY DATA  ANALYSIS (EDA): Probability Density Function (PDF) 
"""
# load total_views data-set
input_file_name_3 = ('simple_median_standardized.csv')
#loading the .csv file with raw data already flattend (not deeply nested JSON)
simple_median = pd.read_csv(os.path.sep.join([BASE_DIR, 
                                         input_file_name_3]), header = 0)
    
# delete extreme outliers for 'total_views' 
#simple_median_second_quantile = simple_median.loc[:, 'total_views'].quantile(q = 0.05, interpolation='higher')
simple_median_second_quantile = 0
simple_median_third_quantile = simple_median.loc[:, 'total_views'].quantile(q = 0.95, interpolation='lower')

# subsetting 'vid_duration': only the instances > 0.25 and < 0.75 quantile are considered
mask = (simple_median.loc[simple_median['total_views'] > simple_median_second_quantile]).index & (simple_median.loc[simple_median['total_views'] < simple_median_third_quantile ]).index  
raw_simple_median_quantiles = simple_median.iloc[mask, :]  
raw_simple_median_quantiles.reset_index(drop=True, inplace=True)

# computing meadin for 'total_views'
#median_total_views = pd.Series(np.median(raw_simple_median_quantiles.loc[:, ['total_views']])) 
#median_total_views = pd.Series(stats.median_grouped(raw_simple_median_quantiles.loc[:, ['total_views']])) 

    

# delete extreme outliers for 'vid_duration' 
vid_duration_second_quantile = raw_data_frame.loc[:, 'vid_duration'].quantile(q = 0.05, interpolation='higher')
vid_duration_third_quantile = raw_data_frame.loc[:, 'vid_duration'].quantile(q = 0.95, interpolation='lower')

# subsetting 'vid_duration': only the instances > 0.25 and < 0.75 quantile are considered
mask = (raw_data_frame.loc[raw_data_frame['vid_duration'] > vid_duration_second_quantile]).index & (raw_data_frame.loc[raw_data_frame['vid_duration'] < vid_duration_third_quantile]).index  
raw_vid_duration_quantiles = raw_data_frame.iloc[mask, :]  
raw_vid_duration_quantiles.reset_index(drop=True, inplace=True)

median_vid_duration_quantiles = np.median(raw_data_frame.loc[raw_data_frame['vid_duration']]) 


# close picture in order to avoid overwriting with previous pics
plt.clf()

# set image size
pdf = plt.figure(figsize=(26,14))

# plot PDF
#pdf = sns.distplot(raw_simple_median_quantiles.loc[:, 'total_views'], bins = 100, kde = True)
pdf = sns.distplot(raw_data_frame.loc[:, 'percentage_watched'], bins = 90, kde = True)
#pdf = sns.distplot(raw_vid_duration_quantiles.loc[:, 'vid_duration'], bins = 90, kde = True)

#pdf = plt.title('PDF of total_views not standardized')
#pdf = plt.title('PDF of total_views standardized')
#pdf = plt.title('PDF of percentage_watched not standardized')
pdf = plt.title('PDF of percentage_watched standardized')
#pdf = plt.title('PDF of vid_duration not standardized')
#pdf = plt.title('PDF of vid_duration standardized')

#pdf = plt.xlabel('total_views for each video_id')
#pdf = plt.ylabel('frequency of total_views')
#pdf = plt.xlabel('percentage_watched (values from 0 to 1)')
pdf = plt.xlabel('percentage_watched')
pdf = plt.ylabel('frequency of percentage_watched')
#pdf = plt.xlabel('seconds of vid_duration')
#pdf = plt.ylabel('frequency of vid_duration')



#fig = pdf.get_figure() # Is it working?
#date = str(datetime.datetime.now())
#plt.savefig(os.path.sep.join([BASE_DIR, 
                  date[0:10] + "_" + date[11:19] + "_" + date[21:23] +".jpg"]))
"""

"""
###############################################################################
##  SUBSETTING BY QUANTILES:

# compute 0.95 quantile (not standardized data):
vid_duration_second_quantile = raw_data_frame.loc[:, 'vid_duration'].quantile(q = 0.25, interpolation='higher')
vid_duration_third_quantile = raw_data_frame.loc[:, 'vid_duration'].quantile(q = 0.75, interpolation='lower')

# subsetting 'vid_duration': only the instances > 0.25 and < 0.75 quantile are considered
mask = (raw_data_frame.loc[raw_data_frame['vid_duration'] > vid_duration_second_quantile]).index & (raw_data_frame.loc[raw_data_frame['vid_duration'] < vid_duration_third_quantile]).index  

raw_vid_duration_quantiles = raw_data_frame.iloc[mask, :]  

# compute 0.95 quantile (not standardized data):
total_views_quantile_0_95th = raw_vid_duration_quantiles.loc[:, 'total_views'].quantile(q = 0.95, interpolation='higher')

# subsetting 'total_views': only the instances > 0.95 quantile are considered
raw_total_views_quantiles = raw_vid_duration_quantiles.loc[raw_vid_duration_quantiles['total_views'] > total_views_quantile_0_95th]

# delete outliers from original dataset
#threshold = pd.Series(clusterer.outlier_scores_).quantile(0.95)
#outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
"""

###############################################################################
## EXPLORATORY DATA  ANALYSIS (EDA): PAIRPLOT BY SEABORN 
# copy sample
#sample = raw_data_frame.copy() 
#sample = raw_data_frame.loc[:, ['duration', 'total_views_x', 'liked_x', 'disliked_x',  ]] 

sample = df_A.loc[:, ['RE',  'liked', 'full_screen']] 

# build a KPI/metric 
#sample['full_vs_views'] = (sample.loc[:, 'full_screen']/sample.loc[:, 'total_views'])
# drop variable related to KPI/metric in order to avoid multi-collinearity
#sample.drop(columns =['video_id', 'vid_duration', 'med_simp_time', 'med_simp_perc', 'med_grouped_time', 'full_screen', 'total_views' ], inplace = True)

# drop useless variables
#sample.drop(columns =['video_id', 'med_simp_time', 'med_simp_perc', 'med_grouped_time'], inplace = True)

# plot pair plot
pair_plot = sns.pairplot(sample, diag_kind = 'kde')
#sns.pairplot(iris, kind="reg")

  

