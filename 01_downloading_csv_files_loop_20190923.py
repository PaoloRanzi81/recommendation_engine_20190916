"""
TITLE: "Downloading .CSV files by "for loop" "
AUTHOR: Paolo Ranzi 
PYTHON VERSION: 3.6.7

DESCRIPTION: 
The script download serially .csv raw files from the servers to local folder.
Further, please change the following sections according to your individidual input preferences:
    - 'SETTING PATHS AND KEYWORDS'; 
"""


###############################################################################
## 1. IMPORTING LIBRARIES
# import required Python libraries
import sys, os, requests
import platform
import time

# start clocking time
start_time = time.clock()

# function for downloding .csv file in a loop
def download(folder, file_name):
    """Helper function
    
    Arguments:
        folder {string} -- absolute path to the location where you want to store csv files. With trailing slash
        file_name {string} -- remote file name. Example: "5.csv"
    """
    link = 'http://...' + file_name
    path = folder + file_name

    with open(path, "wb") as f:
            print("Downloading %s" % path)
            response = requests.get(link, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )    
                    sys.stdout.flush()


# Where to store files
#folder = os.getcwd() + "/csv/"
                    
                    
###############################################################################
## SETTING PATHS AND KEYWORDS
# In order to set the correct pathways/folders, check which system are you
# using. It should be either Linux laptop (release == '5.0.0-29-generic') 
# or Linux server (release == '4.15.0-1051-aws').
RELEASE = platform.release()

if RELEASE == '5.0.0-32-generic': # Linux laptop
   folder = ('/media/paolo/4C0F-08B1/analyses/python/tensorflow/recommendation_engine/raw_data')

else:
   folder = ('/home/ubuntu/raw_data')


# How many files you want to retrieve
fnames = range(0, 900)

for i in fnames:
    download(folder, str(i) + '.csv')
    
# shows execution time
print( time.clock() - start_time, "seconds")



