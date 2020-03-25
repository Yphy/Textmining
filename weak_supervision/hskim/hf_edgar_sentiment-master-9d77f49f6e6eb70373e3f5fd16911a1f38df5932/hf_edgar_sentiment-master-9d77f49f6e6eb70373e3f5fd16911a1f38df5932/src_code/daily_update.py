import os
import sys
import numpy as np
from datetime import datetime
from hf.framework.hfclndr import HFCLNDR
from shutil import copyfile
import shutil
from distutils.dir_util import copy_tree
from configparser import ConfigParser
from generate import Generate

global g; global cfg
cfg = ConfigParser()
cfg.read('/mnt/code/generator/hf_edgar_sentiment/config.ini')
g = Generate()

hfclndr = HFCLNDR(start=20100101)
date_l = hfclndr.get_date_list("tradingdays")
TODAY = date_l[-2]
ONE_DAY_AGO = date_l[-3]
TWO_DAY_AGO = date_l[-4]
ONE_WEEK_AGO = date_l[-8]

today_path = str(TODAY)
today_path = today_path[:4] + '/' + today_path[4:6]

one_day_path = str(ONE_DAY_AGO)
one_day_path = one_day_path[:4] + '/' + one_day_path[4:6]

two_day_path = str(ONE_DAY_AGO)
two_day_path = two_day_path[:4] + '/' + two_day_path[4:6]

one_week_path = str(ONE_WEEK_AGO)
one_week_path = one_week_path[:4] + '/' + one_week_path[4:6]

def prepare():

    print "Preparing {}".format(today_path)
    edgar_one_day = str(ONE_DAY_AGO)[:4] + '/' + str(ONE_DAY_AGO)[4:6] + '/' + str(ONE_DAY_AGO)[6:]
    edgar_two_day = str(TWO_DAY_AGO)[:4] + '/' + str(TWO_DAY_AGO)[4:6] + '/' + str(TWO_DAY_AGO)[6:]
    dir_10k_yester = os.path.join(cfg['edgar']['ROOT_10_K'],edgar_one_day)
    dir_10k_two_ago = os.path.join(cfg['edgar']['ROOT_10_Q'],edgar_one_day)
    dir_10q_yester = os.path.join(cfg['edgar']['ROOT_10_K'],edgar_two_day)
    dir_10q_two_ago = os.path.join(cfg['edgar']['ROOT_10_Q'],edgar_two_day)

    dir_ls = \
    [dir_10k_yester,dir_10q_yester,dir_10k_two_ago,dir_10q_two_ago]

    for path in dir_ls :
        if not os.path.isdir(path):
            print "WARNING : There's no EDGAR filing file: %s" % path
#            sys.exit(1)
    print "Prepared! (0)"
    sys.exit(0)

def generate_update():

    # 10-K periodic generation
    g.load(start_date = 20100101,end_date = 30000101,report='10-K',ratio=0.1)
    path_ls = g.load_path(start_d = ONE_WEEK_AGO, end_d = TODAY)
    try :
        g.generate(path_ls,dump=True)
    except Exception as e :
        print e
        print "Processing Error in 10-K (11)"
        sys.exit(11)

    # 10-Q periodic generation
    g.load(start_date = 20100101,end_date = 30000101,report='10-Q',ratio=0.1)
    path_ls = g.load_path(start_d = ONE_WEEK_AGO, end_d = TODAY)
    try :
        g.generate(path_ls,dump=True)
    except Exception as e :
        print e
        print "Processing Error in 10-Q (11)"
        sys.exit(11)

    print "Generate indexed partition successfully (0)"
    sys.exit(0)

def validate_update():

    criteria = os.listdir(cfg['locdisk']['LOC_UPDATE'])
    if len(criteria) == 0:
        print "ERROR!!!!: There's no EDGAR filing file"
        sys.exit(12)
    print "There are {} files".format(len(criteria))
    print "Prepared! (0)"
    sys.exit(0)

def append_update_to_short_cache():
    '''
    because method named g.generate_multiprocessing dumps daily file into yyyy/mm, So to append NAS,
    this stage copy the daily file to update folder. yyyy/mm -> update/
    And copy the lemma dict pickle file also.
    '''

    update_dir = cfg['locdisk']['LOC_UPDATE']
    src_ls = []
    for report in os.listdir(update_dir) :# 10-K or 10-Q
        report_path = os.path.join(update_dir,report)
    	for year in os.listdir(report_path) :# 2010,2011
    	    year_path = os.path.join(report_path,year)
    	    for month in os.listdir(year_path) :# 01,02
    	        month_path = os.path.join(year_path,month)
    		for fname in os.listdir(month_path) :
    		    file_path = os.path.join(month_path,fname)
    		    src_ls.append(file_path)
    try :
        for src in src_ls :
            des = src.replace('hf_edgar_sentiment/update/','hf_edgar_sentiment/')
	    tmp_dir = os.path.split(des)[0]
	    if not os.path.isdir(tmp_dir) :
	        os.makedirs(tmp_dir)
        copyfile(src,des)
        # /locdisk/processed/hf_edgar_sentiment/update/10-K/2018/10/02.csv ->
        # /locdisk/processed/hf_edgar_sentiment/10-K/2018/10/02.csv

    except Exception as e :
        print e
        print "Processing Error during copy folder in locdisk"
        sys.exit(13)

    print "NAS' update folder is well generated (0)"
    sys.exit(0)

def validate_short_cache():
    """
    method name check_acc_no is for checking the whether access number between file_dict and mapper.
    its match guarantees the partitioning data exist.
    """
    print "Nothing to do"
    sys.exit(0)

def append_update_to_long_cache():
    '''
    append(spread) synced update file to yyyy/dd folder
    '''
    src = cfg['nas']['NAS_UPDATE']
    des = cfg['nas']['NAS_ROOT']
    src_ls = []
    for report in os.listdir(src) :
	# 10-K or 10-Q
        report_path = os.path.join(src,report)
	for year in os.listdir(report_path) :
	# 2010,2011
	    year_path = os.path.join(report_path,year)
	    for month in os.listdir(year_path) :
	    # 01,02
	        month_path = os.path.join(year_path,month)
		for fname in os.listdir(month_path) :
		    file_path = os.path.join(month_path,fname)
		    src_ls.append(file_path)

    for src_fname in src_ls :
	des_fname = src_fname.replace("hf_edgar_sentiment/update/",'hf_edgar_sentiment/')
	des_dir = os.path.split(des_fname)[0]
	if not os.path.isdir(des_dir) :
	    os.makedirs(des_dir)
    try :
        copyfile(src_fname,des_fname)
    except Exception as e:
        print e
        print "Processing Error during copy folder in mnt"
        sys.exit(14)

    print "locdisk update/ and nas update/ folders are going to be deleted."

    shutil.rmtree(cfg['locdisk']['LOC_UPDATE'])
    shutil.rmtree(cfg['nas']['NAS_UPDATE'])

    print "NAS's long cache is well appended"
    sys.exit(0)

def extend_short_cache_universe():
    print "Nothing to do"
    sys.exit(0)

if __name__ == "__main__":
    sys.argv.append(1)
    if len(sys.argv) < 2:
        print "Usage: {} [Step #1-7]".format(sys.argv[0])
        sys.exit(-1)
    else:
        step = int(sys.argv[1])


    if step==1:
        prepare()
    elif step == 2:
        generate_update()
    elif step==3:
        validate_update()
    elif step==4:
        append_update_to_short_cache()
    elif step==5:
        validate_short_cache()
    elif step==6:
        append_update_to_long_cache()
    elif step==7:
        extend_short_cache_universe()
    else:
        print "Usage: {} [Step #1-7]".format(sys.argv[0])
        sys.exit(-1)
