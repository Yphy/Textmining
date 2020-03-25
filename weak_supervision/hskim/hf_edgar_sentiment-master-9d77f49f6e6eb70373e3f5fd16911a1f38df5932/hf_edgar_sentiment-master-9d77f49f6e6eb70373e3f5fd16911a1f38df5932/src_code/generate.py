### python ###
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import torch
import warnings
from configparser import ConfigParser
try :
    from nltk.sentiment import vader
    analyzer = vader.SentimentIntensityAnalyzer()
except :
    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment import vader
    analyzer = vader.SentimentIntensityAnalyzer()
### custom ###
from src_code import utils
import indexer
import discrete_cnn as model
import noise_aware_cnn

class Generate() :

    def load(self,start_date,end_date,report,ratio) :
        # core params
        self.start_date = start_date
        self.end_date = end_date
        self.report = report
        if report not in ['10-K','10-Q'] :
            raise ValueError("report only can be between 10-K and 10-Q")
        self.ratio = ratio
        self.cfg = ConfigParser()
        self.cfg.read('/mnt/code/generator/hf_edgar_sentiment/config.ini')

        # path
        if self.report == '10-K' :
            self.__edgar_root_dir = self.cfg['edgar']['ROOT_10_K']
        elif self.report == '10-Q':
            self.__edgar_root_dir = self.cfg['edgar']['ROOT_10_Q']
        self.__loc_dir = self.cfg['locdisk']['LOC_ROOT']
	self.__update_dir = self.cfg['locdisk']['LOC_UPDATE']
        self.__lemma_dir = self.cfg['locdisk']['LOC_LEMMA']
        self.__model_dir = self.cfg['locdisk']['LOC_MODEL']
        self.__nas_dir = self.cfg['nas']['NAS_ROOT']
	self.__nas_update_dir = self.cfg['nas']['NAS_UPDATE']
        self.__item2_dir = self.cfg['locdisk']['LOC_10_Q']
        self.__item7_dir = self.cfg['locdisk']['LOC_10_K']

        self.__lemma_cache_dir = os.path.join(self.__lemma_dir,'lemma_dict.pkl')
        self.__discrete_model_dir = os.path.join(self.__model_dir,'cnn_model.model')
        self.__noise_aware_model_dir = os.path.join(self.__model_dir,'noise_aware_cnn_model.model')

        print('[load] CNN model load is started')
        discrete_params = model.set_params()
        noise_aware_params = noise_aware_cnn.set_params()

        self.cnn_model = model.CNN(**discrete_params)
        self.cnn_model.load_state_dict(torch.load(self.__discrete_model_dir))#,map_location= {'cuda:0' : 'cpu'}))

        self.noise_cnn_model = noise_aware_cnn.CNN(**noise_aware_params)
        self.noise_cnn_model.load_state_dict(torch.load(self.__noise_aware_model_dir))#,map_location= {'cuda:0' : 'cpu'}))
        if self.cnn_model.device != 'cpu' :
            print("[load] model device is on cuda")
            self.device = 'cuda'
        else :
            self.device = 'cpu'
            print("[load] model device is on cpu.")
        # load cache
        if os.path.exists(self.__lemma_cache_dir) :
            print('[load] lemma dict is loaded')
            self.lemma_dict = pd.read_pickle(self.__lemma_cache_dir)
        else :
            warnings.warn('where is no lemma dict')
            self.lemma_dict = None

    def __str__(self) :
        return '[UPDATE NOTE] THIS IS THE LASTEST'

    def load_path(self,start_d=None,end_d=None) :
        if (start_d == None) and (end_d == None) :
            start_d = self.start_date
            end_d = self.end_date

        ls = []
        if self.report == '10-Q' :
            item = 'item2.'
        elif self.report == '10-K' :
            item = 'item7.'
        for root in sorted(os.listdir(self.__edgar_root_dir)) :
            year_dir = os.path.join(self.__edgar_root_dir,root)
            year_list = sorted(os.listdir(year_dir))
            for year in year_list :
                month_dir = os.path.join(year_dir,year)
                month_list = sorted(os.listdir(month_dir))
                for day in month_list :
                    day_dir = os.path.join(month_dir,day)
                    day_list = sorted(os.listdir(day_dir))
                    for file_ in day_list :
                        if item in file_ :
                            file_path = os.path.join(day_dir,file_)
                            ls.append(file_path)

        yyyymmdd_ls = [int(','.join(i.split("/")[5:8]).replace(",",'')) for i in sorted(ls)]
        start_idx = np.searchsorted(yyyymmdd_ls,start_d)
        end_idx = np.searchsorted(yyyymmdd_ls,end_d)
        path_ls = sorted(ls)[start_idx:end_idx]

        return path_ls

    def get_summarized(self,path_ls=None) :
        """
        extract from the documents by using the textrank
        """
        print('[get_summarized] summarization ratio : {}'.format(self.ratio))
        if path_ls == None :
            path_ls = self.load_path()
        summarized_df = utils.read_and_summarize(path_ls, self.ratio)
        summarized_df.columns = ['path','text']
        summarized_df['text'] = summarized_df.text.apply(lambda x : x[-1] if isinstance(x,tuple) else x)
        return summarized_df

    def get_parsed(self,summarized_df) :
        """
        parsing the raw sentences
        """
        print("[get_parsed] parsing is on progress")
        summarized_df['text'] = summarized_df.text.apply(lambda x : x[-1] if isinstance(x,tuple) else x)
        summarized_df['lemmatized_text'] = summarized_df.text.str.split(" ")\
        .apply(lambda sent : ','.join(list(map(utils.finer_lemmatize, sent))).replace(",", ' ')\
        if isinstance(sent,list) else sent) # kind of try except
        print("[get_parsed] parsing core function is started")
        summarized_df['parsed_text'] = summarized_df.lemmatized_text.apply(utils.parsing)
        print("[get_parsed] length filtering (<=100) is going to be applied")
        summarized_df['length'] = summarized_df.parsed_text.apply(lambda x : len(x.split(" ")))
        parsed_df = summarized_df[summarized_df.length <= 100].reset_index(drop=True)
        return parsed_df

    def get_indexed(self,parsed_df) :
        """
        indexing to extracted sentence
        """
        print("[get_indexed] indexing is on progress")
        if self.report == '10-K' :
            item = 'item7'
        elif self.report == '10-Q' :
            item = 'item2'
        indexed_df = utils.assign_ii_di(parsed_df, item)
        print("[get_indexed] indexing is started")
        indexed_df = indexer.indexing(indexed_df,self.lemma_dict,train=False)
        indexed_df = \
        indexed_df[['path', 'parsed_text', 'HFSID', 'PERIOD_OF_REPORT', 'UPDATED_DAY', 'indexed_text']]
        return indexed_df

    def get_assigned(self,indexed_df) :
        """
        assign probabilistic label to indexed sentence
        """

        print("[get_assigned] assigning is on progress")
        assigned_df = indexed_df.copy()
        if type(assigned_df.indexed_text.iloc[0]) == str :
            assigned_df['indexed_text'] = assigned_df.indexed_text.apply(eval)
        test_X = torch.tensor(list(assigned_df.indexed_text.apply(lambda x : np.array(x,dtype=float)).values))
        test_X = test_X.type(torch.float)

        print('[get_assigned] DL model start to predict')
        if self.device == 'cuda' :
            test_X = test_X.cuda()
            self.cnn_model = self.cnn_model.cuda()
            self.noise_cnn_model = self.noise_cnn_model.cuda()
        discrete_predict = model.cpu_test(self.cnn_model,test_X)
        noise_aware_predict = noise_aware_cnn.cpu_test(self.noise_cnn_model,test_X)
        print('[get_assigned] DL model finish to predict')

        bin_pred = discrete_predict[:,1] # convert softmax to probability form.
        bin_pred = bin_pred[:len(indexed_df)].numpy()

        con_pred = noise_aware_predict.squeeze()
        con_pred = con_pred[:len(indexed_df)].numpy()

        assigned_df['bin_score'] = bin_pred
        assigned_df['noise_score'] = con_pred
        assigned_df['vader'] = \
            assigned_df.parsed_text.apply(lambda x : analyzer.polarity_scores(x)['compound'])
        return assigned_df

    def generate(self,custom_path=None,dump=False) :
        """
        overall pipeline wrapper
        """
        if type(custom_path) != list :
            summarized_df = self.get_summarized()
        else :
            summarized_df = self.get_summarized(custom_path)
        parsed_df = self.get_parsed(summarized_df)
        indexed_df = self.get_indexed(parsed_df)
        assigned_df = self.get_assigned(indexed_df)

        if dump :
            ### dump logic ###
            assigned_df['dump_path'] = assigned_df.path.\
            apply(lambda x : os.path.split(x)[0]\
            .replace("/mnt/processed/edgar_footnote_partition",'{}'.format(self.__update_dir))+'.csv')
# /locdisk/processed/hf_edgar_sentiment/10-K
# /locdisk/processed/hf_edgar_sentiment/update/10-K

            dump_path_ls = assigned_df.dump_path.unique()
            for i in dump_path_ls :
                cache = assigned_df[assigned_df.dump_path == i].reset_index(drop=True)
                if os.path.exists(i) :
                    tmp = pd.read_csv(i)
                    orgin_path = set(tmp.path);new_path = set(cache.path)
                    unique_path = list(new_path.difference(orgin_path))
                    if len(unique_path) == 0 :
                        continue
                    cache = cache[cache.path.isin(unique_path)].reset_index(drop=True)
                    cache = pd.concat([tmp,cache],ignore_index=True)
                if not os.path.isdir(os.path.split(i)[0]) :
                    os.makedirs(os.path.split(i)[0])
                cache = cache\
                [['path','parsed_text','HFSID','PERIOD_OF_REPORT','UPDATED_DAY','indexed_text','bin_score','noise_score','vader']]
                cache.to_csv(i,index=False)
        else :
            return assigned_df
