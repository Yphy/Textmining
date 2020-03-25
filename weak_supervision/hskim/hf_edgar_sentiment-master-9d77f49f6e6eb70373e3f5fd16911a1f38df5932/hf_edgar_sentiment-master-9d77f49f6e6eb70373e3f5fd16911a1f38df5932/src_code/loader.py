from __future__ import print_function
import cPickle
import os
import numpy as np
import pandas as pd
from hf.framework.base_dm import BaseDM
from hf.framework.hfclndr import HFCLNDR
from hf.framework.hfsubunvrs import HFSUBUNVRS
from hf.hfss.tool.config_helper import ConfigHelper
from hf.service.module_directory import ModuleDirectory
from numba import jit
from collections import Counter
import dask

class DM(BaseDM):

    def __init__(self):

        self.__default_ffill_days = None
        self.__remove_holidays = None
        self.__make_dataframe = None
        self.__trimmed_hfsid_list = None
        self.__ffill = None
        self.__start = None
        self.__end = None

        #cache_dir = str(os.getenv("DM_CACHE_LOAD_PATH"))
        cache_dir = str(os.getenv('DM_CACHE_LOAD_PATH'))
        self.DM_dir = os.path.join(cache_dir, "hf_edgar_sentiment")
        self.__lemma_dir = os.path.join(self.DM_dir,'lemma')
        self.__model_dir = os.path.join(self.DM_dir,'model')

        self.__reverse_lemma_dict = None

        self.hf_funda_op = None
        self.__hf_funda_lib = None

        module_directory = ModuleDirectory()
        self.__hff_lib = module_directory.get_lib("lib_hff_basic")

    def load(self, start=20100101, end=40000101, remove_holidays=True, make_dataframe=False, trim=None, ffill=False):

            """
            Modify
            """
            dm_start_date = 20100101
            self.__default_ffill_days = 250
            self.__remove_holidays = remove_holidays
            self.__make_dataframe = make_dataframe
            self.__trimmed_hfsid_list = trim
            self.__ffill = ffill

            # Setting Date List
            self.__start = start
            self.__end = end
            if self.__start < dm_start_date:
                self.__start = dm_start_date

            # Setting Date List
            self.__set_date_list()

            # Setting Security List
            self.__set_security_list()

            # Setting forward-fill date_list
            self.__set_ffill_date_list()

            self.hf_funda_op = ModuleDirectory().get_lib("lib_hf_funda_op")
            self.__hf_funda_lib = ModuleDirectory().get_lib("lib_hf_funda")

            self.lemma_dict = self.get_lemma_dict()

    def date_list(self):
        return self.__date_list

    def security_list(self):
        return self.__trimmed_hfsid_list

    def __set_date_list(self):

        hfclndr = HFCLNDR()
        if self.__remove_holidays:
            self.__full_date_list = hfclndr.get_date_list("tradingdays") # total period without holliday
        else:
            self.__full_date_list = hfclndr.get_date_list("alldays") # total period with holliday

        if self.__full_date_list[-1] < self.__end:
            self.__end = self.__full_date_list[-1]
        self.__date_list = self.__full_date_list[(self.__full_date_list>=self.__start)&(self.__full_date_list<=self.__end)]

    def __set_ffill_date_list(self):
        # apply forward fill to level 2
        if self.__ffill is True:
            self.__ffill_days = self.__default_ffill_days
        elif self.__ffill is False:
            self.__ffill_days = 0
        else:
            self.__ffill_days = int(self.__ffill)

        start_idx = np.searchsorted(self.__full_date_list, self.__start) - self.__ffill_days # shift to forward day before.
        ffill_start_date = self.__full_date_list[start_idx]
        self.__ffill_date_list = self.__full_date_list[(self.__full_date_list>=ffill_start_date)&(self.__full_date_list<=self.__end)]
        self.__ffill_start = self.__ffill_date_list[0]
        self.__ffill_end = self.__ffill_date_list[-1]

    def __set_security_list(self):

        if self.__trimmed_hfsid_list is None:
            self.__trimmed_hfsid_list = HFSUBUNVRS().additive_hfsid_list()
        else:
            pass
        self.__ii_dict = {
            hfsid:ii
            for ii, hfsid in enumerate(self.__trimmed_hfsid_list)
        }

    def __convert_to_dataframe(self, input_m):

        input_m = pd.DataFrame(input_m, index=self.__ffill_date_list, columns=self.security_list())
        output_m = input_m.fillna(method='pad', limit=self.__ffill_days)
        output_m = output_m.iloc[self.__ffill_days:]

        if not self.__make_dataframe:
            output_m = output_m.values
        return output_m

    def __load_path(self,item) :
        ls = []
        if item == 'item7' :
            root_dir = os.path.join(self.DM_dir,'10-K')
        elif item == 'item2' :
            root_dir = os.path.join(self.DM_dir,'10-Q')

        year_list = sorted(os.listdir(root_dir))

        for year in year_list :
            year_path = os.path.join(root_dir,year)
            month_list = sorted(os.listdir(year_path))
            for month in month_list :
                month_path = os.path.join(year_path,month)
                file_list = sorted(os.listdir(month_path))
                for fn in file_list :
                    fn_path = os.path.join(month_path,fn)
                    ls.append(fn_path)

        yyyymmdd_ls = [int(','.join(i.split(".")[0].split("/")[5:8]).replace(",",'')) for i in sorted(ls)]
        start_idx = np.searchsorted(yyyymmdd_ls,self.__ffill_start)
        end_idx = np.searchsorted(yyyymmdd_ls,self.__ffill_end)
        path_ls = sorted(ls)[start_idx:end_idx]

        return path_ls

    def get_lemma_dict(self, reverse=False):

        with open(os.path.join(self.__lemma_dir,'lemma_dict.pkl') , 'rb') as read_pickle:
            lemma_dict = cPickle.load(read_pickle)
        if reverse:
            reverse_lemma_dict = {
                wi:lemma
                for lemma, wi in lemma_dict.iteritems()
            }
            return reverse_lemma_dict
        else:
            return lemma_dict

    def get_lv1(self,item=None,ratio=0.1,research=False):
        """
        item : [item2(=10-Q), item7(=10-K)]
        ratio : {0.1} more ratio will be updated soon
        research : if True use Dask (multi-threading) else : load by looping
        """
        self.__set_ffill_date_list()
        if item not in ['item2','item7'] :
            raise ValueError("item should be between item2 or item7")

        if research :
            print('[get_lv1] it is research mode. Please do not use research mode when you sumit.')
            container = []
        else :
            result_df = pd.DataFrame()

        total_path_ls = self.__load_path(item)

        for num,path in enumerate(total_path_ls) :
            if research :
                tmp = dask.delayed(pd.read_csv)(path)
                container.append(tmp)
            else :
                print('[get_lv1] PROGRESS : {} | TOTAL : {}'.format(num,len(total_path_ls)),end='\r')
                tmp = pd.read_csv(path)
                result_df = pd.concat([result_df,tmp],ignore_index=False)
        if research :
            result_df = dask.delayed(pd.concat)(container,ignore_index=False)
            result_df = result_df.compute()
        ### trimming ###
        print("[get_lv1] trimming...")
        result_df = result_df[result_df.HFSID.isin(self.__trimmed_hfsid_list)].reset_index(drop=True)

        return result_df

    def __make_lv2_df(self,lv1_df,col_name,func) :
        """
        func : {mean,median,max,min}
        """
        lv1_df.sort_values('UPDATED_DAY',inplace=True)
        lv1_df['UPDATED_DAY'] = \
        self.__ffill_date_list[np.searchsorted(self.__ffill_date_list,lv1_df.UPDATED_DAY,'right')-1]

        if func == 'mean' :
            lv2_df = lv1_df[col_name].groupby([lv1_df.UPDATED_DAY,lv1_df.HFSID]).mean().unstack('HFSID')
        elif func == 'median' :
            lv2_df = lv1_df[col_name].groupby([lv1_df.UPDATED_DAY,lv1_df.HFSID]).median().unstack('HFSID')
        elif func == 'max' :
            lv2_df = lv1_df[col_name].groupby([lv1_df.UPDATED_DAY,lv1_df.HFSID]).max().unstack('HFSID')
        elif func == 'min' :
            lv2_df = lv1_df[col_name].groupby([lv1_df.UPDATED_DAY,lv1_df.HFSID]).min().unstack('HFSID')
        else :
            raise ValueError("level2 function shuold be among {mean,median,max,min}")

        append_idx = set(self.__ffill_date_list).difference(set(lv2_df.index))
        append_df = \
        pd.DataFrame(np.full((len(append_idx),lv2_df.shape[1]),fill_value=np.nan),index=append_idx,columns=lv2_df.columns)
        concat_df = pd.concat((lv2_df,append_df))

        for col in self.__trimmed_hfsid_list :
            if col not in concat_df.columns :
                concat_df[col] = np.nan

        concat_df.sort_index(inplace=True)
        lv2_df = concat_df.loc[:,self.__trimmed_hfsid_list]

        return lv2_df

    def get_score(self,lv1_df=None,score='vader',func='mean',period=1,item=None,ratio=0.1,research=False) :
        """
        score : {'vader','noise_score','bin_score'}
        if you use lv1_df, then generate lv2_df speed is much faster
        but it's okay to use without lv1_df. Then just put the others.
        func : {mean,median,max,min}
        period=1 for annual data / period=2 for quarter
        """

        if type(lv1_df) != pd.core.frame.DataFrame :
            lv1_df = self.get_lv1(item=item,ratio=ratio,research=research)

        period_m = np.full((len(self.__date_list),len(self.__trimmed_hfsid_list)),fill_value=np.nan)
        use_df = lv1_df[['HFSID','UPDATED_DAY','{}'.format(score),'PERIOD_OF_REPORT']]
        val_m = self.__make_lv2_df(use_df,score,func).values
        print('[get_score] generate lv2_df is ready.')
        for hfsid, date, score, fy_ended_date in use_df.values:
            ii = self.__ii_dict[int(hfsid)]
            di = np.searchsorted(self.__ffill_date_list, date,'right')-1
            # value = val_m[di,ii]
            # if np.isnan(value):
            period_m[di, ii] = self.__hff_lib.get_period_grouping(fy_ended_date=int(fy_ended_date), period=period)
        print("[get_score] generate period_df is ready.")
        return self.__convert_to_dataframe(val_m), self.__convert_to_dataframe(period_m)

    def calculate_QoQ(self, input_tuple):

        input_tuple = self._convert_tuple(input_tuple)
        result_tuple = self.hf_funda_op.calculate_QoQ(input_tuple=input_tuple)

        if self.__make_dataframe:
            return self._convert_tuple_into_dataframe(input_tuple=result_tuple, date_list=self.date_list(), security_list=self.security_list())
        else:
            return result_tuple

    def calculate_QoQ_delta(self, input_tuple):
        input_tuple = self._convert_tuple(input_tuple)
        result_tuple =  self.hf_funda_op.calculate_QoQ_delta(input_tuple=input_tuple)

        if self.__make_dataframe:
            return self._convert_tuple_into_dataframe(input_tuple=result_tuple, date_list=self.date_list(), security_list=self.security_list())
        else:
            return result_tuple

    def calculate_YoY(self, period, input_tuple):
        input_tuple = self._convert_tuple(input_tuple)
        result_tuple =  self.hf_funda_op.calculate_YoY(period=period, input_tuple=input_tuple)

        if self.__make_dataframe:
            return self._convert_tuple_into_dataframe(input_tuple=result_tuple, date_list=self.date_list(), security_list=self.security_list())
        else:
            return result_tuple

    def calculate_YoY_delta(self, period, input_tuple):
        input_tuple = self._convert_tuple(input_tuple)
        result_tuple =  self.hf_funda_op.calculate_YoY_delta(period=period, input_tuple=input_tuple)

        if self.__make_dataframe:
            return self._convert_tuple_into_dataframe(input_tuple=result_tuple, date_list=self.date_list(), security_list=self.security_list())
        else:
            return result_tuple

    @staticmethod
    def _convert_tuple_into_dataframe(input_tuple, date_list, security_list):
        value_m = pd.DataFrame(data=input_tuple[0], index=date_list, columns=security_list)
        fy_m = pd.DataFrame(data=input_tuple[1], index=date_list, columns=security_list)

        return value_m, fy_m

    @staticmethod
    def _convert_tuple(input_tuple):

        if type(input_tuple[0]) == type(pd.DataFrame()):

            value_m = input_tuple[0].values
        else:
            value_m = input_tuple[0]

        if type(input_tuple[1]) == type(pd.DataFrame()):
            fy_m = input_tuple[1].values
        else:
            fy_m = input_tuple[1]

        return value_m,fy_m

if __name__ == '__main__':
    config_helper = ConfigHelper()
    assert config_helper.createDM(
        name = 'hf_edgar_sentiment',
        description='hf_edgar_sentiment',
        birthday='20191206',
        valid_date_from='20100101',
        module_path=os.getcwd(),
        module_file_name='loader.py',
        min_simulation_period = 260,
        update_note='FIRST UPDATE; NEXT STEP : SUPPORT MORE RARIO',
        min_attribution_ratio=0.2
    )

    assert config_helper.addAuthor(name="hyunsikkim", attribution_ratio=0.8)
    assert config_helper.addAuthor(name="hymaeng", attribution_ratio=0.1)
    assert config_helper.addLIB('lib_hf_funda_op',attribution_ratio=0.1)
    assert config_helper.overwrite()
