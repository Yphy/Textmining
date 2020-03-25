### built-in ###
import sys
sys.path.append('../')
import os
import numpy as np
import pandas as pd
import time
import itertools
import argparse

### for preprocessing ###
from transformers import BertTokenizer
from gensim.summarization import summarize
import pysbd

### custom ###
import utils

class Preprocess():

    def __init__(self,path_ls):
        self.path_ls = path_ls
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add_tokens(['^NUM', '^COMPANY'])
        self.seg = pysbd.Segmenter(language="en", clean=True)
        self.lemma_dict = dict(self.tokenizer.vocab)
        self.dump_path = '/home/hyunsikkim/HF_EDGAR_SENTIMENT/data'

    def __read_txt(self, x):
        with open(x, 'r') as f:
            text = f.read()
        return text

    def __padding(self, x, max_length=100):
        pad_length = max_length - len(x)
        pad_arr = [0]
        padded_x = x + pad_arr * pad_length
        return padded_x

    def preprocess(self, path=None):
        if path == None:
            path = self.path_ls

        concat_df = pd.DataFrame()
        for idx, fname in enumerate(path):
            print("##############################################################################")
            print("PROCESS : {} | TOTAL : {}".format(idx, len(path)))
            print("##############################################################################")
            text = self.__read_txt(fname)
            parsed_text = utils.parsing(text)

            try :
                summarized_text = summarize(parsed_text)
            except ValueError :
                segmented_text = self.seg.segment(parsed_text)
                summarized_text = ','.join(segmented_text[len(segmented_text)//4:]).replace(","," ")
                print("naive assumption is applied; total sentence length : {} | extracted length : {}".format(len(segmented_text),len(segmented_text)/4))

            splitted_text = self.seg.segment(summarized_text)
            tokenized_ls = list(map(self.tokenizer.tokenize, splitted_text))
            indexed_ls = list(map(self.tokenizer.convert_tokens_to_ids, tokenized_ls))
            # TODO :CONVERT TO GENEJRATOR

            data = list(zip(itertools.repeat(fname, len(indexed_ls)), indexed_ls))

            tmp = pd.DataFrame(data, columns=['path', 'indexed_data'])
            tmp['sentence'] = splitted_text
            tmp['indexed_data'] = tmp['indexed_data'].apply(self.__padding)
            tmp['year'] = tmp.path.apply(lambda x: x.split("/")[5])

            concat_df = pd.concat([concat_df, tmp], ignore_index=True)
            if len(concat_df) > 100000:
                for year in tmp.year.unique():
                    yearly_dump_path = os.path.join(self.dump_path,'{}.csv'.format(year))

                    if os.path.exists(yearly_dump_path):
                        orgin_df = pd.read_csv(yearly_dump_path)
                        dump_df = pd.concat([orgin_df, concat_df[concat_df['year'] == year]], ignore_index=True)
                        dump_df.to_csv(yearly_dump_path, index=False)
                    else:
                        dump_df = concat_df[concat_df['year'] == year].reset_index(drop=True)
                        dump_df.to_csv(yearly_dump_path, index=False)

                concat_df = pd.DataFrame()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Type the report {10-K,10-Q}.')
    parser.add_argument('--report', type=str, default='10-K',help='It should be between 10-K or 10-Q')
    args = parser.parse_args()

    data_dir = '/mnt/processed/edgar_footnote_partition/{}'.format(args.report)
    print("##############################################################################")
    print(data_dir)
    print("##############################################################################")
    if args.report == '10-K' :
        item = 'item7.'
    else :
        item = 'item2.'
    fname_ls = []

    for year in sorted(os.listdir(data_dir)):
        year_dir = os.path.join(data_dir, year)
        for month in sorted(os.listdir(year_dir)):
            day_dir = os.path.join(year_dir, month)
            for day in sorted(os.listdir(day_dir)):
                fname_dir = os.path.join(day_dir, day)
                for fname in sorted(os.listdir(fname_dir)):
                    if item in fname:
                        fname_ls.append(os.path.join(fname_dir, fname))
    print("fname list is ready")
    time.sleep(1)

    g = Preprocess(fname_ls)
    g.preprocess()
