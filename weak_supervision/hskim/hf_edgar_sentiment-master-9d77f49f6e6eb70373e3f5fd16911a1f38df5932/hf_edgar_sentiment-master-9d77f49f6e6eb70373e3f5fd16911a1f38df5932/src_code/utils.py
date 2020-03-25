from __future__ import print_function
import pandas as pd
import numpy as np
import os
import re
from gensim.summarization.summarizer import summarize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def read_txt(path):
    with open(path, 'r') as f:
        str_ = f.read()
        return str_


def read_and_summarize(path_ls, ratio):
    total_ls = []
    for idx, path in enumerate(path_ls):
        print("PROGRESS : {} | TOTAL : {}".format(idx, len(path_ls)))
        str_ = read_txt(path)
        try:
            summarized = summarize(str_, ratio, split=True)
        except ValueError:
            pass
        summarized = list(zip([path for i in range(len(summarized))], summarized))
        total_ls += summarized
        df = pd.DataFrame(total_ls)

    return df


def parsing(data):
    """
    The lastest!
    """
    data = data.lower()
    data = re.sub("\n", ' ', data)
    data = re.sub("-", ' ', data)
    data = re.sub('%', ' % ', data)
    data = re.sub("$", ' $ ', data)

    line = re.sub('item\s[0-9]\s*[.]*', ' ', data)  # e.g) item 7.
    line = re.sub('\(i+\)', '', line)  # remove (iii) form
    line = re.sub('\([0-9]\)', '', line)  # remove (3) form
    line = re.sub("\$\s*\d+[,]\d+[,]\d+", ' $ ^NUM ',
                  line)  # number represents dollar and over 1,000,000 and type is int
    line = re.sub("\$\s*\d+[,]\d+", ' $ ^NUM ', line)  # number represents dollar and over 1,000 and type is int
    line = re.sub("\d+[.]\d+\s*%", ' ^NUM %', line)  # number represents percent and type is float
    line = re.sub("\d+\s*%", ' ^NUM %', line)  # number represents percent and unber 100 and type is int
    line = re.sub("\d+[,]\d+[,]\d+", ' ^NUM ', line)  # number represents nothing and over 1,000,000 and type is int
    line = re.sub("\d+[,]\d+", ' ^NUM ', line)  # number represents nothing and over 1,000 and type is int
    line = re.sub("\d+[.]\d+", ' ^NUM ', line)  # number represents nothing and type is float
    line = re.sub("\d+", ' ^NUM ', line)  # number represents nothing and type is float
    line = re.sub('\(|\)', ' ', line)  # remove parentheses
    line = re.sub("\^NUM\s+\^NUM", ' ^NUM ', line)

    filtered_ls = line.split(" ")
    for idx, token in enumerate(filtered_ls):
        if token in ["we", "us", "our", "ours"]:
            filtered_ls[idx] = ' ^COMPANY '
        elif "www" in token:
            filtered_ls[idx] = ' ^WEB '
        elif "thousand" in token:
            filtered_ls[idx] = ' ^NUM '
        elif "million" in token:
            filtered_ls[idx] = ' ^NUM '
        elif "billion" in token:
            filtered_ls[idx] = ' ^NUM '

    output = ','.join(filtered_ls).replace(",", ' ')
    output = re.sub('[^a-zA-Z\s\^\.\,]', '', output)
    output = re.sub("\^NUM\s+\^NUM", ' ^NUM ', output)
    output = re.sub("\s{2,}",' ',output)

    return output


def finer_lemmatize(word):
    word = lemmatizer.lemmatize(word, 'n')
    word = lemmatizer.lemmatize(word, 'v')
    return word


def make_chunk(data, cpu_num, dtype='list'):
    """
    chunking for making inputs of multiprocessing.
    dtype : {list,dataframe}
    """
    if dtype == 'list':
        chunk = []
        tmp = []
        for i in data:
            if len(tmp) < len(data) // cpu_num:
                tmp.append(i)
            else:
                chunk.append(tmp)
                tmp = []
        return chunk

    elif dtype == 'dataframe':
        chunk_df_ls = []
        index_ls = data.index.tolist()
        chunk = []
        tmp = []

        for i in index_ls:
            if len(tmp) < len(index_ls) // cpu_num:
                tmp.append(i)
            else:
                chunk.append(tmp)
                tmp = []

        for chunked_idx in chunk:
            chunk_df_ls.append(data.iloc[chunked_idx])
        return chunk_df_ls


def assign_ii_di(df, item='item7'):
    root_dir = "/mnt/processed/edgar_footnote_partition_indexing/word_index/"
    path_ls = []
    year_list = sorted(os.listdir(root_dir))
    for year in year_list:
        year_dir = os.path.join(root_dir, year)
        month_list = sorted(os.listdir(year_dir))
        for month in month_list:
            month_dir = os.path.join(year_dir, month)
            file_list = sorted(os.listdir(month_dir))
            for fname in file_list:
                path_ls.append(os.path.join(month_dir, fname))
    #            path_ls.append(os.path.join(month_dir,'item7.pkl'))
    path_ls = [i for i in path_ls if '{}.pkl'.format(item) in i]
    result_df = pd.DataFrame()

    for idx, fname in enumerate(path_ls):
        chunk = pd.read_pickle(fname)  # [['HFSID','UPDATED_DAY','path']]
        result_df = pd.concat([result_df, chunk], ignore_index=True)

    result_df['path'] = result_df.path.apply(lambda x: x.replace("locdisk", 'mnt'))
    merge_df = pd.merge(df, result_df, on='path')
    return merge_df


def gen_indexing(ls, g):
    for idx in range(len(ls)):
        print("PROGRESS : {} | TOTAL : {}".format(idx, len(ls)), end='\r')
        try:
            df = pd.read_csv(ls[idx])

            parsed_df = g.get_parsed(df)
            indexed_df = g.get_indexed(parsed_df)
            indexed_df = indexed_df[['path', 'parsed_text', 'HFSID', 'PERIOD_OF_REPORT', 'UPDATED_DAY', 'indexed_text']]
            indexed_df.to_csv(ls[idx], index=False)
        except Exception as e:
            print(e)
            print(ls[idx])
