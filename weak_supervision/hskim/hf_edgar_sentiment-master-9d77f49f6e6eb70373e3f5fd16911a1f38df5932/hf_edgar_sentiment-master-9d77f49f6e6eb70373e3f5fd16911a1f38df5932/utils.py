import pandas as pd
import numpy as np
import os
import re
from dateutil.parser import parse
import torch 

def parsing(data) : 
    data = data.lower()
    data = re.sub("\n",' ',data)
    data = re.sub("-",' ',data)
    data = re.sub('%',' % ',data)
    data = re.sub("$",' $ ',data)

    line = re.sub('item\s[0-9]\s*[.]*',' ',data) # e.g) item 7.
    line = re.sub('\(i+\)','',line) # remove (iii) form
    line = re.sub('\([0-9]\)','',line) # remove (3) form
    line = re.sub("\$\s*\d+[,]\d+[,]\d+",' $ ^NUM ',line) # number represents dollar and over 1,000,000 and type is int
    line = re.sub("\$\s*\d+[,]\d+",' $ ^NUM ',line) # number represents dollar and over 1,000 and type is int
    line = re.sub("\d+[.]\d+\s*%",' ^NUM %',line) # number represents percent and type is float
    line = re.sub("\d+\s*%",' ^NUM %',line) # number represents percent and unber 100 and type is int
    line = re.sub("\d+[,]\d+[,]\d+",' ^NUM ',line) # number represents nothing and over 1,000,000 and type is int
    line = re.sub("\d+[,]\d+",' ^NUM ',line) # number represents nothing and over 1,000 and type is int
    line = re.sub("\d+[.]\d+",' ^NUM',line) # number represents nothing and type is float
    line = re.sub('\(|\)',' ',line) # remove parentheses

#    month_ls = ['january','feburary','march','april','may','june','july','august','september','october','november','december',\
#               'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
#    for month in month_ls :
#        regex = month + "\s*[0-9]{,2}\s*[,]*\s*20[0-9]{2}" # e.x) jan 12, 2019
#        line = re.sub(regex,' ^DATE ',line)

#    filtered_ls = []
#    for idx in range(0,len(line.split(" ")),3) :
#        tmp = ','.join(line.split(" ")[idx:idx+3]).replace(","," ")
#        try :
#            parse(tmp) # if there is no day string type. raise error
#            filtered_ls += [" ^DATE "]
#        except :
#            filtered_ls += line.split(" ")[idx:idx+3]
    filtered_ls = line.split(" ")
    for idx,token in enumerate(filtered_ls) :
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
            
    return ','.join(filtered_ls).replace(",",' ')


def read_txt(path) : 
    with open(path,'r') as f :
        str_ = f.read()
    return str_

def read_and_summarize(path_ls) : 
    total_ls = []
    for idx,path in enumerate(path_ls) : 
        print((idx,len(path_ls)))
        str_ = read_txt(path)
        try : 
            summarized = summarize(str_,0.1,split=True)
        except ValueError : 
            pass
        summarized = list(zip([path_ls for i in range(len(summarized))],summarized))
        total_ls += summarized
    df = pd.DataFrame(total_ls)
    csv_path = os.path.split(path)[1].split('.')[0] + '.csv'
    csv_path = os.path.join('/home/hyunsikkim/HF_EDGAR_SENTIMENT',csv_path)
    df.to_csv("{}.csv".format(csv_path),index=False)
    
def make_chunk(ls,cpu_num) : 
    chunk = []
    tmp = []
    for i in ls : 
        if len(tmp) < len(ls)// cpu_num :
            tmp.append(i)
        else : 
            chunk.append(tmp)
            tmp = []
    return chunk

def padding(arr) : 
    arr = arr + [0] * (100-len(arr)) 
    return arr

def gen_lemma_dict(df) : 
    corpus,count = np.unique([word for sent in use_df.text for word in sent.split(" ")],return_counts=True)
    lemma_dict = {'pad':0,'unk':1}
    
    for word in corpus : 
        lemma_dict[word] = len(lemma_dict)

    word_to_freq_dict = dict(zip(corpus,count))
    
    return lemma_dict,word_to_freq_dict 

def indexing(df,lemma_dict,train=True) : 
    """
    target text columns name should be 'text'
    target label columns name should be 
        in case of continuous : "score"
        in case of discrete : "label"
    """
    if train : 
        print("Data is trimmed by label constraint")
        use_df = df[(df.score <= 0.25) | (df.score >= 0.75) ]
        use_df[use_df.score > 0.5].shape[0],use_df[use_df.score < 0.5].shape[0]

        use_df['label'] = [1 if i > 0.5 else 0 for i in use_df.score]
        max_size = use_df['label'].value_counts().max()
        
    else : 
        use_df = df.copy()
    
    use_df['indexed_text'] = use_df.text.str.split(" ").apply(lambda x : list(map(lemma_dict.get,x)))
    use_df['indexed_text'] = use_df.indexed_text.apply(padding)
    
    if train : 
        print("Over-Sampling is started")
        lst = [use_df]
        for class_index, group in use_df.groupby('label'):
            lst.append(group.sample(max_size-len(group), replace=True))
        final_df = pd.concat(lst)
    
    else : 
        final_df = use_df.copy()

    train_X = torch.tensor(list(final_df.indexed_text.apply(lambda x : np.array(x,dtype=float)).values))
    train_X = train_X.type(torch.float)
    
    if train :         
        train_y = torch.tensor(final_df.label.values)
    else : 
        train_y = None

    return final_df,train_X,train_y
        
def assign_ii_di(df) : 
    root_dir = "/mnt/processed/edgar_footnote_partition_indexing/word_index/"
    path_ls = []
    year_list = sorted(os.listdir(root_dir))
    for year in year_list : 
        year_dir = os.path.join(root_dir,year)
        month_list = sorted(os.listdir(year_dir))
        for month in month_list : 
            month_dir = os.path.join(year_dir,month)
            file_list = sorted(os.listdir(month_dir))
            for fname in file_list : 
                path_ls.append(os.path.join(month_dir,fname))
#            path_ls.append(os.path.join(month_dir,'item7.pkl'))
    path_ls = [i for i in path_ls if 'item7.pkl' in i]
    result_df = pd.DataFrame()

    for idx,fname in enumerate(path_ls) : 
        chunk = pd.read_pickle(fname)[['HFSID','UPDATED_DAY','path']]
        result_df = pd.concat([result_df,chunk],ignore_index=True)

    result_df['path'] = result_df.path.apply(lambda x : x.replace("locdisk",'mnt'))
    merge_df = pd.merge(df,result_df,on='path')
    
    print("ORGIN : {} | ASSIGNED : {}".format(df.shape, result_df.shape))
    
    return merge_df
        