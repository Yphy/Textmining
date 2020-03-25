### python ###
import numpy as np
import pandas as pd
import itertools

### pytorch ###

### custom packages ###

def mapping(lemma_dict,x) :
    result = lemma_dict.get(x)
    if result == None :
        return 1 # unk
    else :
        return result

def indexing(df,lemma_dict,train=True,prunning=True,sampling='over') :
    """
    target text columns name should be 'text'
    target label columns name should be
        in case of continuous : "score"
        in case of discrete : "label"
    sampling : {over,under,None}
    """
    if train and prunning :
        print("Data is trimmed by label constraint")
        use_df = df[(df.score <= 0.25) | (df.score >= 0.75) ]

        use_df['label'] = [1 if i > 0.5 else 0 for i in use_df.score]

    else :
        use_df = df.copy()

    print("[indexing] indexed_text's dtype is np.array with dtype of int")
    use_df['indexed_text'] = use_df.parsed_text.str.split(" ").apply(lambda x : np.array(list(map(mapping,itertools.repeat(lemma_dict,len(x)),x))))
    use_df['indexed_text'] = use_df.indexed_text.apply(padding)

    if train :
        use_df['label'] = [1 if i > 0.5 else 0 for i in use_df.score]
        print("[indexing] class-imbalance degree is \n {}".format(use_df.label.value_counts()))
        if sampling == 'over':
            print("Over-Sampling is started")
            max_size = use_df['label'].value_counts().max()
            lst = [use_df]
            for class_index, group in use_df.groupby('label'):
                lst.append(group.sample(max_size-len(group), replace=True))
            final_df = pd.concat(lst,ignore_index=True)
        elif sampling == 'under' :
            min_value = use_df['label'].value_counts().min()
            tmp_ls = []
            for l in use_df.label.unique():
                tmp_ls.append(use_df[use_df.label == l].sample(min_value))
            final_df = pd.concat(tmp_ls,ignore_index=True)
        elif sampling == None :
            return use_df
    else :
        final_df = use_df.copy()

    return final_df#,train_X,train_y

def padding(arr) :
    arr = np.concatenate((arr , np.array([0] * (100-len(arr)))))
    return arr

def gen_lemma_dict(df,lemma_dict=None) :
    #corpus = np.unique([word for sent in df.text for word in sent.split(" ")],return_counts=False)
    corpus = [word for sent in df.text for word in sent.split(" ")]
    if lemma_dict == None :
        lemma_dict = {'^PAD':0,'^UNK':1}

    for word in corpus :
        try :
            lemma_dict[word] # same as isin function
        except :
            lemma_dict[word] = len(lemma_dict)

    return lemma_dict
