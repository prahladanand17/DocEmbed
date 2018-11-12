import numpy as np
from pathlib import Path
import pandas as pd
import os, sys

'''
Helper function to transform subfolders of data in train/validation subfolders
into train and validation csvs so that the data can be loaded and prepared using
fast.ai TextClasDataBunch from_csv function
'''

def build_dataset(source_dir, train):
    categories = os.listdir(source_dir)
    texts,labels = [],[]
    label_to_idx = {}
    idx_to_label = {}
    i = 0
    for c in categories:
        print (i)
        label_to_idx[c] = i
        idx_to_label[i] = c
        folder_path = source_dir + '/' + c
        files = os.listdir(folder_path)
        for f in files:
            filepath = Path(folder_path + '/' + f)
            doc = filepath.open('r', encoding='utf8').read()
            texts.append(doc)
            labels.append(i)
        i += 1
    texts,labels = np.array(texts),np.array(labels)
    df = pd.DataFrame({'text':texts, 'labels':labels}, columns=['labels','text'])
    if train:
        df.to_csv(path_or_buf='/homes/anand39/DocEmbed/data/20_news/train.csv', index=False, header=False)
    else:
        df.to_csv(path_or_buf='/homes/anand39/DocEmbed/data/20_news/valid.csv', index=False, header=False)

if __name__ == "__main__":
    build_dataset('20_news/train', train=True)
    build_dataset('20_news/valid', train=False)
