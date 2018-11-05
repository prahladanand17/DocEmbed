import os
import shutil
import numpy as np

'''
Helper function to split subfolders of data across num_classes into
train and test folders, each containing subfolders of data across num_classes.
Train/Validation split is 70-30.
'''

def create_data_splits(source_dir):
    train_dir = "train/" + source_dir
    os.mkdir(train_dir)
    val_dir = "valid/" + source_dir
    os.mkdir(val_dir)
    files = os.listdir(source_dir)
    for f in files:
        if np.random.rand(1) < 0.3:
            shutil.move(source_dir + '/'+ f, val_dir + '/'+ f)
        else:
            shutil.move(source_dir + "/" + f, train_dir + '/' + f)

'''
Helper function to validate splits and make sure that split is within reasonable
difference of 70-30
'''
def check_split(source_dir):
    folders = os.listdir(source_dir)
    num_train = 0
    total = 0
    for f in folders:
        subfolders = os.listdir(source_dir + "/" + f)
        for s in subfolders:
            texts = os.listdir(source_dir + "/" + f + "/" + s)
            if f == "train":
                num_train += len(texts)
            total += len(texts)
    print (num_train, total, num_train/total)

'''
Helper function to change all files into .txt format
'''

def change_to_txt(root):
    for category in os.listdir(root):
        if category != '.DS_Store':
            for file in os.listdir(root + "/" + category):
                head, tail = os.path.splitext(file)
                if not tail:
                    src = os.path.join(root + "/" + category, file)
                    dst = os.path.join(root + "/" + category, file + '.txt')
                    if not os.path.exists(dst): # check if the file doesn't exist
                        os.rename(src, dst)

if __name__ == "__main__":
    create_data_splits('business')
    create_data_splits('entertainment')
    create_data_splits('sport')
    create_data_splits('politics')
    create_data_splits('tech')
