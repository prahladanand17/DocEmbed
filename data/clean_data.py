import os
from pathlib import Path

'''
Helper function to identify location/number of files with non unicode data
to make manual cleaning of data easier so that data can be loaded into fast.ai
TextClasDataBunch
'''

def clean_data(data_root):
    subfolders = os.listdir(data_root)
    file_count = 0
    write_file = open('file_list.txt', 'w')
    for s in subfolders:
        folder_path = data_root + "/" + s
        text_files = os.listdir(folder_path)
        for t in text_files:
            filepath = folder_path + "/" + t
            try:
                file = Path(filepath).open('r', encoding='utf8').read()
            except:
                print (s,t)
                write_file.write(str(s) + " " + str(t) + "\n")
                file_count += 1
    print file_count

if __name__ == "__main__":
    clean_data('20_news/train')
