import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path
import re, string
import unicodedata
import inflect
import nltk
import os

def test():
    doc1 = Path('/Users/anprahlad/Developer/DL_F2018/DocEmbed/data/20_news/train/alt.atheism/49960.txt').open('r', encoding='utf8').read()
    tokens = doc1.split()
    processed_tokens = normalize(tokens)
    word_to_idx = {}
    vocab_size = 1
    for t in tokens:
        if t not in word_to_idx:
            word_to_idx[t] = vocab_size
            vocab_size += 1
    sentence_idxs = torch.tensor([word_to_idx[w] for w in tokens], dtype = torch.long)
    word_embeddings = nn.Embedding(len(tokens), 10)
    embeds = word_embeddings(sentence_idxs)
    import pdb; pdb.set_trace()
    print (embeds)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def new_test():
    for i in range(0,5):
        import pdb; pdb.set_trace()


def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    return words



if __name__ == "__main__":
    new_test()
