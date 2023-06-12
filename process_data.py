import nltk
import numpy as np
import torch

nltk.download('stopwords')


# This function is used to clean up the dataset by removing duplicates, stop words and punctuation
def clean_data(summaries):
    corpus = list()
    corp_voc = dict()
    # We remove all English stopwords and punctuation and also tokenize all words in the summaries
    stop = nltk.corpus.stopwords.words("english")
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+(?:'\w)?")
    # cleanup for each summary
    for summary in summaries:
        summary = tokenizer.tokenize(summary)
        summary_cleaned = dict()
        for tok in summary:
            # We make all characters lower case and filter out numbers
            tok = tok.lower()
            if not tok.isdigit() and tok not in stop:
                # add clean token to summary
                if tok in summary_cleaned:
                    summary_cleaned[tok] += 1
                else:
                    summary_cleaned[tok] = 1
        for tok in summary_cleaned.keys():
            # increase corpus vocabulary
            if tok in corp_voc:
                corp_voc[tok] += 1
            else:
                corp_voc[tok] = 1
        corpus.append(summary_cleaned)

    return corpus, corp_voc


# This function applies the filtering process to each corpus from the vocabulary
def get_filtered_corpus(summaries, vocab):
    clean_corpus = list()
    for summary in summaries:
        clean_summary = dict()
        for tok in summary:
            if tok in vocab:
                clean_summary[tok] = summary[tok]
        clean_corpus.append(clean_summary)
    return clean_corpus


# This function does a onehot encoding of our genres
def to_categorical(array):
    unique, inverse = np.unique(array, return_inverse=True)
    onehot = np.eye(unique.shape[0])[inverse]
    return onehot


# This is a class which defines a custom dataset for working with TF-IDF data
class TF_IDF_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    # Gets the length
    def __len__(self):
        return len(self.Y)

    # Gets the value at certain coordinates
    def __getitem__(self, index):
        X = self.X[index].float()
        Y = self.Y[index].float()
        return X, Y
