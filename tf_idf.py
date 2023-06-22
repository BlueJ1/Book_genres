import numpy as np


# This function calculates the TF-IDF values for every token in the document
# corpus must be a list of dicts of form (token: occurences)
# vocab must be a dict of form (token: documents in corpus containing token)
def preprocess_tf_idf(corpus, vocab):
    processed = np.zeros((len(corpus), len(vocab)), dtype=np.float32)
    idf = get_idf(vocab)
    token_order = {tok: i for i, tok, in enumerate(sorted(vocab.keys()))}
    for n_doc, doc in enumerate(corpus):
        tf = get_tf(doc)
        for tok in set(doc):
            tok_pos = token_order[tok]
            processed[n_doc][tok_pos] = tf[tok]*idf[tok]
    return processed


# This function calculates the TF value of each token from the document by
# dividing its occurrence count by the total number of tokens in the document
# and returns a dictionary containing the TF values for each token in the vocabulary.
def get_tf(doc):
    tf = dict()
    for tok, occ in doc.items():
        tf[tok] = occ / len(doc)
    return tf


# This function calculates the IDF value for each token based on the inverse
# ratio of the token's occurrence across the document collection and returns
# a dictionary containing the IDF values for each token in the vocabulary.
def get_idf(corp_voc):
    idf = dict()
    for tok, docs_containing in corp_voc.items():
        idf[tok] = np.log10(len(corp_voc) / docs_containing)
    return idf
