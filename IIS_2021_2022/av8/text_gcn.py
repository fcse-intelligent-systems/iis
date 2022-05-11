import numpy as np
import pandas as pd
from math import log
from tqdm import tqdm
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer


def create_word_word_edges(documents, vocabulary, word_to_id, sliding_window_size):
    num_windows = 0
    num_windows_i = np.zeros(len(vocabulary))
    num_windows_i_j = np.zeros((len(vocabulary), len(vocabulary)))

    for tokens in tqdm(documents, total=len(documents)):
        for window in range(max(1, len(tokens) - sliding_window_size)):
            num_windows += 1
            window_words = set(tokens[window:(window + sliding_window_size)])
            for word in window_words:
                if word in vocabulary:
                    num_windows_i[word_to_id[word]] += 1
            for word1, word2 in combinations(window_words, 2):
                if word1 in vocabulary and word2 in vocabulary:
                    num_windows_i_j[word_to_id[word1]][word_to_id[word2]] += 1
                    num_windows_i_j[word_to_id[word2]][word_to_id[word1]] += 1

    p_i_j_all = num_windows_i_j / num_windows
    p_i_all = num_windows_i / num_windows

    word_word_edges = []
    for word1, word2 in tqdm(combinations(vocabulary, 2), total=len([c for c in combinations(vocabulary, 2)])):
        p_i_j = p_i_j_all[word_to_id[word1]][word_to_id[word2]]
        p_i = p_i_all[word_to_id[word1]]
        p_j = p_i_all[word_to_id[word2]]
        val = log(p_i_j / (p_i * p_j)) if p_i * p_j > 0 and p_i_j > 0 else 0
        if val > 0:
            word_word_edges.append((word1, word2, val))

    return pd.DataFrame(word_word_edges, columns=['source', 'target', 'weight'])


def create_word_document_edges(dataset, vocabulary):
    tf_idf = TfidfVectorizer(vocabulary=vocabulary)
    tf_idf_weights = tf_idf.fit_transform(dataset['tweet'].values).toarray()
    feature_names = tf_idf.get_feature_names()
    tf_idf_weights_df = pd.DataFrame(tf_idf_weights, columns=feature_names, index=dataset.index)

    word_tweet_edges = []
    for row in tqdm(tf_idf_weights_df.iterrows(), total=len(tf_idf_weights_df)):
        non_zero_indices = row[1].array.to_numpy().nonzero()
        for i in non_zero_indices[0]:
            word_tweet_edges.append((row[0], feature_names[i], row[1][i]))

    return pd.DataFrame(word_tweet_edges, columns=['source', 'target', 'weight'])
