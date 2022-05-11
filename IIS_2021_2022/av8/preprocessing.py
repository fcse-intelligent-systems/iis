import os
import numpy as np
import _pickle as pickle
from collections import Counter
from nltk.tokenize import word_tokenize


def tokenize(dataset):
    dataset['tokens'] = dataset['tweet'].apply(lambda x: word_tokenize(x.lower()))
    return dataset


def load_vocabulary(dataset, frequency):
    vocabulary = [t for tokens in dataset['tokens'].values for t in tokens]
    counts = Counter(vocabulary)
    vocabulary = [c for c in counts if counts[c] >= frequency]
    word_to_id = {w: i for w, i in zip(vocabulary, range(len(vocabulary)))}
    id_to_word = {i: w for w, i in zip(vocabulary, range(len(vocabulary)))}

    return vocabulary, word_to_id, id_to_word


def load_embeddings(file_name, vocabulary):
    """
    Loads word embeddings from the file with the given name.
    :param file_name: name of the file containing word embeddings
    :type file_name: str
    :param vocabulary: captions vocabulary
    :type vocabulary: numpy.array
    :return: word embeddings
    :rtype: dict
    """
    embeddings = dict()
    with open(file_name, 'r', encoding='utf-8') as doc:
        line = doc.readline()
        while line != '':
            line = line.rstrip('\n').lower()
            parts = line.split(' ')
            vals = np.array(parts[1:], dtype=np.float)
            if parts[0] in vocabulary:
                embeddings[parts[0]] = vals
            line = doc.readline()
    return embeddings


def load_embedding_weights(vocabulary, embedding_size):
    """
    Creates and loads embedding weights.
    :param vocabulary: vocabulary
    :type vocabulary: numpy.array
    :param embedding_size: embedding size
    :type embedding_size: int
    :return: embedding weights
    :rtype: numpy.array
    """
    if os.path.exists(f'embedding_matrix_{embedding_size}.pkl'):
        with open(f'embedding_matrix_{embedding_size}.pkl', 'rb') as f:
            embedding_matrix = pickle.load(f)
    else:
        print('Creating embedding weights...')
        embeddings = load_embeddings(f'glove.6B.{embedding_size}d.txt', vocabulary)
        embedding_matrix = np.zeros((len(vocabulary), embedding_size))
        for i in range(len(vocabulary)):
            if vocabulary[i] in embeddings.keys():
                embedding_matrix[i] = embeddings[vocabulary[i]]
            else:
                embedding_matrix[i] = np.random.standard_normal(embedding_size)
        with open(f'embedding_matrix_{embedding_size}.pkl', 'wb') as f:
            pickle.dump(embedding_matrix, f)
    return embedding_matrix
