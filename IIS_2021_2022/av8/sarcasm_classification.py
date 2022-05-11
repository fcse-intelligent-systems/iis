import numpy as np
import pandas as pd
from stellargraph import StellarGraph
from stellargraph.mapper import HinSAGENodeGenerator
from stellargraph.layer import HinSAGE
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from preprocessing import tokenize, load_vocabulary, load_embedding_weights
from text_gcn import create_word_word_edges, create_word_document_edges


def load_data():
    dataset = pd.read_csv('train.En.csv', usecols=['tweet', 'sarcastic']).dropna()
    dataset['TweetID'] = [f'Tweet{i + 1}' for i in range(len(dataset))]
    dataset = dataset.set_index('TweetID')
    return dataset


if __name__ == '__main__':
    dataset = load_data()

    dataset = tokenize(dataset)

    tweet_labels = dataset.drop(['tokens', 'tweet'], axis=1)

    vocabulary, word_to_id, id_to_word = load_vocabulary(dataset, 1)

    embedding_weights = load_embedding_weights(vocabulary, 50)
    emb_dict = {word: embedding for word, embedding in zip(vocabulary, embedding_weights)}
    word_nodes_df = pd.DataFrame.from_dict(emb_dict, orient='index')

    tweet_dict = {}
    for tweet, i in zip(dataset['tokens'].values, dataset.index):
        tweet_vector = []
        for token in tweet:
            tweet_vector.append(emb_dict[token])
        tweet_dict[i] = np.array(tweet_vector).mean(axis=0)
    tweet_nodes_df = pd.DataFrame.from_dict(tweet_dict, orient='index')

    word_word_df = create_word_word_edges(dataset['tokens'].values, vocabulary, word_to_id, 5)
    word_word_df['ID'] = range(len(word_word_df))
    word_word_df = word_word_df.set_index('ID')

    word_tweet_df = create_word_document_edges(dataset, vocabulary)
    word_tweet_df['ID'] = range(len(word_word_df), len(word_word_df) + len(word_tweet_df))
    word_tweet_df = word_tweet_df.set_index('ID')

    graph = StellarGraph({'tweet': tweet_nodes_df,
                          'word': word_nodes_df},
                         {'word-word': word_word_df,
                          'part of': word_tweet_df})

    train_tweets, test_tweets = train_test_split(tweet_labels, test_size=0.15, stratify=tweet_labels)
    generator = HinSAGENodeGenerator(graph, batch_size=16,
                                     num_samples=[8, 8], head_node_type='tweet')
    train_gen = generator.flow(train_tweets.index, train_tweets)
    test_gen = generator.flow(test_tweets.index, test_tweets)

    hinsage_layer = HinSAGE(layer_sizes=[16, 32], generator=generator,
                            bias=True, dropout=0.1)
    x_inp, x_out = hinsage_layer.in_out_tensors()

    predictions = Dense(units=1, activation='sigmoid')(x_out)
    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy, metrics=['accuracy'])

    print(model.summary())

    model.fit(train_gen, epochs=5, verbose=2)

    model.evaluate(test_gen)
