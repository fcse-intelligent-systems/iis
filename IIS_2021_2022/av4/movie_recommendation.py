import numpy as np
import pandas as pd
from stellargraph import StellarGraph
from sklearn.model_selection import train_test_split
from stellargraph.mapper import HinSAGELinkGenerator
from stellargraph.layer import HinSAGE, LinkEmbedding
from tensorflow.keras.layers import Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy


def read_graph():
    data = pd.read_csv('ratings.csv')
    data['userId'] = data['userId'].apply(lambda x: f'user{x}')
    data['movieId'] = data['movieId'].apply(lambda x: f'movie{x}')

    users = pd.DataFrame(set(data['userId'].values.tolist()), columns=['userId'])
    users['feat1'] = np.ones(len(users))
    users['feat2'] = np.ones(len(users))
    users['feat3'] = np.ones(len(users))
    users.set_index('userId', inplace=True)

    movies = pd.DataFrame(set(data['movieId'].values.tolist()), columns=['movieId'])
    movies['feat1'] = np.ones(len(movies))
    movies['feat2'] = np.ones(len(movies))
    movies['feat3'] = np.ones(len(movies))
    movies.set_index('movieId', inplace=True)

    ratings = data.drop('timestamp', axis=1)
    ratings.columns = ['source', 'target', 'rating']
    ratings['rating'] = [1 if rating > 3 else 0 for rating in ratings['rating'].values]

    edges = ratings.drop('rating', axis=1)

    return StellarGraph({'movie': movies, 'user': users}, {'rating': edges}), ratings


if __name__ == '__main__':
    graph, ratings = read_graph()

    print(graph.info())

    train_edges, test_edges = train_test_split(ratings, test_size=0.2)
    val_edges, test_edges = train_test_split(test_edges, test_size=0.5)

    train_labels = train_edges['rating']
    val_labels = val_edges['rating']
    test_labels = test_edges['rating']

    # user123 - movie456
    train_edges = list(train_edges[['source', 'target']].itertuples(index=False))
    val_edges = list(val_edges[['source', 'target']].itertuples(index=False))
    test_edges = list(test_edges[['source', 'target']].itertuples(index=False))

    generator = HinSAGELinkGenerator(graph, batch_size=16,
                                     num_samples=[8, 4], head_node_types=['user', 'movie'])

    train_gen = generator.flow(train_edges, train_labels)
    val_gen = generator.flow(val_edges, val_labels)
    test_gen = generator.flow(test_edges, test_labels)

    hinsage = HinSAGE(layer_sizes=[16, 16], activations=['relu', 'relu'],
                      generator=generator, dropout=0.5)

    x_inp, x_out = hinsage.in_out_tensors()
    predictions = LinkEmbedding(activation='relu', method='ip')(x_out)
    predictions = Reshape((-1,))(predictions)

    predictions = Dense(1, activation='sigmoid')(predictions)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.01), loss=binary_crossentropy, metrics=['accuracy'])

    model.fit(train_gen, epochs=3, validation_data=val_gen, shuffle=False, verbose=2)

    model.evaluate(test_gen)
