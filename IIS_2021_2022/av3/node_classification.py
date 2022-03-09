from stellargraph.datasets import Cora
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import TSNE
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt


def plot_embeddings(data, subjects):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(data[:, 0], data[:, 1], c=subjects.astype('category').cat.codes,
               cmap='jet', alpha=0.7)
    ax.set(aspect='equal', xlabel='$X_1$', ylabel='$X_2$',
           title=f'TSNE visualization of GCN embeddings for CORA dataset')
    plt.show()


if __name__ == '__main__':
    dataset = Cora()
    G, node_subjects = dataset.load()
    print(G.info())

    train_subjects, test_subjects = train_test_split(node_subjects, test_size=0.2, stratify=node_subjects)
    val_subjects, test_subjects = train_test_split(test_subjects, test_size=0.5, stratify=test_subjects)

    label_binarizer = LabelBinarizer()
    train_targets = label_binarizer.fit_transform(train_subjects)
    val_targets = label_binarizer.transform(val_subjects)
    test_targets = label_binarizer.transform(test_subjects)

    generator = FullBatchNodeGenerator(G, method='gcn')
    train_gen = generator.flow(train_subjects.index, train_targets)
    val_gen = generator.flow(val_subjects.index, val_targets)
    test_gen = generator.flow(test_subjects.index, test_targets)

    gcn = GCN(layer_sizes=[16, 16], activations=['relu', 'relu'],
              generator=generator, dropout=0.5)

    x_inp, x_out = gcn.in_out_tensors()

    predictions = Dense(units=train_targets.shape[1], activation='softmax')(x_out)

    model = Model(inputs=x_inp, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])

    model.fit(train_gen, epochs=3, validation_data=val_gen, shuffle=False, verbose=2)

    model.evaluate(test_gen)

    all_nodes = node_subjects.index
    all_gen = generator.flow(all_nodes)

    embedding_model = Model(inputs=x_inp, outputs=x_out)
    emb = embedding_model.predict(all_gen).squeeze(0)

    tsne = TSNE(n_components=2)
    X_reduced = tsne.fit_transform(emb)

    plot_embeddings(X_reduced, node_subjects)
