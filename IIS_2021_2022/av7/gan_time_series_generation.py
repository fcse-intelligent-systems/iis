import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, LeakyReLU, Flatten, Input, \
    Reshape, UpSampling1D, BatchNormalization, Activation


def plot(sequence):
    sns.lineplot(x=range(30), y=sequence)
    plt.show()


def load_data():
    data = pd.read_csv('Melbourne_daily_temp.csv')

    X_train = []

    for i in range(29, len(data)):
        window = data.loc[i - 29: i]
        X_train.append(window['Temp'].values)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train


def build_discriminator(sequence_shape):
    model = Sequential()

    model.add(Conv1D(16, 3, padding='same', input_shape=sequence_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Conv1D(32, 3, padding='same', input_shape=sequence_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    seq = Input(shape=sequence_shape)
    validity = model(seq)

    return Model(seq, validity)


def build_generator(latent_dim):
    model = Sequential()

    model.add(Dense(15, input_dim=latent_dim))
    model.add(Reshape((15, 1)))

    # [2, 3]
    # [[2, 3], [2, 3]]

    model.add(UpSampling1D())
    model.add(Conv1D(64, 3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))

    model.add(Conv1D(1, 3, padding='same'))
    model.add(Activation('tanh'))

    model.summary()

    noise = Input(shape=(latent_dim,))
    seq = model(noise)

    return Model(noise, seq)


if __name__ == '__main__':
    X_train = load_data()
    plot(X_train[0])

    sequence_shape = (30, 1)
    latent_dim = 100
    batch_size = 128
    epochs = 500

    optimizer = Adam(0.0002, 0.5)

    discriminator = build_discriminator(sequence_shape)
    discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])

    generator = build_generator(latent_dim)

    noise = Input(shape=(latent_dim,))
    seq = generator(noise)

    discriminator.trainable = False

    valid = discriminator(seq)

    combined = Model(noise, valid)
    combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    X_train = np.expand_dims(X_train, axis=2)

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of sequences
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        sequences = X_train[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generate a half batch of new images
        gen_sequences = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(sequences, valid)
        d_loss_fake = discriminator.train_on_batch(gen_sequences, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator
        g_loss = combined.train_on_batch(noise, valid)

        # Plot the progress
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]' % (epoch, d_loss[0], 100 * d_loss[1], g_loss))




