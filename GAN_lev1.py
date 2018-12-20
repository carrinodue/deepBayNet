# -*- coding: utf-8 -*-
""" Simple implementation of Generative Adversarial Neural Network """

import numpy as np

import h5py

from keras.layers import Dense, Reshape, Flatten, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
from matplotlib import cm

plt.switch_backend('agg')   # allows code to run without a system DISPLAY


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=50, height=80, channels=1):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.channels, self.height, self.width)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)


    def __generator(self):
        """ Declare generator """
        # it should generates images in (1,80,50)

        model = Sequential()
        # FC 1: 12,20,16
        model.add(Dense(3840, input_shape=(100,)))
        model.add(Reshape((12, 20, 16)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2)) #

        # Conv 1: 24,40,32
        model.add(Conv2DTranspose(32, (5, 5), strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Conv 2: 50, 80, 1
        model.add(ZeroPadding2D(((0, 1), (0, 0))))
        model.add(Conv2DTranspose(1, (5, 5), strides=2, padding='same', activation='tanh'))

        # model.summary()


        return model

    def __discriminator(self):
        """ Declare discriminator """
        # classifies images in (1,80,50)
        model = Sequential()
        # Conv 1: 40,25,32
        model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', input_shape=(50, 80, 1)))
        model.add(LeakyReLU(alpha=0.2))

        # Conv 2: 20,13,16
        model.add(Conv2D(16, kernel_size=5, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # FC 1
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense(729, input_shape=self.shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        # Output
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        return model

    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X_train, y_train, scaler_X, scaler_y, epochs=20000, batch=32, save_interval=100):

        for cnt in range(epochs):

            ## train discriminator
            random_index = np.random.randint(0, len(y_train) - batch/2)
            legit_images = y_train[random_index : random_index + batch//2].reshape(batch//2, self.width, self.height, self.channels)

            gen_noise = np.random.normal(0, 1, (batch//2, 100))
            syntetic_images = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((batch//2, 1)), np.zeros((batch//2, 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)


            # train generator

            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)


    def plot_images(self, save2file=False, samples=1, step=0):
        ''' Plot and generated images '''
        filename = "./imagesEli/slowness_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        image = self.G.predict(noise)

        # Make plot with vertical (default) colorbar
        fig, ax = plt.subplots()

        image = np.reshape(image, [50, 80]).transpose()
        cax = ax.imshow(image, cmap=cm.coolwarm)
        ax.set_title('generated')
        # Add colorbar, make sure to specify tick locations to match desired ticklabels
        cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    #load data

    h5_TTfile = h5py.File('./DataEli/simulatedTT.h5')
    h5_Vfile = h5py.File('./DataEli/simulatedV.h5')

    TTdata = h5_TTfile.get('travel_time_Y') # my X
    Vdata = h5_Vfile.get('slowness_X') # my y

    TTdata = np.transpose(np.array(TTdata))
    Vdata = np.transpose(np.array(Vdata))

    print(TTdata.shape)
    print(Vdata.shape)

    Vdata_rect = np.reshape(Vdata, [-1, 50, 80])
    # Make plot with vertical (default) colorbar
    fig, ax = plt.subplots()
    data = Vdata_rect[0]
    data = np.reshape(data, [50, 80]).transpose()
    cax = ax.imshow(data, cmap=cm.coolwarm)
    ax.set_title('before prep')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

    filename = "./imagesEli/y_sample_before_prep.png"
    plt.savefig(filename)

    # Preprocessing: train_test_split...
    test_size = 0.20
    seed = 46
    X_train, X_test, Y_train, Y_test = train_test_split(TTdata, Vdata, test_size=test_size, shuffle=False)#random_state=seed)

    # ...and normalization
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(Y_train)
    y_test_scaled = scaler_y.transform(Y_test)

    Vdata_rect = np.reshape(y_train_scaled, [-1, 50, 80])
    # Make plot with vertical (default) colorbar
    fig, ax = plt.subplots()
    data1 = Vdata_rect[0]
    data1 = np.reshape(data1, [50, 80]).transpose()
    cax = ax.imshow(data1, cmap=cm.coolwarm)
    ax.set_title('after prep')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

    filename = "./imagesEli/y_sample_after_prep.png"
    plt.savefig(filename)


    gan = GAN()
    gan.train(X_train_scaled, y_train_scaled, scaler_X, scaler_y)