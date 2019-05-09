import numpy as np
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Cropping2D
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam


from encoder import preproces_avs
from encoder import make_generator as make_data_generator


def general_discriminator(shape, resize):
    i = Input(shape)
    o = i
    for size, convs in resize:
        for filters, kernel_size in convs:
            o = Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=(1,1),
                    padding="same")(o)
            o = BatchNormalization()(o)
            o = LeakyReLU(alpha=0.01)(o)
        o = MaxPooling2D(pool_size=size)(o)

    o = Flatten()(o)
    o = Dense(1, activation='sigmoid')(o)
    m = Model(inputs=i, outputs=o)
    return m


def general_generator(z_dim, in_shape, resize):
    i = Input(shape=(z_dim,))
    o = Dense(in_shape[0]*in_shape[1]*in_shape[2])(i)
    o = Reshape(in_shape)(o)
    for size, convs in resize[:-1]:
        o = UpSampling2D(size=size)(o)
        for filters, kernel_size in reversed(convs):
            o = Conv2DTranspose(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=(1,1),
                    padding="same")(o)
            o = BatchNormalization()(o)
            o = LeakyReLU(alpha=0.01)(o)

    size = resize[-1][0]
    o = UpSampling2D(size=size)(o)
    filters, kernel_size = resize[-1][1][0]
    o = Conv2DTranspose(
            activation="tanh",
            filters=filters,
            kernel_size=kernel_size,
            strides=(1,1),
            padding="same")(o)

    m = Model(inputs=i, outputs=o)
    return m


def make_discriminator(shape):
    i = Input(shape)
    o = Conv2D(
            filters=3,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(i)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = MaxPooling2D(pool_size=(2,2))(o)
    o = Conv2D(
            filters=64,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = Conv2D(
            filters=128,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = Conv2DTranspose(
            filters=128,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = MaxPooling2D(pool_size=(2,2))(o)
    o = Conv2DTranspose(
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = Conv2DTranspose(
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = Conv2DTranspose(
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = MaxPooling2D(pool_size=(3,3))(o)
    o = Conv2DTranspose(
            filters=256,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = MaxPooling2D(pool_size=(5,5))(o)
    o = Conv2DTranspose(
            filters=512,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = MaxPooling2D(pool_size=(5,5))(o)

    o = Flatten()(o)
    o = Dense(1, activation='sigmoid')(o)

    m = Model(inputs=i, outputs=o)
    return m


def make_generator(z_dim):
    i = Input(shape=(z_dim,))
    o = Dense(128)(i)
    o = Reshape((1,1,128))(o)
    o = UpSampling2D(size=(5,5))(o)
    # 5, 5, 2048
    o = Conv2DTranspose(
            filters=512,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = UpSampling2D(size=(5,5))(o)
    o = Conv2DTranspose(
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = UpSampling2D(size=(3,3))(o)
    o = Conv2DTranspose(
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = Conv2DTranspose(
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = Conv2DTranspose(
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    # 75, 75, 256
    o = UpSampling2D(size=(2,2))(o)
    # 150, 150, 256
    o = Conv2DTranspose(
            filters=128,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = Conv2DTranspose(
            filters=128,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    o = Conv2DTranspose(
            filters=64,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = BatchNormalization()(o)
    o = LeakyReLU(alpha=0.01)(o)
    # 150, 150, 128
    o = UpSampling2D(size=(2,2))(o)
    # 300, 300, 128
    o = Conv2DTranspose(
            activation="tanh",
            filters=3,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    # 300, 300, 3

    m = Model(inputs=i, outputs=o)
    return m


def train(generator, discriminator, imag_gen, batch_size, cycles, sub_cyc, sample_interval):

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for iteration in range(cycles):
        # -------------------------
        #  Train the Discriminator
        # -------------------------
        imgs = next(imag_gen)

        z = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(z)

        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train the Generator
        # ---------------------
        discriminator.trainable = False
        combined.compile(loss='binary_crossentropy', optimizer=Adam())
        for gen_it in range(int(100*d_loss[1]) + 1):
            z = np.random.normal(0, 1, (batch_size, 100))
            #gen_imgs = generator.predict(z)
            g_loss = combined.train_on_batch(z, real)
        print ("----- [G loss: %f]" % g_loss)
        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy', 
                          optimizer=Adam(), metrics=['accuracy'])

        if iteration % sample_interval == 0:
            # Output training progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % 
                         (iteration, d_loss[0], 100*d_loss[1], g_loss))


if __name__ == "__main__":
    img_shape = (200, 200, 3)
    z_dim = 100
    # Build and compile the Discriminator
    #disc_shape = [
    #((2,2), [(3,(2,2))]),
    #((2,2), [(64,(2,2)), (128, (2,2)), (128, (2,2))]),
    #((2,2), [(128,(2,2)), (256, (2,2)), (256, (2,2))]),
    #((2,2), [(256,(2,2)), (512,(2,2))]),
    #((2,2), [(512,(2,2)), (1024,(2,2))])]
    disc_shape = [
    ((2,2), [(3,(2,2))]),
    ((2,2), [(32,(2,2)), (64, (2,2)), (64, (2,2))]),
    ((2,2), [(64,(2,2)), (128, (2,2)), (128, (2,2)), (128, (2,2))]),
    ((5,5), [(128,(5,5)), (256,(5,5)), (256,(5,5))]),
    ((5,5), [(256,(5,5)), (512,(5,5))])]
    discriminator = general_discriminator(img_shape, disc_shape)
    discriminator.compile(loss='binary_crossentropy', 
                          optimizer=Adam(), metrics=['accuracy'])
    
    # Build the Generator
    generator = general_generator(z_dim, discriminator.layers[-3].output_shape[1:], list(reversed(disc_shape)))
    
    # Generated image to be used as input
    z = Input(shape=(z_dim,))
    img = generator(z)
    
    # Keep Discriminator’s parameters constant during Generator training
    #discriminator.trainable = False
    
    # The Discriminator’s prediction
    prediction = discriminator(img)
    
    # Combined GAN model to train the Generator
    combined = Model(z, prediction)
    combined.compile(loss='binary_crossentropy', optimizer=Adam())

    batch_size = 8
    shape = img_shape[:-1]
    pre = preproces_avs(shape=shape)
    pre_for_gen = lambda x : pre.transform(x.reshape((1, shape[0]*shape[1]*3))).reshape((1,)+ shape +(3,))
    imag_gen = make_data_generator(target_size=shape, batch_size=batch_size, class_mode=None, preprocessing_function=pre_for_gen)
    train(generator, discriminator, imag_gen, batch_size, 1000, 25, 5)
