from os import listdir
from os.path import join
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model, Input
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Dense, Dropout, Flatten, Reshape, Conv2DTranspose, Cropping2D
from sklearn.preprocessing import MaxAbsScaler

# 475.3721287128713
# 495.79139603960397

def preproces_avs(path="food-101/images/", shape=(80,80), batch_size=1000):
    pre = MaxAbsScaler()
    imag_gen = ImageDataGenerator()
    gen = imag_gen.flow_from_directory(path, target_size=shape, batch_size=batch_size, class_mode=None)
    flat = shape[0] * shape[1] * 3
    pre.fit(next(gen).reshape((batch_size, flat)))
    return pre


def text_example(mode, image_path, save_to="./"):
    i = Image.open(image_path).resize((256,256))
    pred = model.predict(np.asanyarray(i).reshape((1,256,256,3)))
    ii = Image.fromarray(pred.reshape(256,256,3), "RGB")
    if save_to:
        i.save(save_to+"origin", format="jpeg")
        ii.save(save_to+"pred", format="jpeg")
    return pred


def make_generator(target_size=(224,224), batch_size=32, validation_split=0.0, class_mode="categorical", preprocessing_function=lambda x:x):
    imag_gen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rotation_range=45, width_shift_range=0.2,
            height_shift_range=0.2, brightness_range=None,
            shear_range=0.05, zoom_range=0.1,
            channel_shift_range=0.0, fill_mode='nearest',
            cval=0.0, horizontal_flip=True,
            vertical_flip=False, validation_split=validation_split)
    gen = imag_gen.flow_from_directory(
            "food-101/images/", target_size=target_size, batch_size=batch_size, class_mode=class_mode)
    return gen


def make_vgg():
    vgg = vgg16.VGG16(weights="imagenet")
    vgg.layers.pop()
    vgg.layers.pop()
    vgg.layers.pop()
    o = Dense(4012)(vgg.layers[-1].output)
    o = Dense(4012)(o)
    o = Dense(101, activation="softmax")(o)

    model = Model(inputs=vgg.input, outputs=o)
    for l in model.layers[:-3]:
        l.trainable = False

    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def make_inception(in_shape=(299,299)):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=in_shape + (3,)))
    x = base_model.output
    #x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.4)(x)
    x = Flatten()(x)
    predictions = Dense(101, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)
    
    model = Model(input=base_model.input, output=predictions)
    
    opt = SGD(lr=.01, momentum=.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def reverse_vgg(model):
    for l in model.layers:
        l.trainable = False

    o = Dense(2048)(model.layers[-2].output)
    o = Reshape((1,1,2048))(o)
    o = UpSampling2D(size=(5,5))(o)
    # 5, 5, 2048
    o = Conv2DTranspose(
            activation="relu",
            filters=2048,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=1024,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same")(o)
    # 5, 5, 1024
    o = UpSampling2D(size=(5,5))(o)
    # 25, 25, 1024
    o = Conv2DTranspose(
            activation="relu",
            filters=1024,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=512,
            kernel_size=(5,5),
            strides=(1,1),
            padding="same")(o)
    # 25, 25, 512
    o = UpSampling2D(size=(3,3))(o)
    # 75, 75, 512
    o = Conv2DTranspose(
            activation="relu",
            filters=512,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=512,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    # 75, 75, 256
    o = UpSampling2D(size=(2,2))(o)
    # 150, 150, 256
    o = Conv2DTranspose(
            activation="relu",
            filters=256,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=256,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=128,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    # 150, 150, 128
    o = UpSampling2D(size=(2,2))(o)
    # 300, 300, 128
    o = Conv2DTranspose(
            activation="relu",
            filters=3,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    # 300, 300, 3
    o = Cropping2D(((1,0),(1,0)))(o)
    # 299, 299, 3

    m = Model(inputs=model.input, outputs=o)
    m.compile(optimizer='adadelta', loss='binary_crossentropy')
    return m


def reverse_vgg_mini(model):
    for l in model.layers:
        l.trainable = False

    o = Dense(2048)(model.layers[-2].output)
    o = Reshape((1,1,2048))(o)
    o = UpSampling2D(size=(5,5))(o)
    # 5, 5, 2048
    o = Conv2DTranspose(
            activation="relu",
            filters=512,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = UpSampling2D(size=(5,5))(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = UpSampling2D(size=(3,3))(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=256,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding="same")(o)
    # 75, 75, 256
    o = UpSampling2D(size=(2,2))(o)
    # 150, 150, 256
    o = Conv2DTranspose(
            activation="relu",
            filters=128,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=128,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    o = Conv2DTranspose(
            activation="relu",
            filters=64,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    # 150, 150, 128
    o = UpSampling2D(size=(2,2))(o)
    # 300, 300, 128
    o = Conv2DTranspose(
            activation="sigmoid",
            filters=3,
            kernel_size=(2,2),
            strides=(1,1),
            padding="same")(o)
    # 300, 300, 3
    o = Cropping2D(((1,0),(1,0)))(o)
    # 299, 299, 3

    m = Model(inputs=model.input, outputs=o)
    m.compile(optimizer='adadelta', loss='mse')
    return m


def autoencoder():
    imag_gen = ImageDataGenerator()
    gen = imag_gen.flow_from_directory(
            "food-101/images/",
            class_mode="input")
    vgg = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
    ccc = Conv2D(filters=2, kernel_size=(3,3), padding="same")
    ccc = Conv2D(filters=2, kernel_size=(3,3), padding="same")
    model = Sequential(vgg.layers)
    for l in model.layers:
        l.trainable = False

    for l in reversed(model.layers):
        if isinstance(l, MaxPooling2D):
            model.add(UpSampling2D(
                size=l.pool_size))
        elif isinstance(l, Conv2D):
            model.add(Conv2D(
                activation="relu",
                filters=l.filters,
                kernel_size=l.kernel_size,
                strides=l.strides,
                padding="same"))

    model.add(Conv2D(filters=3, kernel_size=(3, 3), padding="same"))
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model

    model.fit_generator(gen, steps_per_epoch=500)#101000/32)

    for l in model.layers:
        l.trainable = True
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.fit_generator(gen, steps_per_epoch=500)#101000/32)

if __name__ == "__main__":
    batch = 64
    shape = (200, 200)
    pre = preproces_avs(shape=shape, batch_size=1050)
    pre_for_gen = lambda x : pre.transform(x.reshape((1, shape[0]*shape[1]*3))).reshape((1,)+ shape +(3,))
    gen = make_generator(shape, batch,
            preprocessing_function=pre_for_gen)
    model = make_inception(shape)
    model.fit_generator(gen, steps_per_epoch=101000/265, epochs=1,
            max_queue_size=10, workers=2, use_multiprocessing=True)
