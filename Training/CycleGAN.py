from matplotlib import pyplot
from numpy import ones, zeros, load, asarray
from numpy.random import randint
from random import random
from cv2 import resize
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Activation, Conv2D, LeakyReLU, Concatenate, Conv2DTranspose, Input
from tensorflow.keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import datetime


img_rows = 256
img_cols = 256
img_channels = 1
img_shape = (img_rows, img_cols, img_channels)


# define the discriminator model
def define_discriminator(image_shape, type):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # source image input
    in_image = Input(shape=image_shape)
    # C64
    d = Conv2D(64, (4, 4), strides=(
        2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4, 4), strides=(
        2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4, 4), strides=(
        2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4, 4), strides=(
        2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4, 4), padding='same',
                            kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    patch_out = Conv2D(
        1, (4, 4), padding='same', kernel_initializer=init)(d)
    # define model
    model = Model(
        in_image, patch_out, name=f"Discriminator_{type}")
    # compile model
    model.compile(loss='mse', optimizer=Adam(
        lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


# generator a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same',
                            kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(
        n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


# define the standalone generator model
def define_generator(image_shape, type, n_resnet=9):
    # weight initialization

    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # c7s1-64
    g = Conv2D(64, (7, 7), padding='same',
                            kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv2D(128, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv2D(256, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    # u128
    g = Conv2DTranspose(128, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # u64
    g = Conv2DTranspose(64, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # c7s1-3
    g = Conv2D(1, (7, 7), padding='same',
                            kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image, name=f"Generator_{type}")
    # print(model.summary())
    return model


# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gen =Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    # identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    # forward cycle
    output_f = g_model_2(gen1_out)
    # backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    # define model graph
    model = Model([input_gen, input_id], [
                               output_d, output_id, output_f, output_b])
    # define optimization algorithm configuration
    opt = Adam(lr=0.0002, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', 'mae', 'mae', 'mae'],
                  loss_weights=[1, 5, 10, 10], optimizer=opt)
    # print(model.summary())
    return model


# load and prepare training images
def load_real_samples(filename):
    # load the dataset
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape1, patch_shape2):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape1, patch_shape2, 1))
    return X, y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset, patch_shape1, patch_shape2):
    # generate fake instance
    X = g_model.predict(dataset)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape1, patch_shape2, 1))
    return X, y


def load_weights(g_model_AtoB, g_model_BtoA, d_model_A, d_model_B):
    f = open('logs.txt', 'r')
    paths = f.readlines()
    g_model_AtoB.load_weights(f'{paths[0][:-1]}')
    g_model_BtoA.load_weights(f'{paths[1][:-1]}')
    d_model_A.load_weights(f'{paths[2][:-1]}')
    d_model_B.load_weights(f'{paths[3][:-1]}')
    f.close()
    print('Weights Loaded!')
    return (g_model_AtoB, g_model_BtoA, d_model_A, d_model_B)


def losslogs(dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2):
    f = open("loss_logs.csv", "a")
    f.write(f'{dA_loss1}, {dA_loss2}, {dB_loss1}, {dB_loss2}, {g_loss1}, {g_loss2}\n')
    f.close()


# save the generator models to file
def save_weights(step, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B):

    f = open("logs.txt", "w")
    # save the first generator model
    filename1 = f'saved_model/g_model_AtoB_{step+1}.h5'
    g_model_AtoB.save_weights(filename1,save_format='h5')

    # save the second generator model
    filename2 = f'saved_model/g_model_BtoA_{step+1}.h5'
    g_model_BtoA.save_weights(filename2,save_format='h5')

    filename3 = f'saved_model/d_model_A_{step+1}.h5'
    d_model_A.save_weights(filename3,save_format='h5')

    filename4 = f'saved_model/d_model_B_{step+1}.h5'
    d_model_B.save_weights(filename4,save_format='h5')

    f.writelines([filename1+'\n', filename2+'\n', filename3+'\n', filename4+'\n'])
    print(f'>> Saved: {filename1} | {filename2} | {filename3} | {filename4}')
    f.close()


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, trainX, name, n_samples=1):
    # select a sample of input images
    X_in, _ = generate_real_samples(trainX, n_samples, 0, 0)
    # generate translated images
    X_out, _ = generate_fake_samples(g_model, X_in, 0, 0)
    # scale all pixels from [-1,1] to [0,1]
    X_in = (X_in + 1) / 2.0
    X_out = (X_out + 1) / 2.0
    # plot real images
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        temp = X_in[i]
        temp = temp.reshape((temp.shape[0], temp.shape[1]))
        temp = resize(temp, (300, 256))
        pyplot.imshow(temp,cmap='gray')
    # plot translated image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        temp = X_out[i]
        temp = temp.reshape((temp.shape[0], temp.shape[1]))
        temp = resize(temp, (300, 256))
        pyplot.imshow(temp,cmap='gray')
    # save plot to file
    filename1 = f'{name}_generated_plot_{step+1}.png'
    pyplot.tight_layout()
    pyplot.savefig(f'images_cycle/{filename1}')
    pyplot.close()


# update image pool for fake images
def update_image_pool(pool, images, max_size=50):
    selected = list()
    for image in images:
        if len(pool) < max_size:
            # stock the pool
            pool.append(image)
            selected.append(image)
        elif random() < 0.5:
            # use image, but don't add it to the pool
            selected.append(image)
        else:
            # replace an existing image and use replaced image
            ix = randint(0, len(pool))
            selected.append(pool[ix])
            pool[ix] = image
    return asarray(selected)


# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset):
    # define properties of the training run
    n_epochs, n_batch, = 30, 1
    # determine the output square shape of the discriminator
    n_patch1 = d_model_A.output_shape[1]
    n_patch2 = d_model_A.output_shape[2]
    # unpack dataset
    start_time = datetime.datetime.now()
    trainA, trainB = dataset
    datasetsize=trainA.shape[0]
    print('Dataset Size',datasetsize)
    # prepare image pool for fakes
    poolA, poolB = list(), list()
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    orignalepoch=1
    # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        X_realA, y_realA = generate_real_samples(
        trainA, n_batch, n_patch1, n_patch2)
        X_realB, y_realB = generate_real_samples(
        trainB, n_batch, n_patch1, n_patch2)

        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(
        g_model_BtoA, X_realB, n_patch1, n_patch2)
        X_fakeB, y_fakeB = generate_fake_samples(
        g_model_AtoB, X_realA, n_patch1, n_patch2)

        # update fakes from pool
        X_fakeA = update_image_pool(poolA, X_fakeA)
        X_fakeB = update_image_pool(poolB, X_fakeB)
        # update generator B->A via adversarial and cycle loss
        g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
        # update discriminator for A -> [real/fake]
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
        # update generator A->B via adversarial and cycle loss
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

        # update discriminator for B -> [real/fake]
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        elapsed_time = datetime.datetime.now() - start_time
        print(f'>> {str(i+1).center(7)} | D_A Loss: {dA_loss1:0.4f}, {dA_loss2:0.4f} | D_B Loss: {dB_loss1:0.4f}, {dB_loss2:0.4f} | G Loss: {g_loss1:0.4f}, {g_loss2:0.4f} | Elapsed Time: {elapsed_time}')

        losslogs(dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2)
        if((i+1)%datasetsize==0):
            print('|',orignalepoch,'completed! |')
            orignalepoch+=1
            
        if((i+1) % 500 == 0 or (i+1) == n_steps):
            # summarize_performance(i, g_model_AtoB, trainA, 'AtoB')
            summarize_performance(i, g_model_BtoA, trainB, 'BtoA')
            save_weights(i, g_model_AtoB, g_model_BtoA, d_model_A, d_model_B)


dataset = load_real_samples('words2handwritingresize.npz')
print('Loaded', dataset[0].shape, dataset[1].shape)
img_shape = dataset[0].shape[1:]

g_model_AtoB = define_generator(img_shape, type='A2B')
g_model_BtoA = define_generator(img_shape, type='B2A')
d_model_A = define_discriminator(img_shape, type='A')
d_model_B = define_discriminator(img_shape, type='B')

try:
    g_model_AtoB, g_model_BtoA, d_model_A, d_model_B = load_weights(g_model_AtoB, g_model_BtoA, d_model_A, d_model_B)
except:
    f = open("loss_logs.csv", "w")
    f.write('dA_loss1,dA_loss2,dB_loss1,dB_loss2,g_loss1,g_loss2\n')
    f.close()
    print('Could not load a pre-weights for the model!')

c_model_AtoB = define_composite_model(
    g_model_AtoB, d_model_B, g_model_BtoA, img_shape)

c_model_BtoA = define_composite_model(
    g_model_BtoA, d_model_A, g_model_AtoB, img_shape)

train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA,
      c_model_AtoB, c_model_BtoA, dataset)