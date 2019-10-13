import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input,MaxPooling2D
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.utils import to_categorical 

K.common.set_image_dim_ordering('th')

latent_dim = 100+10

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, np.newaxis, :, :]
y_train = to_categorical(y_train)

adam = Adam(lr=0.0002, beta_1=0.5)

generator = Sequential()

# Transforms the input into a 7 × 7 128-channel feature map
generator.add(Dense(128*7*7, input_dim=latent_dim))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 7, 7)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))

# Produces a 28 × 28 1-channel feature map (shape of a MNIST image)
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
print(generator.summary())
generator.compile(loss='binary_crossentropy', optimizer=adam)

# Make Discriminator Model
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                         input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid',name="dis_output"))
print(discriminator.summary())
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Make Classifier Model
classifier = Sequential()
classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                 input_shape=(1,28,28)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
classifier.add(MaxPooling2D(pool_size=(2,2)))    
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(10, activation='softmax',name="class_output"))
print(classifier.summary())
print(classifier.name)
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Creating the Adversarial Network. We need to make the Discriminator weights
# non trainable. This only applies to the GAN model.
discriminator.trainable = False
classifier.trainable = False
ganInput = Input(shape=(latent_dim,))
x = generator(ganInput)
gan = Model(inputs=ganInput, outputs=[discriminator(x),classifier(x)])
losses = {
	discriminator.name: "binary_crossentropy",
	classifier.name: "categorical_crossentropy",
}
lossWeights = {discriminator.name: 1.0, classifier.name: 1.0}
gan.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)

dLosses = []
gLosses = []
cLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.plot(cLosses, label='Classifier loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim-10])
    labels = None
    for i in range(10):
      for j in range(10):
        if labels is None:
          labels = np.array([[int(i==k) for k in range(10)]])
        else:
          labels = np.concatenate((labels,np.array([[int(i==k) for k in range(10)]])),axis=0)
    print(labels.shape)
    noise = np.concatenate((noise,labels),axis=1)
    
    generatedImages = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/dcgan_discriminator_epoch_%d.h5' % epoch)
    classifier.save('models/dcgan_classifier_epoch_%d.h5' % epoch)

"""## Train our GAN and Plot the Synthetic Image Outputs 

After each consecutive Epoch we can see how synthetic images being improved
"""

epochs = 20
batchSize = 128
batchCount = X_train.shape[0] / batchSize

print('Epochs:', epochs)
print('Batch size:', batchSize)
print('Batches per epoch:', batchCount)

for e in range(1, epochs+1):
    print('-'*15, 'Epoch %d' % e, '-'*15)
    for i in tqdm(range(int(batchCount))):
        # Get a random set of input noise and images
        noise = np.random.normal(0, 1, size=[batchSize, latent_dim])
        random_index = np.random.randint(0, X_train.shape[0], size=batchSize)
        imageBatch = X_train[random_index]
        labels = y_train[random_index]

        # Generate fake MNIST images
        generatedImages = generator.predict(noise)
        X = np.concatenate([imageBatch, generatedImages])

        # Labels for generated and real data
        yDis = np.zeros(2*batchSize)
        # One-sided label smoothing
        yDis[:batchSize] = 0.9

        # Train discriminator
        discriminator.trainable = True
        dloss = discriminator.train_on_batch(X, yDis)
        
        # Train Classifier
        classifier.trainable = True
        closs,_ = classifier.train_on_batch(np.concatenate([imageBatch]),labels)
        
        # Train generator
        noise = np.random.normal(0, 1, size=[batchSize, latent_dim-10])
        noise = np.concatenate((noise,labels),axis=1)
        yGen = np.ones(batchSize)
        discriminator.trainable = False
        classifier.trainable = False
        gloss = gan.train_on_batch(noise, {discriminator.name:yGen,classifier.name:labels})

    # Store loss of most recent batch from this epoch
    dLosses.append(dloss)
    gLosses.append(gloss)
    cLosses.append(closs)
 
    if e == 1 or e % 5 == 0:
        # Plot losses from every epoch
        plotGeneratedImages(e)
        plotLoss(e)
        saveModels(e)


