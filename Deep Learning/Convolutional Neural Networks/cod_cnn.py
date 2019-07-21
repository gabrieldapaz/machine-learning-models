from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()

#convolução
#por estar usando tensorflow backend o formato de entrada do input_shape eh assim, ao contrario do que esta no 'help'
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# segunda camada de convolucao
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening
classifier.add(Flatten())

#Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN. sthocastic gradient descent - adam
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

""" tem que pre processar as imagens para previnir o overfiting
por ex: quando a accuray no train set eh maior que no test set

quando se trata de imagens, o modelo de machine learning tem que encontrar padroes
em varios pixels de imagens e nao apenas entre variaveis independentes, por isso
eh necessario muitas imagens para o treino, afinal um dos motivos para o overfitting
eh quando se tem pouco dados

nessa base tem 10000 fotos porem ainda eh pouco, por isso precisamos dessa etapada
de aumento, em que esse algoritmo ira alterar as fotos, rotacionando, modificando elas
para termos mais imagens"""