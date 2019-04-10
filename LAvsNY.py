#created by: Andrew Chabot
#Computer Vision Final Project Proof of Concept
import os
import numpy as np
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU

def LAvsNY():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(100, activation=LeakyReLU(alpha=0.3)))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(50, activation=LeakyReLU(alpha=0.3)))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=4, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    training_set = train_datagen.flow_from_directory('training_set', target_size=(64,64),batch_size=32,\
                                                     class_mode='categorical')
    test_set = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32,\
                                                class_mode='categorical')

    classifier.fit_generator(training_set, steps_per_epoch=511, epochs=25, validation_data=test_set, validation_steps=149)

    for img in os.listdir("test"):
        test_image = image.load_img('test/' + img, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'NY'
        else:
            prediction = 'LA'
        print(img + " :" + prediction)

LAvsNY()