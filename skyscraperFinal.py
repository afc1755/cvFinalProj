#created by: Andrew Chabot
#RIT Computer Vision Final Project Part 2?
import os
import numpy as np
from keras.engine.saving import model_from_json
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

def train():
    classifier = Sequential()
    classifier.add(Conv2D(64, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(64, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(BatchNormalization())
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(BatchNormalization())
    classifier.add(Dropout(0.2))
    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(4, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('sTrain', target_size=(128, 128), batch_size=32,
                                                     class_mode='categorical', shuffle=True)
    test_set = test_datagen.flow_from_directory('sTest', target_size=(128, 128), batch_size=32,
                                                class_mode='categorical', shuffle=True)

    classifier.fit_generator(training_set, steps_per_epoch=950, epochs=10, validation_data=test_set, validation_steps=78)

    #save our classifier after training
    model_json = classifier.to_json()
    with open("sModel1.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("sModel1.h5")
    print("Saved model to disk")

def testModel():
    json_file = open('sModel1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("sModel1.h5")
    print("Loaded model from disk")
    while True:
        imageName = input("Please enter the image you would like to classify:")
        test_image = image.load_img('test/' + imageName, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)
        city = np.argmax(result)
        confidence = np.max(result)
        cityName = ""
        if city == 0:
            cityName = "EIFFEL"
        elif city == 1:
            cityName = "EMPIRE"
        elif city == 2:
            cityName = "GATEWAY"
        elif city == 3:
            cityName = "WASHINGTON"
        print(imageName + " : " + cityName + ", confidence: " + str(confidence))

train()
#largeTestModel()
#testModel()