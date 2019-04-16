#created by: Andrew Chabot
#RIT Computer Vision Final Project
import os
import numpy as np
from keras.engine.saving import model_from_json
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

def train():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(100, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(50, activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(units=4, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, horizontal_flip=True)
    training_set = train_datagen.flow_from_directory('training_set', target_size=(64,64),batch_size=32,\
                                                     class_mode='categorical')
    test_set = test_datagen.flow_from_directory('test_set', target_size=(64, 64), batch_size=32,\
                                                class_mode='categorical')

    classifier.fit_generator(training_set, steps_per_epoch=512, epochs=2, validation_data=test_set, validation_steps=149)

    #save our classifier after training
    model_json = classifier.to_json()
    with open("model3.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("model3.h5")
    print("Saved model to disk")

def largeTestModel():
    json_file = open('model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model2.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("\n\nBOSTON tests")
    print("---------------------------")
    right = 0.0
    total = 0.0
    for img in os.listdir("test_set/BOSTON"):
        if img[0] == ".":
            continue
        total+=1
        test_image = image.load_img('test_set/BOSTON/' + img, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)
        city = np.argmax(result)
        confidence = np.max(result)
        cityName = ""
        if city == 0:
            cityName = "BOSTON"
            right+=1
        elif city == 1:
            cityName = "CHICAGO"
        elif city == 2:
            cityName = "LOS ANGELES"
        elif city == 3:
            cityName = "NEW YORK CITY"
        print(img + " : " + cityName + ", confidence: " + str(confidence))
    print("overall correct: " + str(right/total))

    print("\n\nCHICAGO tests")
    print("---------------------------")
    right = 0.0
    total = 0.0
    for img in os.listdir("test/CHICAGO"):
        total += 1
        if img[0] == ".":
            continue
        test_image = image.load_img('test/CHICAGO/' + img, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)
        city = np.argmax(result)
        confidence = np.max(result)
        cityName = ""
        if city == 0:
            cityName = "BOSTON"
        elif city == 1:
            cityName = "CHICAGO"
            right += 1
        elif city == 2:
            cityName = "LOS ANGELES"
        elif city == 3:
            cityName = "NEW YORK CITY"
        print(img + " : " + cityName + ", confidence: " + str(confidence))
    print("overall correct: " + str(right / total))


    print("\n\nLOS ANGELES tests")
    print("---------------------------")
    right = 0.0
    total = 0.0
    for img in os.listdir("test/LA"):
        total += 1
        if img[0] == ".":
            continue
        test_image = image.load_img('test/LA/' + img, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)
        city = np.argmax(result)
        confidence = np.max(result)
        cityName = ""
        if city == 0:
            cityName = "BOSTON"
        elif city == 1:
            cityName = "CHICAGO"
        elif city == 2:
            cityName = "LOS ANGELES"
            right += 1
        elif city == 3:
            cityName = "NEW YORK CITY"
        print(img + " : " + cityName + ", confidence: " + str(confidence))
    print("overall correct: " + str(right / total))

    print("\n\nNEW YORK tests")
    print("---------------------------")
    right = 0.0
    total = 0.0
    for img in os.listdir("test/NY"):
        total += 1
        if img[0] == ".":
            continue
        test_image = image.load_img('test/NY/' + img, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = loaded_model.predict(test_image)
        city = np.argmax(result)
        confidence = np.max(result)
        cityName = ""
        if city == 0:
            cityName = "BOSTON"
        elif city == 1:
            cityName = "CHICAGO"
        elif city == 2:
            cityName = "LOS ANGELES"
        elif city == 3:
            cityName = "NEW YORK CITY"
            right += 1
        print(img + " : " + cityName + ", confidence: " + str(confidence))
    print("overall correct: " + str(right / total))

#test pictures for model2:
#Boston: BOSTON/1.jpg
#Chicago: CHICAGO/4.jpg
#LA: LA/1.jpg
#NY: NY/3.jpeg
def testModel():
    json_file = open('model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model2.h5")
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
            cityName = "BOSTON"
        elif city == 1:
            cityName = "CHICAGO"
        elif city == 2:
            cityName = "LOS ANGELES"
        elif city == 3:
            cityName = "NEW YORK CITY"
        print(imageName + " : " + cityName + ", confidence: " + str(confidence))

#train()
largeTestModel()
#testModel()