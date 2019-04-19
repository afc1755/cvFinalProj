README for building detector built using a faster R-CNN keras model.

### IMPORTANT ###
This code does NOT work with keras 2.2.4. REQUIRES keras 2.2.0

------------------------------------------------------------------------------------------------------------------------

### LABELING TRAINING DATA ###
You need to use LabelImg to label the training data.
https://github.com/tzutalin/labelImg

1. Place all training images in the 'images' directory
2. Open the directory with LabelImg
3. Box the building(s) you want to detect and give it a class name ('Willis Tower', 'World Trade Center', etc.)
4. Make sure to click save image and save the XML in 'images'

The XML file saves the name of the class(es) and the coordinates of the bounding box(es) you made.

------------------------------------------------------------------------------------------------------------------------

### CONVERTING TO LABELS ###
Verify that DIR_NAME is the name of the image directory and that OUT_NAME is the label file name you want to use
    xml_to_label.py lines: 6 - 7

To convert to labels all you need to do is run
#    python xml_to_label.py

------------------------------------------------------------------------------------------------------------------------

### TRAINING THE MODEL ###
Verify that the model name(s) is/are correct.
    train_frcnn.py lines: 32 - 35

Verify that the label name is correct. (Default is same as xml_to_label's default OUT_NAME)
    train_frcnn.py line: 36

To train the model you need to only perform the following command
#    python train_frcnn.py

------------------------------------------------------------------------------------------------------------------------

### TESTING THE MODEL ###
This will automatically use the same model that train_frcnn.py created. To change this modify
    predict.py line: 150

Tests will automatically store results in 'results_images' (overwriting duplicate file names). To change this modify
    predict.py line: 135

To run a single image test
#    python predict -p /path/to/image.jpg

For multiple files use
#    python predict -p /dir/with/images/