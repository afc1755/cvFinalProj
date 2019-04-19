import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
"""
Convert the XML training files to the following csv format:
    filename, xmin, ymin, xmax, ymax, class

The min/max variables are the coordinates of the bounding box you made in LabelImg
"""
DIR_NAME = "train_images"
OUT_NAME = "building_labels.txt"

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            # For some reason having the full path [path + root.find(..)] would break it when it started training
            # Below would give us something like 'images\__.jpg' which works. Also back slash cause windows is gr8
            value = (DIR_NAME + "/" + root.find('filename').text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     member[0].text
                     )
            xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']  # Doesn't actually do anything
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    print("Reading from '" + DIR_NAME + "'")
    image_path = os.path.join(os.getcwd(), DIR_NAME)
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(OUT_NAME, index=None, header=False)  # Write it without a header or line numbers
    print("Done converting. " + OUT_NAME + " created.")


main()
