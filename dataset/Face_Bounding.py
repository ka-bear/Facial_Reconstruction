import os
from os import listdir
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import pandas as pd
import numpy as np


if __name__ == '__main__':

    # get the path/directory
    folder_dir = "C:\\Users\\admin\\Downloads\\archive\\faces_0\\"
    df = pd.DataFrame({"bx": [],
                       "by": [],
                       "tx": [],
                       "ty": []},dtype=np.float64)

    for i in range(1, 25):
        s = ""
        if (i < 10):
            s = "0" + str(i)
        else:
            s = str(i)
        folder_dirs = folder_dir + s + "\\"

        for images in os.listdir(folder_dirs):

            # check if the image ends with png
            #print(images)
            if images.endswith(".png"):
                im = cv2.imread(folder_dirs + images)

                faces, confidences = cv.detect_face(im)
                # loop through detected faces and add bounding box
                for face in faces:
                    (startX, startY) = face[0], face[1]
                    (endX, endY) = face[2], face[3]
                    df.loc[len(df)] = [startX, startY, endX, endY]
        df.to_csv("test"+str(i)+".csv")
        print("hi")


