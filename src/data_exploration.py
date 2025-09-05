import os 
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np 

print("Ready to explore emotion data!")

def explore_dataset():
    # define emotion labels
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # check what folders we actually have
    data_path = "../data/raw/fer2013/test"

    print("Dataset overview")
    print("-" * 30)

    # count images in each folder
    for emotion in emotions:
        folder_path = os.path.join(data_path, emotion)
        if os.path.exists(folder_path):
            count = len(os.listdir(folder_path))
            print(f"{emotion.capitalize()}: {count} images")

    # display a few sample images from each category
    plt.figure(figsize=(10, 15))
    for i, emotion in enumerate(emotions[:6]): # show first 6 emotions
        folder_path = os.path.join(data_path, emotion)
        if os.path.exists(folder_path):
            # get first image from folder
            image_files = os.listdir(folder_path) # get list of files
            if image_files: # check if list is not empty
                img_path = os.path.join(folder_path, image_files[0]) # get first image
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # read as grayscale

                plt.subplot(2, 3, i+1) # 2 rows, 3 cols 
                plt.imshow(img, cmap='gray') # show image in grayscale
                plt.title(emotion.capitalize())
                plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    explore_dataset()
                
            
