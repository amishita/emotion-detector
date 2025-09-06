import cv2
import os
import numpy as np

data_path = "../data/raw/fer2013/test"
processed_data_path = "../data/processed/test.npz"
emotions = os.listdir(data_path) # list of folders in the data path

X, Y = [], [] # X -> images, Y -> labels (emotions)

# Loop through each emotion folder
for idx, emotion in enumerate(emotions): # idx -> 
    folder = os.path.join(data_path, emotion) # path to each emotion folder

    # loop through each image in the folder
    for image in os.listdir(folder): 
        image_path = os.path.join(folder, image) # path to each image

        # load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

        # resize to 48x48
        img = cv2.resize(img, (48, 48))

        # normalize to (0, 1)
        img = img.astype('float32') / 255.0 

        X.append(img) # append image to X
        Y.append(idx) # append label to Y 

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# save to .npz file
# np.savez -> saves multiple arrays into a single file
# save X as X, Y as Y, and emotions list as emotions in processed_data_path
np.savez (processed_data_path, X=X, Y=Y, emotions=emotions)

print(f"Data saved to {processed_data_path}")
print("X shape:", X.shape) # normalized images -> (num_samples, 48, 48)
print("Y shape:", Y.shape) # integer labels -> 0-6 for 7 emotions
