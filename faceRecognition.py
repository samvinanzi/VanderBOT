import os

import cv2

from trainingData import TrainingData

MODEL_FILE = ".\\classifiers\\robotvision.yml"

"""
Methods included in this file manage the face recognition algorithms.
Selectable algorithms:  0: EigenFaces
                        1: FisherFaces
                        2: Local Binary Patterns Histograms (LBPH)
Recommended (and default) is LBPH, as it is the only one which supports updating. Other methods will need to run
a new training with all the old samples plus the new ones.
"""

ALGORITHM_NUMBER = 2


# Selects a model
def model_initialize(model_number, withTreshold=False, threshold=100.0):
    if model_number == 0:
        if withTreshold:
            return cv2.face.createEigenFaceRecognizer(threshold=threshold)
        else:
            return cv2.face.createEigenFaceRecognizer()
    elif model_number == 1:
        if withTreshold:
            return cv2.face.createFisherFaceRecognizer(threshold=threshold)
        else:
            return cv2.face.createFisherFaceRecognizer()
    elif model_number == 2:
        if withTreshold:
            return cv2.face.createLBPHFaceRecognizer(threshold=threshold)
        else:
            return cv2.face.createLBPHFaceRecognizer()
    else:
        print "[ERROR] Invalid algorithm selected: " + str(ALGORITHM_NUMBER)
        quit()


# Acquires training data from a directory containing images
def data_from_file(directory):
    data = TrainingData()
    informers = 2
    for i in range(informers):
        dir = os.path.join(directory, str(i))
        file_list = [f for f in os.listdir(dir)]
        for file in file_list:
            img = cv2.imread(os.path.join(dir, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_equ = cv2.equalizeHist(gray)
            data.images.append(gray_equ)
            data.labels.append(i)
    return data


# Trains the face recognition module using the selected model. Saves is for future use.
def recognition_train(data):
    if isinstance(data, TrainingData):
        model = model_initialize(ALGORITHM_NUMBER, withTreshold=False)
        model.train(data.images, data.labels)
        # Cleares up previous models
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        model.save(MODEL_FILE)
    else:
        print "[ERROR] recognition_train: input is not a TrainingData instance."
        quit(-1)


# Loads the selected model and does a prediction.
# Threshold regulates the unknown informant detection
# I assume frame is already been cropped, resized and converted to greyscale
def recognition_predict(frame):
    model = model_initialize(ALGORITHM_NUMBER, withTreshold=True)
    model.load(MODEL_FILE)
    [predicted_label, predicted_confidence] = model.predict(frame)
    # Returns class name
    return predicted_label


# Updates the model with new training data
def recognition_update(new_data):
    if isinstance(new_data, TrainingData):
        model = model_initialize(ALGORITHM_NUMBER, withTreshold=False)
        model.load(MODEL_FILE)
        model.update(new_data.images, new_data.labels)
        # Cleares up previous models
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        model.save(MODEL_FILE)
    else:
        print "[ERROR] recognition_update: input is not a TrainingData instance."
        quit(-1)
