import numpy as np

"""
Support class to manage the training dataset for the face recognition algorithms.
"""


class TrainingData:
    def __init__(self):
        self.images = []
        self.labels = []

    # Converts images from OpenCV image to nparray and generates a new TrainingData object to be passed
    # to the training function
    def prepare_for_training(self):
        nparrays = []
        for image in self.images:
            nparrays.append(np.asarray(image, dtype=np.uint8))
        new_item = TrainingData()
        new_item.images = nparrays
        new_item.labels = np.asarray(self.labels, dtype=np.int32)
        return new_item
