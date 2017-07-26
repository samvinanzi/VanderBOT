import os.path

import cv2


# Creates the working directory or empties it
def prepare_workspace(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        file_list = [f for f in os.listdir(dir_name)]
        for f in file_list:
            os.remove(os.path.join(dir_name, f))


def facial_detection(img, scale_factor=1.4, min_neighbours=5, single=True, debug=False, grayscale=True):
    """ Performs facial detection within an image
    :param img: image data matrix
    :param scale_factor: how much the image size is reduced at each image scale
    :param min_neighbours: how many neighbors each candidate rectangle should have to retain it
    :param single: search for single (True) or multiple (False) faces
    :param debug: if True, enables verbose output
    :param grayscale: if True, converts the image to grayscale
    :return: greyscale region(s) of interest, scaled to 64x64 pixels
    """

    if img is None:
        return

    # Creates a Cascade Classifier and loads the default training data
    haar_xml = ".\\classifiers\\haarcascade_frontalface_default.xml"
    if not os.path.isfile(haar_xml):
        print "[ERROR] Unable to load the HaarCascade classifier. Verify file: \"" + os.path.relpath(haar_xml) + "\""
        quit(-1)
    face_cascade = cv2.CascadeClassifier(haar_xml)

    if grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Histogram equalization
        gray = cv2.equalizeHist(gray)
    else:
        # I keep the variable name because this is a late update
        gray = img
    if debug:
        # Shows the color (and eventually the gray) image
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if grayscale:
            cv2.imshow("gray", gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    faces = face_cascade.detectMultiScale3(gray, scaleFactor=scale_factor, minNeighbors=min_neighbours,
                                           outputRejectLevels=True)
    rects = faces[0]
    #neighbours = faces[1]
    weights = faces[2]
    roi_list = []
    roi_areas = []
    c = 0
    for (x, y, w, h) in rects:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, str(weights[c][0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if debug:
            cv2.imshow("rectangled", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        roi = gray[y:y + h, x:x + w]
        roi_area = reduce(lambda x, y: x * y, roi.shape)
        roi_resized = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
        roi_list.append(roi_resized)
        roi_areas.append(roi_area)
        c += 1

    if len(roi_list) == 0:
        return None
    elif single:
        # Gets the biggest rectangle (nearest to the robot)
        max_value = max(roi_areas)
        max_index = roi_areas.index(max_value)
        return roi_list[max_index]
    else:
        return roi_list
