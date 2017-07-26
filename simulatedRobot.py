from faceRecognition import *
from robot import Robot
import numpy as np

""" 
This simulated robot is to be used in virtual experiments. It inheritates the RobotCV.Robot methods, but
disables those which are not executable and that depend on a phisical robot.
Camera tests will be performed by the computer's webcam.
Movement and postural methods will only produce a textual description of the action.
"""


class SimulatedRobot(Robot):
    def __init__(self):
        # This class doesn't call it's superclass initializer because it can't connect a session and retrieve services
        self.IP = 'pepper.local'
        self.PORT = 9559
        self.training_data = TrainingData()
        self.informants = 0
        self.beliefs = []
        self.time = None
        self.load_time()
        # Adds the landmark position to the simulation
        self.landmark_position = 'A'

    # Disables functions not runnable in simulation

    def service_setup(self):
        print "proxy_setup: method not available for the simulated robot."

    def set_face_tracking(self, enabled, face_width=0.5):
        print "set_face_tracking: method not available for the simulated robot."

    def set_led_color(self, color, speed=0.5):
        print "set_led_color: method not available for the simulated robot."

    def landmark_detect(self):
        print "landmark_detect: method not available for the simulated robot."

    # Descriptive postural and movement methods

    def look_A(self):
        print "{Robot is looking at box A}"

    def look_B(self):
        print "{Robot is looking at box B}"

    def look_forward(self):
        print "{Robot is looking forward}"

    def standup(self):
        print "{Robot is standing up}"

    def sitdown(self):
        print "{Robot is sitting down}"

    # Working, redefined methods

    def get_camera_image(self):
        image = np.zeros((480, 640, 3), np.uint8)
        cam = cv2.VideoCapture(0)  # 0 -> index of camera
        success, img = cam.read()
        if success:  # frame captured without any errors
            return img
        else:
            return None

    # Includes a visual debug screen
    def collect_face_frames(self, number):
        face_frames = []
        found_faces = 0
        while found_faces < number:
            image = self.get_camera_image()
            detected, roi = self.detect_face(image, grayscale=True)
            if detected:
                found_faces += 1
                face_frames.append(roi)
            cv2.putText(image, str("Detected: " + str(found_faces) + " / " + str(number)),
                        (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow("Robot Eyes", image)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        return face_frames

    # Input: A or B
    def look_for_landmark(self, side):
        if side != "A" and side != "B":
            print "[ERROR] look_for_landmark: invalid input " + str(side)
            quit()
        if side == "A":
            self.look_A()
        else:
            self.look_B()
        result = True if side == self.landmark_position else False
        print "{Sticker " + ("found" if result else "not found") + "}"
        self.look_forward()
        return result

    def set_landmark_position(self, position):
        if position != 'left' and position != 'right':
            print "[ERROR] set_landmark_position: invalid input " + str(position)
            quit()
        else:
            if position == "left":
                self.landmark_position = 'A'
            else:
                self.landmark_position = 'B'

    def say(self, words):
        print "[ROBOT SAYS] " + words

    def listen_for_words(self, vocabulary):
        while True:
            word = input('Input: ')
            word = word.lower()
            if word in vocabulary:
                break
        return word.lower()

    def listen_for_side(self, vocabulary):
        side = input('Side: ')
        side = side.lower()
        if side == vocabulary[0]:
            return "A"
        elif side == vocabulary[1]:
            return "B"
        else:
            print "[ERROR] listen_for_side: invalid input: " + str(side)
            quit()
