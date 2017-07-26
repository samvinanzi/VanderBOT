import qi

from bayesianNetwork import BeliefNetwork
from faceDetection import *
from faceRecognition import *

import numpy as np
import time

"""
This class models a physical Softbank robot (NAO or Pepper).
Programmed on NAOqi version 2.5.5.
"""


class Robot:
    def __init__(self, ip="nao.local", port=9559):
        self.IP = ip
        self.PORT = port
        self.session = qi.Session()
        try:
            self.session.connect("tcp://" + self.IP + ":" + str(self.PORT))
        except RuntimeError:
            print "Can't connect to Naoqi at ip \"" + self.IP + "\" on port " + str(self.PORT) + ".\n"
            quit(-1)
        self.video_service = None
        self.camera_name_id = None
        self.cam_h = None
        self.cam_w = None
        self.tts_service = None
        self.motion_service = None
        self.posture_service = None
        self.tracker_service = None
        self.led_service = None
        self.training_data = TrainingData()
        self.informants = 0
        self.beliefs = []
        self.landmark_service = None
        self.memory_service = None
        self.speech_service = None
        self.time = None
        self.load_time()
        self.animation_service = None
        self.audio_service = None
        self.service_setup()

    # Initializes the services
    def service_setup(self):
        self.video_service = self.session.service("ALVideoDevice")
        # Merely text-to-speech, with no motion
        # self.tts_proxy = self.session.service("ALTextToSpeech")
        self.tts_service = self.session.service("ALAnimatedSpeech")
        self.motion_service = self.session.service("ALMotion")
        self.posture_service = self.session.service("ALRobotPosture")
        self.tracker_service = self.session.service("ALTracker")    # For older versions of NAOqi use ALFaceTracker
        self.led_service = self.session.service("ALLeds")
        self.landmark_service = self.session.service("ALLandMarkDetection")
        self.memory_service = self.session.service("ALMemory")
        self.speech_service = self.session.service("ALSpeechRecognition")
        self.speech_service.setLanguage("English")
        self.animation_service = self.session.service("ALAnimationPlayer")
        self.audio_service = self.session.service("ALAudioPlayer")

    # Sets the color of the head leds
    def set_led_color(self, color, speed=0.5):
        color = str.upper(color)
        self.led_service.on("FaceLeds")
        if color == "YELLOW":
            hexcolor = 0x00ffff00
        elif color == "GREEN":
            hexcolor = 0x0000ff00
        elif color == "BLUE":
            hexcolor = 0x000000ff
        elif color == "RED":
            hexcolor = 0x00ff0000
        elif color == "WHITE":
            hexcolor = 0x00ffffff
        elif color == "OFF":
            self.led_service.off("FaceLeds")
            hexcolor = None
        else:
            print "color " + str(color) + " not recognized"
            hexcolor = None
        if color != "OFF":
            self.led_service.fadeRGB("FaceLeds", hexcolor, speed)

    # Text-to-Speech wrapper
    def say(self, words):
        self.tts_service.say(words)

    # Enables or disables face tracking
    def set_face_tracking(self, enabled, face_width=0.5):
        if enabled:
            self.motion_service.setStiffnesses("Head", 1.0)
            self.tracker_service.registerTarget("Face", face_width)
            self.tracker_service.track("Face")
            # self.set_led_color("green")
        else:
            self.tracker_service.stopTracker()
            self.tracker_service.unregisterAllTargets()
            self.set_led_color("white")

    # Subscribes the video service to retrieve data from cameras
    def video_service_subscribe(self):
        try:
            # "Trust_Video", CameraIndex=1, Resolution=1, ColorSpace=0, Fps=5
            # CameraIndex= 0(Top), 1(Bottom)
            # Resolution= 0(160*120), 1(320*240), VGA=2(640*480), 3(1280*960)
            # ColorSpace= AL::kYuvColorSpace (index=0, channels=1),
            #             AL::kYUV422ColorSpace (index=9,channels=3),
            #             AL::kRGBColorSpace RGB (index=11, channels=3),
            #             AL::kBGRColorSpace BGR (to use in OpenCV) (index=13, channels=3)
            # Fps= OV7670 VGA camera can only run at 30, 15, 10 and 5fps. The MT9M114 HD camera run from 1 to 30fps.
            resolution_type = 1
            fps = 15
            self.cam_w = 320
            self.cam_h = 240
            self.camera_name_id = self.video_service.subscribeCamera("Trust_Video", 0, resolution_type, 13, fps)
        except BaseException, err:
            print("[ERROR] video_proxy_subscribe: catching error " + str(err))
            quit(-1)

    # Unsubscribes to the video service
    def video_service_unsubscribe(self):
        self.video_service.unsubscribe(self.camera_name_id)

    # Captures a single image frame from the cameras
    def get_camera_image(self):
        image = np.zeros((self.cam_h, self.cam_w, 3), np.uint8)
        # Gets the raw image
        result = self.video_service.getImageRemote(self.camera_name_id)
        if result is None:
            print 'cannot capture.'
        elif result[6] is None:
            print 'no image data string.'
        else:
            # Translates the value to mat
            values = map(ord, list(str(result[6])))
            i = 0
            for y in range(0, self.cam_h):
                for x in range(0, self.cam_w):
                    image.itemset((y, x, 0), values[i + 0])
                    image.itemset((y, x, 1), values[i + 1])
                    image.itemset((y, x, 2), values[i + 2])
                    i += 3
        return image

    # If image contains a face, it retrieves the cropped region of interest
    def detect_face(self, image, grayscale=True):
        roi = facial_detection(image, grayscale=grayscale)
        return False if roi is None else True, roi

    # Captures a certain amount of face frames
    def collect_face_frames(self, number):
        face_frames = []
        self.video_service_subscribe()
        self.set_face_tracking(True)    # If should be on by default, but it re-enables is for debugging purposes
        found_faces = 0
        undetected_frames = 0
        while found_faces < number:
            image = self.get_camera_image()
            detected, roi = self.detect_face(image)
            if detected:
                # Blinking effect on face detection
                self.set_led_color("green", speed=0.2)
                self.set_led_color("white", speed=0.2)
                found_faces += 1
                face_frames.append(roi)
                undetected_frames = 0
            else:
                undetected_frames += 1
                if undetected_frames % 10 == 0:
                    self.say("I can't see you well. Can you please move closer?")
            cv2.putText(image, str("Detected: " + str(found_faces) + " / " + str(number)),
                        (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imshow("Robot Eyes", image)
            cv2.waitKey(1)
        #self.set_face_tracking(False)
        self.video_service_unsubscribe()
        return face_frames

    # Obtains training samples of one of the informers
    # Automatically updates the informant number
    # Saves the frames in the captures directory
    def acquire_examples(self, number_of_frames, informant_number):
        self.say("Hello informer number " + str(informant_number) + ". Please look at me")
        frames = self.collect_face_frames(number_of_frames)
        self.say("Thank you")
        count = 1
        for frame in frames:
            self.training_data.images.append(frame)
            self.training_data.labels.append(informant_number)
            cv2.imwrite("captures\\" + str(informant_number) + "-" + str(count) + ".jpg", frame)
            count += 1
        self.informants += 1
        self.look_forward()

    # Finalizes learning by training the model with all the data acquired
    def face_learning(self):
        recognition_train(self.training_data.prepare_for_training())

    # Recognizes a face
    # Collects an amount of frames, gets a prediction on each of them and returns the most predicted label
    def face_recognition(self, number_of_frames=5, announce=True):
        unknown = False
        self.say("Please look at me")
        # Collect face data
        frames = self.collect_face_frames(number_of_frames)
        # Counts the recognized labels
        predictions = [0 for i in range(self.informants+1)]  # An extra slot to consider the -1 (unknown informant) case
        for frame in frames:
            predictions[recognition_predict(frame)] += 1
        # Returns the maximum
        guess = predictions.index(max(predictions))
        # If the maximum value found is in the last position of the list, it's an unrecognized informant
        if guess == len(predictions)-1:
            unknown = True
            # Unknown informant! Adding it to the known ones and generating episodic memory
            self.manage_unknown_informant(frames)
            # This new informant has the biggest label yet
            guess = self.informants - 1
        if announce:
            if not unknown:
                self.say("Hello again, informer " + str(guess))
            else:
                self.say("I've never seen you before, I'll call you informer " + str(guess))
        return guess

    # Manages the unknown informant detection
    def manage_unknown_informant(self, frames):
        # Updates the model with the acquired frames and the right label
        new_data = TrainingData()
        new_data.images = frames
        new_data.labels = [self.informants for i in range(len(frames))]
        recognition_update(new_data.prepare_for_training())
        # Creates an episodic belief network
        name = "Informer" + str(self.informants) + "_episodic"
        episodic_network = BeliefNetwork.create_episodic(self.beliefs, self.get_and_inc_time(), name=name)
        self.beliefs.append(episodic_network)
        # Updates the total of known informants
        self.informants += 1    # This is done at the end because the label for the class is actually self.informants-1

    # Returns True if a landmark is detected
    def landmark_detect(self):
        # Check if any landmark data is available in memory
        mark_data = self.memory_service.getData("LandmarkDetected")
        if not mark_data:
            return False
        else:
            # Data is organized in this way:
            # [ [ TimeStampField ] [ Mark_info_0 , Mark_info_1, . . . , Mark_info_N-1 ] ]
            # Mark_info = [ ShapeInfo, MarkID ]
            # ShapeInfo = [ 1, alpha, beta, sizeX, sizeY, heading]
            return True

    # Looks for a landmark in the specified spot, A or B
    # Returns True if landmark is found, False otherwise
    def look_for_landmark(self, side):
        if side != 'A' and side != 'B':
            return None
        # It is important to disable face tracking while searching for the marker
        self.set_face_tracking(False)
        if side == 'A':
            self.look_A()
        elif side == 'B':
            self.look_B()
        period = 500
        try:
            self.landmark_service.subscribe("findSticker", period, 0.0)
        except BaseException, err:
            print("[ERROR] landmark_service_subscribe: catching error " + str(err))
            quit(-1)
        is_landmark_there = None
        # Tries for a few seconds to see the landmark
        for i in range(3):
            is_landmark_there = self.landmark_detect()
            if is_landmark_there:
                # Flash eyes
                self.set_led_color("green", speed=0.2)
                self.set_led_color("white", speed=0.2)
                self.audio_service.playSoundSetFile("Aldebaran", "sfx_validation_1")
                break
            else:
                time.sleep(0.5)
        self.landmark_service.unsubscribe("findSticker")
        self.look_forward()
        self.set_face_tracking(True)
        return is_landmark_there

    # Looks at box A:
    def look_A(self):
        names = list()
        times = list()
        keys = list()
        names.append("HeadPitch")
        times.append([1.76])
        keys.append([0.413643])
        names.append("HeadYaw")
        times.append([1.76])
        keys.append([-0.244346])
        try:
            self.motion_service.angleInterpolation(names, keys, times, True);
        except BaseException, err:
            print err

    # Looks at box B
    def look_B(self):
        names = list()
        times = list()
        keys = list()
        names.append("HeadPitch")
        times.append([1.76])
        keys.append([0.438078])
        names.append("HeadYaw")
        times.append([1.76])
        keys.append([0.403171])
        try:
            self.motion_service.angleInterpolation(names, keys, times, True);
        except BaseException, err:
            print err

    # Looks forward
    def look_forward(self):
        names = list()
        times = list()
        keys = list()
        names.append("HeadPitch")
        times.append([2.40000])
        keys.append([-0.18259])
        names.append("HeadYaw")
        times.append([2.40000])
        keys.append([-0.00618])
        try:
            self.motion_service.angleInterpolation(names, keys, times, True);
        except BaseException, err:
            print err

    # Listen to speech in order to recognize a word in a list of given words
    def listen_for_words(self, vocabulary):
        self.speech_service.setVocabulary(vocabulary, False)
        try:
            self.speech_service.subscribe("ListenWord")
        except BaseException, err:
            print("[ERROR] speech_proxy_subscribe: catching error " + str(err))
            quit(-1)
        while True:
            words = self.memory_service.getData("WordRecognized")
            if words[0] != '':
                break
        self.speech_service.unsubscribe("ListenWord")
        # The first element is always the most probable one
        word = words[0]
        return word

    # Vocal regonition of simple and short letters as "A" and "B" won't work
    # So the informers need to use another set of words to indicate boxes A and B
    # See the schema below for further indication on the experimental setup nomenclature
    """
          ROBOT
     A            B
    [ ]          [ ]
    left        right
        INFORMANT
    """
    def listen_for_side(self, vocabulary):
        # Checks the vocabulary validity: must be a list of two elements
        if not isinstance(vocabulary, list) or len(vocabulary) != 2:
            print "Invalid vocabulary: ", vocabulary
            quit(-1)
        else:
            word = self.listen_for_words(vocabulary)
            # From the user's vocabulary, gets the desired side in the robot's point of view
            return "A" if word == vocabulary[0] else "B"

    # Makes the robot stand up
    def standup(self):
        try:
            self.motion_service.setStiffnesses("Body", 1.0)
            self.posture_service.goToPosture("Stand", 1.0)
        except BaseException, err:
            print err

    # Makes the robot sit down
    def sitdown(self):
        # Inizializza i motori
        try:
            self.motion_service.setStiffnesses("Body", 1.0)
            self.posture_service.goToPosture("Sit", 1.0)
            self.motion_service.setStiffnesses("Body", 0.0)
        except BaseException, err:
            print err

    # Load time value from file
    def load_time(self):
        if os.path.isfile("current_time.csv"):
            with open("current_time.csv", 'r') as f:
                self.time = int(f.readline())
        else:
            self.time = 0

    # Increases and saves the current time value
    def get_and_inc_time(self):
        previous_time = self.time
        self.time += 1
        with open("current_time.csv", 'w') as f:
            f.write(str(self.time))
        return previous_time

    # Saves the beliefs
    def save_beliefs(self):
        if not os.path.exists(".\\datasets"):
            os.makedirs(".\\datasets")
        for belief in self.beliefs:
            belief.save()

    # Loads the beliefs
    def load_beliefs(self, path=".\\datasets\\"):
        # Resets previous beliefs
        self.beliefs = []
        i = 0
        while os.path.isfile(path + "Informer" + str(i) + ".csv"):
            self.beliefs.append(BeliefNetwork("Informer" + str(i), path + "Informer" + str(i) + ".csv"))
            i += 1

    # Reset time
    def reset_time(self):
        if os.path.isfile("current_time.csv"):
            with open("current_time.csv", 'w') as f:
                f.write("0")
        self.time = 0
