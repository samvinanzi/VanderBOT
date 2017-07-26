import time

import faceDetection
from bayesianNetwork import BeliefNetwork
from episode import Episode
from robot import Robot
from simulatedRobot import SimulatedRobot

"""
Vanderbilt experiment for the evaluation of trust and Theory of Mind matureness, robotic version.
This version only supports LBPH algorithm for face recognition. If you want to modify it to use EigenFaces or 
FisherFaces, you must provide methods to re-train the model with all the face samples ever collected.
"""


class Vanderbilt:
    def __init__(self, robot_ip="pepper.local", demo_number=6, mature=True, simulation=False, withUpdate=False):
        if not simulation:
            self.robot = Robot(robot_ip)
            self.init_robot()
        else:
            self.robot = SimulatedRobot()
        self.demo_number = demo_number
        if demo_number % 2 != 0:
            # Odd trial number, giving a warning
            print "[WARNING] An odd number of demonstrations will make Vanderbilt's experiment non-standard!"
        self.informant_vocabulary = ["left", "right"]
        self.face_frames_captured = 10
        self.mature = mature
        self.simulation = simulation
        self.withUpdate = withUpdate

    # Initializes the robot to the standard, ready to start configuration
    def init_robot(self):
        self.robot.set_led_color("white")
        self.robot.set_face_tracking(True)
        self.robot.standup()
        # Time count starting from 0
        self.robot.reset_time()

    # Helps the experimenters to create the experimental environment setup
    def help_setup(self):
        if self.simulation:
            print "Help not available in simulation"
            return
        self.robot.say("I'm going to help you create the experimental setup.")
        self.robot.say("I'm going to stand up.")
        time.sleep(1)
        self.robot.standup()
        self.robot.say("Now place me in position.")
        time.sleep(5)
        self.robot.say("Now place the mat in front of me and put the sticker on the left.")
        time.sleep(5)
        found = False
        while not found:
            found = self.robot.look_for_landmark('A')
            if not found:
                self.robot.say("I can't see the sticker. Please reposition the mat")
                time.sleep(2)
            else:
                found_alt = self.robot.look_for_landmark('B')
                if found_alt:
                    self.robot.say("I can see the sticker in position A when looking at B. Please reposition the mat.")
                    time.sleep(2)
                    found = False
        self.robot.say("Ok. Move the sticker to the right")
        time.sleep(2)
        found = False
        while not found:
            found = self.robot.look_for_landmark('B')
            if not found:
                self.robot.say("I can't see the sticker. Please replace the mat")
                time.sleep(2)
            else:
                found_alt = self.robot.look_for_landmark('B')
                if found_alt:
                    self.robot.say("I can see the sticker in position B when looking at A. Please reposition the mat.")
                    time.sleep(2)
                    found = False
        self.robot.say("Perfect! Everything is ready.")

    # A whole Vanderbilt experiment execution
    def start(self):
        if self.simulation:
            print "[INFO] Simulation initialized. Please give your inputs surrounded by quotation marks."
        faceDetection.prepare_workspace("captures")
        self.robot.look_forward()
        if not self.simulation:
            self.robot.animation_service.runTag("hello")
        self.robot.say("Hello, nice to meet you.")
        self.robot.say("My name is Pepper and I'm glad to welcome you to the Vanderbot experiment for "
                       "Trust and Theory of Mind in humanoid robots.")
        time.sleep(1)
        self.robot.say("The experiment begins now")
        time.sleep(2)
        self.robot.say("Familiarization Phase")
        self.familiarization()
        time.sleep(2)
        self.robot.say("Decision Making Phase")
        repeat = "yes"
        while repeat == "yes":
            self.decision_making(withUpdate=self.withUpdate)
            self.robot.say("Do you want to repeat? Yes or no.")
            repeat = self.robot.listen_for_words(["yes", "no"])
        self.robot.say("Ok then, let's continue with the experiment.")
        time.sleep(2)
        self.robot.say("Belief Estimation Phase")
        repeat = "yes"
        while repeat == "yes":
            self.belief_estimation()
            self.robot.say("Do you want to repeat? Yes or no.")
            repeat = self.robot.listen_for_words(["yes", "no"])
        self.robot.say("The experiment has ended. Thank you for your participation.")
        if not self.simulation:
            self.robot.animation_service.runTag("hello")
        self.end()

    # Familiarization Phase
    def familiarization(self):
        string_to_int = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        # How many informants will be interacting?
        confirm_message = "no"
        word = None
        while confirm_message != "yes":
            self.robot.say("How many informants will I be interacting with? No less than one and no more than ten")
            word = self.robot.listen_for_words(["one", "two", "three", "four", "five", "six", "seven",
                                                "eight", "nine", "ten"])
            self.robot.say(word + (" partecipant" if string_to_int[word] == 1 else " partecipants") + ", yes or no?")
            confirm_message = self.robot.listen_for_words(["yes", "no"])
            if confirm_message == "no":
                self.robot.say("Sorry")
                if not self.simulation:
                    self.robot.animation_service.runTag("negative")
            elif not self.simulation:
                self.robot.animation_service.runTag("affirmative")
        number_of_informants = string_to_int[word]
        # Face detection and dataset collection
        for i in range(number_of_informants):
            self.demonstration(i)
            if i < number_of_informants - 1:
                self.robot.say("Please leave your place for informer number " + str(i+1))
                time.sleep(10)
        # Face learning
        self.robot.face_learning()

    # Demonstration: the robot familiarizes with the informer's face and habits
    def demonstration(self, informant_number):
        # Gets face samples for future recognition
        if not self.simulation:
            self.robot.animation_service.runTag("show")
        self.robot.acquire_examples(self.face_frames_captured, informant_number)
        self.robot.say("We are starting a brief demonstration. I am going to ask you to tell me where "
                       "the sticker is. We are going to undertake " + str(self.demo_number) +
                       (" trial" if self.demo_number == 1 else " trials"))
        if not self.simulation:
            self.robot.animation_service.runTag("explain")
        time.sleep(2)
        demo_result = []
        for i in range(self.demo_number):
            # demo_sample = [Xr, Yr, Xi, Yi]
            demo_sample = [0, 0, 0, 0]
            if self.simulation:
                self.relocate_sticker()
            if i == self.demo_number - 1:
                self.robot.say("Now for the last time.")
            self.robot.say("Can you suggest me the location of the sticker? Left or right?")
            hint = self.robot.listen_for_side(self.informant_vocabulary)
            found = self.robot.look_for_landmark(hint)
            if self.mature:
                # Mature ToM
                if (hint == 'A' and found) or (hint == 'B' and not found):
                    demo_sample[0] = 1
                    demo_sample[1] = 1
                    demo_sample[2] = 1
                if hint == 'A':
                    demo_sample[3] = 1
            else:
                # Immature ToM
                if hint == 'A':
                    demo_sample = [1, 1, 1, 1]
                else:
                    demo_sample = [0, 0, 0, 0]
            demo_sample_episode = Episode(demo_sample, self.robot.get_and_inc_time())
            demo_result.append(demo_sample_episode)
            # Give experimenters the time to switch the sticker location
            if not self.simulation and i < self.demo_number - 1:
                time.sleep(5)
        self.robot.say("Excellent, now I know you a little more. Thank you")
        # Creates the belief network for this informer
        self.robot.beliefs.append(BeliefNetwork("Informer" + str(informant_number), demo_result))

    # Decision Making Phase
    def decision_making(self, withUpdate=True):
        # The robot first recognizes the informer
        informer = self.robot.face_recognition()
        if self.simulation:
            self.relocate_sticker()
        self.robot.say("Can you suggest me the location of the sticker? Left or right?")
        hint = self.robot.listen_for_side(self.informant_vocabulary)
        # Decision making based on the belief network for that particular informant
        choice = self.robot.beliefs[informer].decision_making(hint)
        if self.simulation:
            print "Robot decides to look at position: " + str(choice)
        else:
            self.robot.say("I'm thinking at where to look based on your suggestion...")
            self.robot.animation_service.runTag("think")
            self.robot.set_led_color("white")
        found = self.robot.look_for_landmark(choice)
        if self.mature:
            # Mature ToM
            if hint == choice and found:
                self.robot.say("I trusted you and your suggestion was correct. Thank you!")
                if not self.simulation:
                    self.robot.animation_service.runTag("friendly")
            elif hint == choice and not found:
                self.robot.say("I trusted you, but you tricked me.")
                if not self.simulation:
                    self.robot.animation_service.runTag("frustrated")
            elif hint != choice and found:
                self.robot.say("I was right not to trust you.")
                if not self.simulation:
                    self.robot.animation_service.runTag("indicate")
            elif hint != choice and not found:
                self.robot.say("I didn't trust you, but I was wrong. Sorry.")
                if not self.simulation:
                    self.robot.animation_service.runTag("ashamed")
        else:
            # Immature ToM
            if found:
                self.robot.say("Oh, here it is!")
            else:
                self.robot.say("The sticker is not here.")
        # If required, update the belief network to consider this last episode
        if withUpdate:
            new_data = []
            if self.mature:
                # Mature ToM
                if choice == "A" and found:
                    new_data = [1, 1, 1, 1]
                elif choice == "B" and found:
                    new_data = [0, 0, 0, 0]
                elif choice == "A" and not found:
                    new_data = [0, 0, 0, 1]
                elif choice == "B" and not found:
                    new_data = [1, 1, 1, 0]
            else:
                # Immature ToM
                if choice == "A":
                    new_data = [1, 1, 1, 1]
                else:
                    new_data = [0, 0, 0, 0]
            new_episode = Episode(new_data, self.robot.get_and_inc_time())
            self.robot.beliefs[informer].update_belief(new_episode)
            # Add the symmetric espisode too (with the same time value)
            self.robot.beliefs[informer].update_belief(new_episode.generate_symmetric())
        # Finally, resets the eye color just in case an animation modified it
        if not self.simulation:
            self.robot.set_led_color("white")

    # Belief Estimation Phase
    def belief_estimation(self):
        if self.simulation:
            self.relocate_sticker()
        # Recognizes the informer
        informer = self.robot.face_recognition()
        # Finds the sticker location
        side = None
        while side is None:
            if self.robot.look_for_landmark('A'):
                side = 'A'
            elif self.robot.look_for_landmark('B'):
                side = 'B'
            else:
                self.robot.say("Where is the sticker? I can't find it. Please put it in place.")
                # Give the experimenters time to replace the sticker
                time.sleep(5)
        [informant_belief, informant_action] = self.robot.beliefs[informer].belief_estimation(side)
        if not self.simulation:
            self.robot.say("Let me think...")
            self.robot.animation_service.runTag("think")
            self.robot.set_led_color("white")
        self.robot.say("I know the sticker is on the " + self.translate_side(side) + ".")
        self.robot.say("I believe you think the sticker is on the " + self.translate_side(informant_belief))
        self.robot.say("I also believe you would point " + self.translate_side(informant_action) + " to me.")

    # Translates the robot's label for the sides in one suitable for the informer
    def translate_side(self, side):
        return self.informant_vocabulary[0] if side == "A" else self.informant_vocabulary[1]

    # Closing processes
    def end(self):
        self.robot.save_beliefs()
        if not self.simulation:
            self.robot.set_face_tracking(False)
        self.robot.standup()

    # Relocates the sticker in the simulated environment
    def relocate_sticker(self):
        while True:
            position = input("Where do you want to relocate the sticker (left or right)? ")
            position = position.lower()
            if position == "left" or position == "right":
                self.robot.set_landmark_position(position)
                break
