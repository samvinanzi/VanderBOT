import copy
import os
from math import log
from random import randint, shuffle

from bayesian.bbn import *
from bayesian.utils import make_key

from datasetParser import DatasetParser
from episode import Episode

"""
This class models the Developmental Bayesian Model of Trust in Artificial Cognitive Systems (Patacchiola, 2016).
Methods include the generation of the Episodic Memory by importance filtering and systematic resampling.
"""


class BeliefNetwork:
    def __init__(self, name, dataset):
        self.name = name
        self.dataset = DatasetParser(dataset)
        self.parameters = self.dataset.estimate_bn_parameters()
        self.bn = None
        # truth_a :
        # lie_a : sticker in A, informar said B
        # ecc...
        self.pdf = {
            'truth_a': 1.0,  # sticker in A, informer said A
            'truth_b': 1.0,  # sticker in B, informer said B
            'lie_a': 1.0,    # sticker in A, informer said B
            'lie_b': 1.0     # sticker in B, informer said A
        }
        self.entropy = None
        # Post-initialization processes
        self.build()
        self.calculate_pdf()
        self.entropy = self.get_entropy()

    # Xi
    def f_informant_belief(self, informant_belief):
        if informant_belief == 'A':
            return self.parameters["Xi"][1]
        return self.parameters["Xi"][0]

    # Xr
    def f_robot_belief(self, robot_belief):
        if robot_belief == 'A':
            return self.parameters["Xr"][1]
        return self.parameters["Xr"][0]

    # Yi
    def f_informant_action(self, informant_belief, informant_action):
        if informant_belief == 'A' and informant_action == 'A':
            return self.parameters["Yi"][0][0]
        if informant_belief == 'A' and informant_action == 'B':
            return self.parameters["Yi"][0][1]
        if informant_belief == 'B' and informant_action == 'A':
            return self.parameters["Yi"][1][0]
        if informant_belief == 'B' and informant_action == 'B':
            return self.parameters["Yi"][1][1]

    # Yr
    def f_robot_action(self, informant_action, robot_belief, robot_action):
        table = dict()
        table['aaa'] = self.parameters["Yr"][0][0]
        table['aab'] = self.parameters["Yr"][0][1]
        table['aba'] = self.parameters["Yr"][1][0]
        table['abb'] = self.parameters["Yr"][1][1]
        table['baa'] = self.parameters["Yr"][2][0]
        table['bab'] = self.parameters["Yr"][2][1]
        table['bba'] = self.parameters["Yr"][3][0]
        table['bbb'] = self.parameters["Yr"][3][1]
        return table[make_key(informant_action, robot_belief, robot_action)]

    # Constructs the bayesian belief network
    def build(self):
        self.bn = build_bbn(
            self.f_informant_belief,
            self.f_robot_belief,
            self.f_informant_action,
            self.f_robot_action,
            domains=dict(
                informant_belief=['A', 'B'],
                robot_belief=['A', 'B'],
                informant_action=['A', 'B'],
                robot_action=['A', 'B']),
            name=self.name
        )

    # Test query
    def test_query(self, prettyTable=False):
        if prettyTable:
            self.bn.q()
        return self.bn.query()

    # Decision Making
    # Sets informant_action as evidence and infers robot_action
    def decision_making(self, informant_action):
        if informant_action != 'A' and informant_action != 'B':
            return None
        else:
            outputs = self.bn.query(informant_action=informant_action)
            if outputs['robot_action', 'A'] > outputs['robot_action', 'B']:
                return 'A'
            else:
                return 'B'

    # Belief Estimation
    # Sets robot_belief and robot_action as evidence and infers informant_belief
    def belief_estimation(self, robot_knowledge):
        if robot_knowledge != 'A' and robot_knowledge != 'B':
            return None
        else:
            outputs = self.bn.query(robot_belief=robot_knowledge, robot_action=robot_knowledge)
            # Duple: [informant_belief , informant_action]
            predicted_behaviour = []
            if outputs['informant_belief', 'A'] > outputs['informant_belief', 'B']:
                predicted_behaviour.append('A')
            else:
                predicted_behaviour.append('B')
            if outputs['informant_action', 'A'] > outputs['informant_action', 'B']:
                predicted_behaviour.append('A')
            else:
                predicted_behaviour.append('B')
            return predicted_behaviour

    # Updates in real-time the belief
    def update_belief(self, new_data):
        if isinstance(new_data, Episode):
            previous_dataset = self.get_episode_dataset()
            previous_dataset.append(new_data)   # "previous_dataset" is now updated with new data
            self.dataset = DatasetParser(previous_dataset)
            self.parameters = self.dataset.estimate_bn_parameters()
            self.build()
            self.calculate_pdf()
        else:
            print "[ERROR] update_belief: new data is not an Episode instance."
            quit(-1)

    # Prints the network parameters
    def print_parameters(self):
        print self.name + "\n" + str(self.parameters)

    # Gets the raw input dataset of the BN
    def get_episode_dataset(self):
        return self.dataset.episode_dataset

    # Static Method
    # Creates a new BN used for episodic memory. Uses all the previous information collected
    @staticmethod
    def create_full_episodic_bn(bn_list, time):
        dataset = []
        for bn in bn_list:
            episode_list = bn.get_episode_dataset()
            for episode in episode_list:
                dataset.append(Episode(episode.raw_data, time))
        episodic_bn = BeliefNetwork("Episodic", dataset)
        return episodic_bn

    # Saves the BN's dataset for future reconstruction
    def save(self, path=".\\datasets\\"):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.dataset.save(path + self.name + ".csv")

    # Calculates the probability distribution
    def calculate_pdf(self):
        n = len(self.dataset.episode_dataset) + 4   # The 4 factor compensates the initialization at 1 insted of 0
        # Resets all the pdf values
        self.pdf = self.pdf.fromkeys(self.pdf, 1.0)
        for item in self.dataset.episode_dataset:
            item = item.raw_data
            if item == [1, 1, 1, 1]:
                self.pdf['truth_a'] += 1.0
            elif item == [0, 0, 0, 0]:
                self.pdf['truth_b'] += 1.0
            elif item == [0, 0, 0, 1]:
                self.pdf['lie_a'] += 1.0
            elif item == [1, 1, 1, 0]:
                self.pdf['lie_b'] += 1.0
            else:
                print "[ERROR] invalid dataset item not counted: " + str(item)
        for key in self.pdf:
            self.pdf[key] /= n

    # Calculates the information entrophy
    def get_entropy(self):
        entropy = 0
        for key, Px in self.pdf.items():
            if Px != 0:
                entropy += Px * log(Px, 2)
        return -1 * entropy

    # Calculates the unlikelihood of an episode in terms of information theory
    def surprise(self, episode):
        return log(1 / self.pdf[episode.get_label()], 2)

    # Distance between the episode's surprise value and the network's entropy
    def entropy_difference(self, episode):
        normalization_factor = 2
        entropy = self.get_entropy()
        surprise = self.surprise(episode)
        return round(abs(surprise - entropy), 1) / normalization_factor

    # Importance sampling: calculates how many copies of each episod need to be generated
    def importance_sampling(self, episode, time):
        mitigation_factor = 2.0
        entropy_diff = self.entropy_difference(episode)
        time_fading = (time - episode.time + 1) / mitigation_factor
        consistency = entropy_diff / time_fading
        if 0.0 <= consistency < 0.02:
            duplication_value = 0
        elif 0.02 <= consistency < 0.2:
            duplication_value = 1
        elif 0.2 <= consistency < 0.5:
            duplication_value = 2
        else:
            duplication_value = 3
        return [episode] * duplication_value

    # Systematic Resampling
    @staticmethod
    def systematic_resampling(samples, to_generate=10):
        output = []
        sample_size = len(samples)
        x = randint(0, sample_size)
        increment = randint(2, sample_size-1)     # Avoids increments with undesirable effects (0, 1, len)
        for i in range(to_generate):
            output.append(samples[x % sample_size])
            x += increment
        return output

    # Creates an episodic belief network based on previous beliefs
    @staticmethod
    def create_episodic(bn_list, time, generated_episodes=6, name="EpisodicMemory"):
        weighted_samples = []
        for bn in bn_list:
            episode_list = bn.get_episode_dataset()
            for episode in episode_list:
                samples = bn.importance_sampling(episode, time)
                if samples:
                    weighted_samples.append(samples)
        # Flattens out list of lists
        weighted_samples = [item for sublist in weighted_samples for item in sublist]
        # Checks that there are enough samples to produce a systematic resampling
        if len(weighted_samples) < 4:
            print "create_episodic: not enough samples. Needed at least 4, found " + str(len(weighted_samples))
            quit()
        # Shuffles the list to prevent the first items to be the most likely to be selected
        shuffle(weighted_samples)
        # Now peform Systematic Resampling
        dataset = BeliefNetwork.systematic_resampling(weighted_samples, to_generate=generated_episodes)
        # Copy the list without reference (avoids timing changes to the original episodes)
        unreferenced_dataset = copy.deepcopy(dataset)
        output_dataset = []
        for sample in unreferenced_dataset:
            # Change the time value of all the samples to the current one
            sample.time = time
            output_dataset.append(sample)
            # Generate the symmetric episode
            output_dataset.append(sample.generate_symmetric())
        return BeliefNetwork(name, output_dataset)
