import csv
import os.path

from episode import Episode

"""
This class collects data samples given by list o by CSV file and performs Maximum Likelihood Estimation (MLE)
"""


class DatasetParser:
    # Initializations at 1 to avoid dividing for zero
    def __init__(self, data):
        self.Xi = [1.0, 1.0]
        self.Yi = [
            [1.0, 1.0],
            [1.0, 1.0]
        ]
        self.Xr = [1.0, 1.0]
        self.Yr = [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ]
        self.trial_number = 1
        # If data contains a csv path, reads the data from it
        if isinstance(data, str) and data[-3:] == "csv" and os.path.isfile(data):
            with open(data, 'rb') as csv_file:
                reader = csv.reader(csv_file, delimiter=',')
                episode_list = []
                for row in reader:
                    # Split data and time
                    datalist = row[:-1]
                    time = row[-1]
                    # Transforms a list of strings in a list of ints
                    int_datalist = map(int, datalist)
                    episode_list.append(Episode(int_datalist, int(time)))
                self.episode_dataset = episode_list
        # If data contains a list, it's pure data
        elif isinstance(data, list):
            self.episode_dataset = data
        else:
            print "[ERROR]. DatasetParser. Invalid data input: " + str(data)
            quit(-1)

    # Parses a dataset and sums each parameter's occurrence
    # Dataset structure: Xr, Yr, Xi, Yi
    # 0: box B, 1: box A
    def read_dataset(self):
        for episode in self.episode_dataset:
            row = episode.raw_data
            row = map(int, row)
            Xr = row[0]
            Yr = row[1]
            Xi = row[2]
            Yi = row[3]
            # Root nodes
            self.Xr[Xr] += 1
            self.Xi[Xi] += 1
            # One-parent node
            self.Yi[Xi][Yi] += 1
            # Two-parent node
            self.Yr[int(str(Yi) + str(Xr), 2)][Yr] += 1
            self.trial_number += 1

    # Normalizes values through the CPT
    def normalize(self):
        self.Xi = self.mle(self.Xi[0], self.Xi[1])
        self.Xr = self.mle(self.Xr[0], self.Xr[1])
        self.Yi = [self.mle(x[0], x[1]) for x in self.Yi]
        self.Yr = [self.mle(x[0], x[1]) for x in self.Yr]

    # Computes MLE
    def mle(self, a, b):
        return [a/(a+b), b/(a+b)]

    # Does all the job
    def estimate_bn_parameters(self):
        self.read_dataset()
        self.normalize()
        parameters = {
            "Xr": self.Xr,
            "Yr": self.Yr,
            "Xi": self.Xi,
            "Yi": self.Yi
        }
        return parameters

    # Saves a dataset on file. Can be used to reconstruct a BN re-estimating its parameters
    def save(self, filename):
        with open(filename, 'wb') as myfile:
            wr = csv.writer(myfile, delimiter=",")
            for episode in self.episode_dataset:
                output = list(episode.raw_data)
                output.append(episode.time)
                wr.writerow(output)

    # Prints episodes
    def print_episodes(self):
        for episode in self.episode_dataset:
            print episode
