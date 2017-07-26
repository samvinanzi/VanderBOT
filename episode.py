"""
This class models single actions performed during the Vanderbilt experiment.
"""

class Episode:
    def __init__(self, data=None, time=0):
        if data != [1, 1, 1, 1] and data != [0, 0, 0, 0] and data != [1, 1, 1, 0] and data != [0, 0, 0, 1]:
            print "[ERROR] Episode. Invalid data input: " + str(data)
            quit(-1)
        self.raw_data = data
        self.time = time

    # Gets an appropriate label describing it's data
    def get_label(self):
        if self.raw_data == [1, 1, 1, 1]:
            return 'truth_a'
        elif self.raw_data == [0, 0, 0, 0]:
            return 'truth_b'
        elif self.raw_data == [0, 0, 0, 1]:
            return 'lie_a'
        elif self.raw_data == [1, 1, 1, 0]:
            return 'lie_b'
        else:
            print "[ERROR] Episode.get_label: invalid raw data " + str(self.raw_data)
            quit(-1)

    def __str__(self):
        return "Time = " + str(self.time) + ", Data = " + str(self.raw_data)

    # Generates the symmetric lie / truth episode
    def generate_symmetric(self):
        new_raw_data = []
        if self.raw_data == [1, 1, 1, 1]:
            new_raw_data = [0, 0, 0, 0]
        elif self.raw_data == [0, 0, 0, 0]:
            new_raw_data = [1, 1, 1, 1]
        elif self.raw_data == [0, 0, 0, 1]:
            new_raw_data = [1, 1, 1, 0]
        elif self.raw_data == [1, 1, 1, 0]:
            new_raw_data = [0, 0, 0, 1]
        else:
            print "[ERROR] Episode.generate_symmetric: invalid raw data " + str(self.raw_data)
            quit(-1)
        return Episode(new_raw_data, self.time)
