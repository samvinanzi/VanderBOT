from Vanderbilt import Vanderbilt

"""
Main module that starts the VanderBOT experiment.
"""


def main():
    robot_ip = "192.168.1.100"
    experiment = Vanderbilt(robot_ip=robot_ip, simulation=False, demo_number=6, mature=True, withUpdate=True)
    experiment.help_setup()
    experiment.start()


if __name__ == "__main__":
    main()
