# VanderBOT
Pepper implementation of the cognitive architecture for Trust and Theory of Mind in humanoid robots applied to the developmental psychological experiment designed by Vanderbilt et al. (2011).

In this experiment, a robot will interact with one or more informants in a sticker finding game and it will progressively learn predict their trustworthiness. For more details, please refer to Vinanzi et. al. (*in progress*).

# Features

- Robotic vision algorithms to detect and recognize user's faces other than the markers: Haar Cascade for face detection, Local Binary Histogram Patterns (LBPH) for face recognition;
- Interaction through vocal commands and NAOmarker detection provided by QI APIs;
- Developmental Bayesian Model of Trust (Patacchiola and Cangelosi, 2016) used as a belief network;
- Maximum Likelihood Estimation and Message Passing Algorithm for bayesian learning and inference;
- Custom particle-filter-inspired algorithm for artificial episodic memory;

# Requirements

This code runs on Python 2.7. In addition, the following libraries are required:

| Library | GitHub URL |
| ------ | ------ |
| Numpy | https://github.com/numpy/numpy.git |
| Bayesian Belief Networks | https://github.com/samvinanzi/bayesian-belief-networks.git |
| OpenCV2 with "face" extra module | https://github.com/opencv/opencv_contrib.git |
| qi | https://github.com/aldebaran/libqi.git |

This software can operate in a simulated virtual environment, provided a camera is connected to the computer at runtime, but for best results a SoftBank robot (Nao or Pepper, for example) of latest generation is recommended.

# Experimental Setup

This module has been written for a SoftBank Pepper robot.

*TODO exact measures*

Some robotic behaviours (as methods *look_A()* and *look_B()* from class `robot.py`) should be revised in case of any changes to the robot or the experimental setup.

Printable images of the experiment mat and the marker are available for download and print. Although the table mat is not mandatory, it is recommended as it serves as a visual guide for both the pointers and the experimenters.
Any kind of NAOmarker can be used in place of the one provided.

Tabletop mat: 

![Tabletop mat](/experimental_setup/Experimental%20setup.jpg)

Marker:

![Marker](/experimental_setup/Marker.png)

A3 print size is recommended for the table mat.

# Usage

```python
from Vanderbilt import Vanderbilt

robot_ip = "pepper.local"
simulation = False
demo_number = 6
mature = True
withUpdate = True

experiment = Vanderbilt(robot_ip, simulation, demo_number, mature, withUpdate)
experiment.help_setup()     # Optional, helps to setup the physical environment
experiment.start()
```
