# CWM-Manipulator
Our codebase for a side research project titled “Curriculum-Based World Models for Robotic Manipulation,” targeted for submission to the 2027 International Conference on Robotics (ICIMIRA ITS) or arXiv preprint.

## Background

The problem we aim to address in this research is how to improve the generalization capability of robotic manipulators in smart home environments, particularly in the kitchen. We aim to develop a multimodal robot that can understand visual observations and natural language instructions while adapting to variations in objects, environments, and manipulation tasks.

The methodology we propose involves using a curriculum-based world model that learns the dynamics of the environment from multimodal data, including images, language, and robot actions. Curriculum learning will be used to gradually increase task complexity and environmental variations to improve generalization. The training environment will be simulated using the [AI2Thor simulator](https://ai2thor.allenai.org/manipulathor/), with the robot's capabilities focused solely on arm manipulation tasks.
