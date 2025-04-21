# Intelligent Control Training for Image-Based Visual Servoing



## Table of contents

- [[#1. Concept of the project:|1. Concept of the project]]
- [[#2. FIRST TASK: Generate a datasets|2. FIRST TASK: Generate a datasets]]
	- [[#2. FIRST TASK: Generate a datasets#2.1 How to run the code:|2.1 How to run the code]]
	- [[#2. FIRST TASK: Generate a datasets#2.2 General explication of the code|2.2 General explication of the code]]
		- [[#2.2 General explication of the code#Other involved files|Other involved files]]
	- [[#2. FIRST TASK: Generate a datasets#2.3 Dataset structure|2.3 Dataset structure]]
		- [[#2.3 Dataset structure#Inputs|Inputs]]
		- [[#2.3 Dataset structure#Outputs|Outputs]]
- [[#3. SECOND TASK: Design the training process|3. SECOND TASK: Design the training process]]

## 1. Concept of the project:

This project explores the integration of artificial intelligence into robot control systems, specifically through the use of artificial neural networks. It builds on previous work with vision-based control approaches and aims to develop a neural network-based controller that can replace traditional control strategies.
## 2. FIRST TASK: Generate a datasets

The goal is to generate a dataset (for training, validation, and testing) by simulating a traditional control system. These datasets will serve as the learning foundation for a neural network. The data is created by running a vision-based controller and recording relevant control inputs and outputs
### 2.1 How to run the code:

**Create the python environment, for which:**

a) You must have **python installed**

b) You can either run (from the root directory of the project):
``` bash
source scripts/.prepare_environment.bash
```

c) Or run the build task made for preparing the environment (from VSCode):
```
F1 > Tasks: Run Build Task > Setup Python Environment
```
This will install all necessary dependencies for running the code

With the environment created, and all the requirements met. The **generation can be started**. 
For testing, there is a script that can be run like:
```bash
source scripts/DATASET_GenerateFullDataSet.bash
```
For running the full dataset generation (defaulting to 20K sequences), there is another script:

### 2.2 General explication of the code

The main script creates a `IVBS` object with the following constraints to make the data consistent:
- **Camera**: `CenteredCamera` with default parameters (with a random position for each sequence)
- **Points to follow**: Set of 4 points, created with the `mkgrid()` method, with a side of 0.5 and a random position in space, forming a plain parallel to the image plain
**Note that:** *For all parameters' random positions, minimum and maximum distances to the* $P(0,0,0)$ *have been set just so the points are not too far away*

Then, for each sequence, the `IVBS` controller is ran step by step and the data from its control is collected in a list of `RowData` objects (defined in `lib/DataStoring.py`) to then be serialized in bunches to save some space in memory.
#### Other involved files

Libraries inside `{root}/lib`:
- `DataStoring`: In charge of storing the data to a specified file, necessary because the data is stored in batches of N sequences
- `IBVSHandler`: In charge of performing each simulation  
### 2.3 Dataset structure

The dataset contains information that can be divided into two groups, depending on weather they will be used as **input** or **output** for training the Machine Learning algorithm
#### Inputs

- `current_points`: The positions on the image plain (in $px$) of the arranged points, as seen by the camera in a certain moment. Set of four points, with two coordinates each $(x, y)$, making a total of 8 columns.
- `desired_points`: The positions on the image plain (in $px$) of the arranged points, As desired when the position is final and correct. Set of four points, with two coordinates each $(x, y)$, making a total of 8 columns.
- `robot_position`: Serialized `SE3` matrix, which is size $4\times4$, 16 columns in total

#### Outputs
- `robot_velocity` Angular and linear velocities for all axes, 6 columns in total
- `final_error`: One column which is the norm between the errors for each point, $(x, y)$

## 3. SECOND TASK: Design the training process

Using the datasets generated in Task 1, this task focuses on designing, training, and validating a neural network that mimics the behavior of a traditional controller. Once trained, the network is integrated into the control loop to act as the decision-making module in place of the conventional controller.
