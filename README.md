# Reinforcement Learning With Unity - Garbage Evaporating Autonomous Robot
![My Image](/GEAR-cover.png)

This is a project completed by three students (Damian Bogunowicz, Sangram Gupta, HyunJung Jung) in conjunction with the Chair for Computer Aided Medical Procedures & Augmented Reality of the Technical University of Munich. For this project we have been awarded with maximum grade 1.0. We have created a prototype of an autonomous, intelligent agent for garbage collection named G.E.A.R (Garbage Evaporating Autonomous Robot). The goal of the agent is collect relevant pieces of garbage, while avoiding collisions with static objects (such as chairs or tables). The agent navigates in the environment (a mock-up of German Oktoberfest tent) using camera RBG-D input.

The purpose of this project was to broaden our knowledge in the areas of Reinforcement Learning, Game Development and 3D Computer Vision. If you would like to get more familiar with our work, the detailed description of the project in form of a blog post can be found [here](https://dtransposed.github.io/blog/GEAR.html)

## Getting Started

### Files and Directories

__images__: Sample images from the training set, which has been used for training Semantic Segmentation network.

__unity-environment__: Contains the whole environment, which can be opened in Unity 3D and run (training or inference) using ML-Agents.

__ml-agents environment__: Contains the ml-agents library, modified for the purpose of this project.

__pre-trained-models__: Contains pre-trained models (ML-Agents model or Semantic Segmentation).

### Prerequisites

Before you run the project, you need to install:

* [Unity 3D and Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) 
* OpenCV Python ```sudo apt-get install python-opencv```

## Installing

To run the project:

1. Open the ```/unity-environment/gear-unity``` project using Unity 3D. Additionally, please sure that the ML-Agents Toolkit (including TensorFlowSharp) is correctly installed

You should be welcomed by the following view:
![My Image](/menu.png)


2. Replace the ```ml-agents``` library (in the virtual environment created during ML-Agents toolkit installation) with ```/unity-environment/gear-unity```  



__ADD SCREEN__

## G.E.A.R Training & Inference

### Setting the Parameters of the Environment 

The user can change the parameters of the environment according to his needs. Those are:

### Using PPO and Build-In Semantic Segmentation

This is the default setup for the environment.

#### Training
To train the robot from the scratch using PPO, simply run the command:
```final_model_ppo_no_segmentation```

#### Inference
Set ```Brain Type``` in ```Academy/Brain``` to ```Internal```. To run the inference and see the robot in action, drag ```pre-trained-models/PPO.bytes``` into ```Graph Model``` and run the simulation.

### Using PPO and Custom Semantic Segmentation

To change from default setup to the one which uses external Semantic Segmentation Network (a SegNet, trained using [Semantic Segmentation Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)):

Change camera input to RGB
Change RGB resolution to 512x512, remove black and white.

### Using Imitation Learning

### Using Heuristic






## Authors

* **Damian Bogunowicz** - [dtransposed](https://dtransposed.github.io)
* **Sangram Gupta** - [PurpleBooth](https://github.com/PurpleBooth)
* **HyunJung Jung** - [HyunJungs's LinkedIn](https://www.linkedin.com/in/hyun-jun-jung-1a5b45107)

## Acknowledgments
We would like to thank:

* Mahdi Saleh, Patrick Ruhrkamp and  Benjamin Busam - for the supervision throughout the project
* [GeorgeSeif](https://github.com/GeorgeSeif) - for the cool repository [Semantic Segmentation Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite)
