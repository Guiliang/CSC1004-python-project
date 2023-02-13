# CSC1004-python-project

## Install miniconda
Download from [here](https://docs.conda.io/en/latest/miniconda.html).
Set up an environment with conda command, check [this tutorial](https://docs.conda.io/projects/conda/en/stable/commands.html).

## Install packages
1. Install pytorch: please refer to [here](https://pytorch.org/get-started/locally/).
2. Install pyyaml:  
Open your commandline and type:  
```pip install pyyaml```  
```pip install matplotlib```

## Task description
The project is about using Convolutional Neural Networks (CNN) to classify handwritten digits.
![minist-img](./imgs/abc.webp "minist")

To achieve it, the code will do the following:
- Load the training config.
- Download the minist dataset (divided into training and testing sets).
- Construct the neural network.
- Update the network parameters with training dataset by minimizing the loss. (Training).
- Test the neural network with the testing dataset. (Testing)
- Plot the results.

## Code filling
Open the [main file](main.py) and modify it according to the requirement at [course page](https://guiliang.github.io/courses/cuhk-csc-1004/project-topics/python_image_net.html). 
