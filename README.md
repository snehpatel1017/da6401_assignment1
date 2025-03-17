# DA6401 Assignment 1

Implementing a FeedForward Neural Network with Backpropagation from scratch. Also implemented various Optimizers and Loss function from scratch. 


**WandB Report** - https://api.wandb.ai/links/cs24m048-iit-madras/pewp0rgp

**GitHub Repo** - https://github.com/snehpatel1017/da6401_assignment1
# Code Structure

## Main Components

### NeuralNetwork Class
Core implementation of the neural network with methods for:

- **Parameter initialization**
- **Forward propagation**
- **Backward propagation**
- **Training and prediction**
- Various **activation functions**:
  - Sigmoid
  - Tanh
  - ReLU
- **Loss functions**:
  - Cross-entropy
  - Mean squared error

### Optimizer Class
Implementation of various optimization algorithms:

- **Stochastic Gradient Descent (SGD)**
- **Momentum**
- **Nesterov Accelerated Gradient (NAG)**
- **RMSProp**
- **Adam**
- **Nadam** (Nesterov-accelerated Adaptive Moment Estimation)

### Training Function
Configurable function to train the network with specified parameters.

### Utility Functions
- **One-hot encoding**
- **Image plotting**

---

# Command-line Interface

The script includes an **argument parser** that allows for easy configuration of hyperparameters:

- **Dataset selection** (MNIST or Fashion MNIST)
- **Network architecture**:
  - Number of layers
  - Hidden layer size
- **Optimization parameters** (learning rate, momentum, etc.)
- **Weight initialization method**
- **Activation functions**
- **Loss functions**

## Usage Instructions
Run the `train.py` script with appropriate command-line arguments to train and evaluate the neural network using different optimization algorithms and configurations.
<br><br>
Some Example
<br><br>
Run with default parameters.
```bash
python train.py
```

<br><br>
Run with custom parameters.
```bash
python train.py --wandb_entity myname --wandb_project myprojectname --dataset fashion_mnist --epochs 10 --batch_size 64 --loss cross_entropy --optimizer adam --learning_rate 0.001 --momentum 0.9 --beta 0.9 --beta1 0.9 --beta2 0.999 --epsilon 1e-10 --weight_decay 0.0005 --weight_init Xavier --num_layers 4 --hidden_size 64 --activation ReLU
```

<br><br>
Run with -h for parameter help.
<br>
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| -wp, --wandb_project | DL-Assignment-1 | Project name used to track experiments in Weights & Biases dashboard |
| -we, --wandb_entity | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| -d, --dataset | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| -e, --epochs | 10 |  Number of epochs to train neural network.|
| -b, --batch_size | 64 | Batch size used to train neural network. | 
| -l, --loss | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| -o, --optimizer | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| -lr, --learning_rate | 0.001 | Learning rate used to optimize model parameters | 
| -m, --momentum | 0.9 | Momentum used by momentum and nag optimizers. |
| -beta, --beta | 0.9 | Beta used by rmsprop optimizer | 
| -beta1, --beta1 | 0.9 | Beta1 used by adam and nadam optimizers. | 
| -beta2, --beta2 | 0.999 | Beta2 used by adam and nadam optimizers. |
| -eps, --epsilon | 1e-10 | Epsilon used by optimizers. |
| -w_d, --weight_decay | 0.0005 | Weight decay used by optimizers. |
| -w_i, --weight_init | Xavier | choices:  ["random", "Xavier"] | 
| -nhl, --num_layers | 5 | Number of hidden layers used in feedforward neural network. | 
| -sz, --hidden_size | 64 | Number of hidden neurons in a feedforward layer. |
| -a, --activation | tanh | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |
<br>

<br><br>
## Requirements

The following libraries are required to run the neural network implementation:

- **NumPy**  
- **Matplotlib**  
- **Wandb**  
- **Scikit-learn**  
- **Keras** (for dataset loading)  

## Results
Maximum accuracy of 88% for Fashion MNIST dataset and 97% for MNIST dataset is achieved for the following configuration
 
- Model Configuration:
  - Epochs : 10 
  - Number of Hidden Layers: 4
  - Weight Decay: 0
  - Activation Function: ReLU
  - Number of Hidden Neurons: 64/128
  - Learning Rate: 0.001
  - Batch Size: 32
  - Weight Initialization: Xavier
  - Optimizer: NADAM
  - Loss Type: Cross Entropy
