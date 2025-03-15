#imports
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import argparse

#utils objects and methods
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def to_one_hot(labels, num_classes):
    one_hot = np.zeros((len(labels), num_classes)) 
    one_hot[np.arange(len(labels)), labels] = 1    
    return one_hot

def plot_image_per_category(project , entity):
    wandb.init(project=project, entity=entity , name="Question 1")
    

    (x_train, y_train), (_, _) = fashion_mnist.load_data()
    class_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    images_per_class = {}
    for cls in np.unique(y_train):  
        index = np.where(y_train == cls)[0][0] 
        images_per_class[class_labels[cls]] = x_train[index]  
    wandb.log({"Class-wise Images": [wandb.Image(img, caption=label) for label, img in images_per_class.items()]})
    wandb.finish()

#NeuralNetwork class

class NeuralNetwork:
   
    def __init__(self):
        self.layers = None
        self.weights = None
        self.biases = None
        self.activation = None
        self.activation_derivative = None
        self.optimizer = None
    
    def initialize_parameters(self, layers, weight_initialization="xavier", activation_function="sigmoid",loss_function="mean_squared_error"):
 
        self.layers = layers
        self.weights = []
        self.biases = []
        
       
        if activation_function == "sigmoid":
            self.activation = self._sigmoid
            self.activation_derivative = self._sigmoid_derivative
        elif activation_function == "tanh":
            self.activation = self._tanh
            self.activation_derivative = self._tanh_derivative
        elif activation_function == "relu":
            self.activation = self._relu
            self.activation_derivative = self._relu_derivative
        else:
            raise ValueError("Invalid activation function. Choose 'sigmoid', 'tanh', or 'relu'.")

        if loss_function == "cross_entropy":
            self.loss = self._cross_entropy_loss
        elif loss_function == "mean_squared_error":
            self.loss = self._mse_loss  
        else:
            raise ValueError("Invalid loss function. Choose 'cross_entropy' or 'mean_squared_error'.")
        
       
        for i in range(len(layers) - 1):
            if weight_initialization == "xavier":
                
                scale = math.sqrt(2.0 / (layers[i] + layers[i+1]))
                self.weights.append(np.random.randn(layers[i], layers[i+1]) * scale)
            else:
                
                self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.01)
            
            self.biases.append(np.zeros((1, layers[i+1])))
    
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        return 1 - np.square(x)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def _softmax(self, x):
        x = np.clip(x, -500, 500)  # Prevent overflow
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
       
        activations = [X]
        
      
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.activation(z)
            activations.append(a)
        
        
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_out = self._softmax(z_out)
        activations.append(a_out)
        
        return activations
    
    def backward(self, X, y_true, activations):
       
        batch_size = X.shape[0]
        num_layers = len(self.weights)
        
        gradients_w = [None] * num_layers
        gradients_b = [None] * num_layers
        
        if getattr(self, 'loss', None) == self._cross_entropy_loss:
            error = activations[-1] - y_true
        elif getattr(self, 'loss', None) == self._mse_loss:
            error = 2 * (activations[-1] - y_true) / batch_size
        else:
            error = activations[-1] - y_true

       
        for i in reversed(range(num_layers)):
           
            gradients_w[i] = np.dot(activations[i].T, error) / batch_size
            gradients_b[i] = np.sum(error, axis=0, keepdims=True) / batch_size
            
            
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self.activation_derivative(activations[i])
        
        return gradients_w, gradients_b
    
    def train(self, X_train, y_train, x_test=None, y_test=None, hidden_layers=3, hidden_layer_size=32, 
              decay=0, batch=128,momentum=0.9,activation_function="sigmoid", epochs=1000, 
              learning_rate=0.01, optimizer="sgd", beta1=0.9, beta2=0.999, 
              epsilon=1e-8, weight_initialization="xavier",iswandb = False,loss_function="mean_squared_error",verbose=False):
       
        
        layers = [X_train.shape[1]] + [hidden_layer_size] * hidden_layers + [y_train.shape[1]]
        self.initialize_parameters(layers, weight_initialization, activation_function,loss_function=loss_function)
        params = { 
            'optimizer':optimizer,
            'learning_rate':learning_rate,
            'beta1':beta1,
            'beta2':beta2,
            'epsilon':epsilon,
            'decay':decay,
            'momentum':momentum}
        self.optimizer = Optimizer(params)
        self.optimizer.initialize(self.weights, self.biases)
       
       
        
        for epoch in range(epochs):
            
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            
            num_batches = max(1, len(X_train) // batch)
            for i in range(num_batches):
                start_idx = i * batch
                end_idx = min((i + 1) * batch, len(X_train))
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
              
                activations = self.forward(X_batch)
                
               
                gradients_w, gradients_b = self.backward(X_batch, y_batch, activations)
                
               
                self.optimizer.step((gradients_w, gradients_b))
            
            
           
            train_activations = self.forward(X_train)
            test_activations = self.forward(x_test)
            
            if verbose==True:
                train_loss = self.loss(y_train, train_activations[-1])
                test_predictions = test_activations[-1]
                test_loss = self.loss(y_test, test_predictions)
                accuracy = self.loss(y_test, test_predictions)
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
            if iswandb:
              lg={
                      'accuracy':np.sum(np.argmax(train_activations[-1], axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0],
                      'val_accuracy':np.sum(np.argmax(test_activations[-1], axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0],
                      'epoch':epoch+1,
                      'loss':self.loss(y_train, train_activations[-1]),
                      'validation_loss':self.loss(y_test, test_activations[-1]),
                      'confustion_matrix_train':wandb.plot.confusion_matrix(preds = np.argmax(train_activations[-1],axis=1),y_true=np.argmax(y_train,axis=1),class_names=class_labels),
                      'confustion_matrix_test':wandb.plot.confusion_matrix(preds = np.argmax(test_activations[-1],axis=1),y_true=np.argmax(y_test,axis=1),class_names=class_labels)
              }
              wandb.log(lg)
        
        
                
    
    def predict(self, X):
       
        activations = self.forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def _cross_entropy_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Clip to avoid log(0)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def _mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def _compute_accuracy(self, y_true, y_pred):
        true_labels = np.argmax(y_true, axis=1)
        predicted_labels = np.argmax(y_pred, axis=1)
        return np.mean(true_labels == predicted_labels) * 100
    


#Optimizer class
import numpy as np

class Optimizer:
   
    def __init__(self, params=None):
       
        
        self.params = {
            'learning_rate': 0.01,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'decay': 0.0,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        }
        
        
        if params:
            self.params.update(params)
            
        
        self.lr = self.params['learning_rate']
        self.optimizer_type = self.params['optimizer'].lower()
        self.momentum = self.params['momentum']
        self.decay = self.params['decay']
        self.beta1 = self.params['beta1'] 
        self.beta2 = self.params['beta2']
        self.epsilon = self.params['epsilon']
        
        # Time step counter for Adam and Nadam
        self.t = 0
        
        
        self.weights = None
        self.biases = None
        self.w_history = None
        self.b_history = None
        self.w_momentum = None
        self.b_momentum = None
    
    def initialize(self, weights, biases):
        self.weights = weights
        self.biases = biases  
        self.w_history = [np.zeros_like(w) for w in weights]
        self.b_history = [np.zeros_like(b) for b in biases]
        self.w_momentum = [np.zeros_like(w) for w in weights]
        self.b_momentum = [np.zeros_like(b) for b in biases]
    
    def step(self, gradients):
        
        
        self.t += 1
        dw, db = gradients
        
        if self.optimizer_type == 'sgd':
            self._sgd(dw, db)
        elif self.optimizer_type == 'momentum':
            self._momentum(dw, db)
        elif self.optimizer_type == 'nesterov':
            self._nesterov(dw, db)
        elif self.optimizer_type == 'rmsprop':
            self._rmsprop(dw, db)
        elif self.optimizer_type == 'adam':
            self._adam(dw, db)
        elif self.optimizer_type == 'nadam':
            self._nadam(dw, db)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")
    
    def _sgd(self, dw, db):
      
        for i in range(len(self.weights)):   
            self.weights[i] -= self.lr * (dw[i] + self.decay * self.weights[i])
            self.biases[i] -= self.lr * (db[i] + self.decay * self.biases[i])
    
    def _momentum(self, dw, db):
        for i in range(len(self.weights)):
            
            self.w_momentum[i] = self.momentum * self.w_momentum[i] + dw[i]
            self.weights[i] -= self.lr * (self.w_momentum[i] + self.decay * self.weights[i])
            self.b_momentum[i] = self.momentum * self.b_momentum[i] + db[i]
            self.biases[i] -= self.lr * (self.b_momentum[i] + self.decay * self.biases[i])
    
    def _nesterov(self, dw, db):
        for i in range(len(self.weights)):
            self.w_momentum[i] = self.momentum * self.w_momentum[i] + dw[i]
            self.weights[i] -= self.lr * (self.momentum * self.w_momentum[i] + dw[i] + self.decay * self.weights[i])
            
            self.b_momentum[i] = self.momentum * self.b_momentum[i] + db[i]
            self.biases[i] -= self.lr * (self.momentum * self.b_momentum[i] + db[i] + self.decay * self.biases[i])
    
    def _rmsprop(self, dw, db):
        for i in range(len(self.weights)):
  
            self.w_history[i] = self.momentum * self.w_history[i] + (1 - self.momentum) * dw[i]**2
            self.weights[i] -= dw[i] * (self.lr / (np.sqrt(self.w_history[i]) + self.epsilon)) + self.decay * self.weights[i] * self.lr
            self.b_history[i] = self.momentum * self.b_history[i] + (1 - self.momentum) * db[i]**2
            self.biases[i] -= db[i] * (self.lr / (np.sqrt(self.b_history[i]) + self.epsilon)) + self.decay * self.biases[i] * self.lr
    
    def _adam(self, dw, db):
        for i in range(len(self.weights)):
            self.w_momentum[i] = self.beta1 * self.w_momentum[i] + (1 - self.beta1) * dw[i]
            self.w_history[i] = self.beta2 * self.w_history[i] + (1 - self.beta2) * dw[i]**2
            w_momentum_corrected = self.w_momentum[i] / (1 - self.beta1**self.t)
            w_history_corrected = self.w_history[i] / (1 - self.beta2**self.t)
            self.weights[i] -= self.lr * (w_momentum_corrected / (np.sqrt(w_history_corrected) + self.epsilon) + self.decay * self.weights[i])
            
            self.b_momentum[i] = self.beta1 * self.b_momentum[i] + (1 - self.beta1) * db[i]
            self.b_history[i] = self.beta2 * self.b_history[i] + (1 - self.beta2) * db[i]**2
            
            b_momentum_corrected = self.b_momentum[i] / (1 - self.beta1**self.t)
            b_history_corrected = self.b_history[i] / (1 - self.beta2**self.t)
            
            self.biases[i] -= self.lr * (b_momentum_corrected / (np.sqrt(b_history_corrected) + self.epsilon) + self.decay * self.biases[i])
    
    def _nadam(self, dw, db):
        for i in range(len(self.weights)):
           
            self.w_momentum[i] = self.beta1 * self.w_momentum[i] + (1 - self.beta1) * dw[i]
            self.w_history[i] = self.beta2 * self.w_history[i] + (1 - self.beta2) * dw[i]**2
            w_momentum_corrected = self.w_momentum[i] / (1 - self.beta1**self.t)
            w_history_corrected = self.w_history[i] / (1 - self.beta2**self.t)
            w_nesterov_term = self.beta1 * w_momentum_corrected + ((1 - self.beta1) / (1 - self.beta1**self.t)) * dw[i]
            self.weights[i] -= self.lr * (w_nesterov_term / (np.sqrt(w_history_corrected) + self.epsilon) + self.decay * self.weights[i])
            self.b_momentum[i] = self.beta1 * self.b_momentum[i] + (1 - self.beta1) * db[i]
            self.b_history[i] = self.beta2 * self.b_history[i] + (1 - self.beta2) * db[i]**2
            
           
            b_momentum_corrected = self.b_momentum[i] / (1 - self.beta1**self.t)
            b_history_corrected = self.b_history[i] / (1 - self.beta2**self.t)
            b_nesterov_term = self.beta1 * b_momentum_corrected + ((1 - self.beta1) / (1 - self.beta1**self.t)) * db[i]
            
            self.biases[i] -= self.lr * (b_nesterov_term / (np.sqrt(b_history_corrected) + self.epsilon) + self.decay * self.biases[i])







#training function
def train_function(config=None):

        if config==None:
          wandb.init()
          config = wandb.config
        else:
          wandb.init(entity=config["wandb_entity"], project=config["wandb_project"])
        wandb.run.name = f'train.py_hl_{config["hidden_layers"]}_bs_{config["batch_size"]}_ac_{config["activation_function"]}_opt_{config["optimizer"]}_lr_{config["learning_rate"]}_wi_{config["weight_initialization"]}'
        np.random.seed(7)
        nn = NeuralNetwork()
        if config["dataset"] == "fashion_mnist":
            (x_train, y_train), (_, _) = fashion_mnist.load_data()
        elif config["dataset"] == "mnist":
            (x_train, y_train), (_, _) = mnist.load_data()
        y_train = to_one_hot(y_train, 10)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_train = x_train / 255.0
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=7)
        
       
        # Train the model
        nn.train(
            X_train=x_train,
            y_train=y_train,
            x_test=x_val,
            y_test=y_val,
            hidden_layers=config["hidden_layers"],
            hidden_layer_size=config['hidden_layer_size'],
            decay=config["decay"],
            batch=config["batch_size"],
            momentum=config["momentum"],
            activation_function=config["activation_function"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            optimizer=config["optimizer"],
            beta1=config["beta1"],
            beta2=config["beta2"],
            epsilon=config["epsilon"],
            weight_initialization=config["weight_initialization"],  
            loss_function=config["loss"],
            iswandb=config["iswandb"],
            verbose=config["verbose"],
        )

   

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network with specified parameters and log to WandB.")

    # Add arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="da6401_assignment1", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="cs24m048-iit-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function to use")
    parser.add_argument("-o", "--optimizer", type=str, default="nadam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.1e-8, help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier", choices=["random", "Xavier"], help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", type=str, default="relu", choices=["sigmoid", "tanh", "ReLU"], help="Activation function to use")
    parser.add_argument("-v", "--verbose", type=bool, default=False, choices=[True,False], help="verbose function to use")
    parser.add_argument("-i_w", "--iswandb", type=bool, default=True, choices=[True,False], help="to log to wandb or not")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    parameter_config = {
        "wandb_project":args.wandb_project.lower(),
        "wandb_entity":args.wandb_entity.lower(),
        "dataset":args.dataset.lower(),
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "loss":args.loss.lower(),
        "optimizer":args.optimizer.lower(),
        "learning_rate":args.learning_rate,
        "momentum":args.momentum,
        "beta":args.beta,
        "beta1":args.beta1,
        "beta2":args.beta2,
        "epsilon":args.epsilon,
        "decay":args.weight_decay,
        "weight_initialization":args.weight_init.lower(),
        "hidden_layers":args.num_layers,
        "hidden_layer_size":args.hidden_size,
        "activation_function":args.activation.lower(),
        "iswandb":args.iswandb,
        "verbose":args.verbose,
    }

    sweep_config = {
    "method": "bayes", 
    "name": "Q4 WandB sweep", 
    "metric": {"goal": "maximize", "name": "accuracy"},
    'parameters': {
        'epochs': {'values': [parameter_config["epochs"]]},
        'dataset':{'values':[parameter_config["dataset"]]},
        'hidden_layers': {'values': [parameter_config["hidden_layers"]]},
        'hidden_layer_size': {'values': [parameter_config["hidden_layer_size"]]},
        'decay': {'values': [parameter_config["decay"]]},
        'learning_rate': {'values': [parameter_config["learning_rate"]]},
        'optimizer': {'values': [parameter_config["optimizer"]]},
        'batch_size': {'values': [parameter_config["batch_size"]]},
        'weight_initialization': {'values': [parameter_config["weight_initialization"]]},
        'activation_function': {'values': [parameter_config["activation_function"]]},
        'loss': {'values': [parameter_config["loss"]]},
        'momentum': {'values': [parameter_config["momentum"]]},
        'beta1': {'values': [parameter_config["beta1"]]},
        'beta2': {'values': [parameter_config["beta2"]]},
        'epsilon': {'values': [parameter_config["epsilon"]]},
        'iswandb':{'values':[parameter_config["iswandb"]]},
        'verbose':{'values':[parameter_config["verbose"]]},
        'wandb_project':{'values':[parameter_config["wandb_project"]]},
        'wandb_entity':{'values':[parameter_config["wandb_entity"]]}
        
    }

    }


    # sweep_id = wandb.sweep(sweep_config, entity=parameter_config["wandb_entity"], project=parameter_config["wandb_project"])
    # wandb.agent(sweep_id, function=train_function,count=1)   

    train_function(parameter_config)


    

