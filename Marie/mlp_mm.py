import torch
import numpy as np

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # Implement the forward function
        # Linear 1
        z1 = x @ self.parameters["W1"].t() + self.parameters["b1"] # nb * is for element-wise multiplication
        
        # f function:
        if self.f_function == "relu":
          z2 = torch.maximum(z1, torch.tensor([0]))
        elif self.f_function == "sigmoid":
          z2 = torch.sigmoid(z1)
        
        # Linear 2
        z3 = z2 @ self.parameters["W2"].t() + self.parameters["b2"] 

        # g function:
        if self.f_function == "relu":
          y_hat = torch.maximum(z3, torch.tensor([0]))
        elif self.g_function == "sigmoid":
          y_hat = torch.sigmoid(z3)

        # Store the intermediate values in cache
        self.cache = {"x" : x, "z1" : z1, "z2" : z2, "z3" : z3, "y_hat" : y_hat}

        return y_hat 
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        
        """
        
        # Backward pass through activation function g
        if self.g_function == "relu":
          mask = self.cache["z2"] > 0
          dyhat_dz3 = torch.where(mask, torch.ones_like(self.cache["z3"]), torch.zeros_like(self.cache["z3"]))

        elif self.g_function == "sigmoid":
          dy_hatdz3 = torch.sigmoid(self.cache["z3"]) @ (1- self.cache["z3"])
        
        elif self.g_function == "identity":
          dy_hatdz3 = torch.ones_like(self.cache["z3"]) # SHOULD THIS BE AN IDENTITY MATRIX?? 

        self.grads["dJdW2"] = dJdy_hat @ dy_hatdz3.t() @ self.cache["z3"] 
        self.grads["dJdb2"] = dJdy_hat.t() @ dy_hatdz3 @ torch.ones_like(self.parameters["b2"]).unsqueeze(dim = 1)

        # Backward pass through activation function f:
        if self.f_function == "relu":
          mask = self.cache["z1"] > 0
          dz2dz1 = torch.where(mask, torch.ones_like(self.cache["z1"]), torch.zeros_like(self.cache["z1"]))
        
        elif self.f_function == "sigmoid":
          dz2dz1 = torch.sigmoid(self.cache["z1"]) @ (1- torch.sigmoid(self.cache["z1"]))
        
        elif self.f_function == "identity":
          dz2dz1 = torch.ones_like(self.cache["z1"])

        self.grads["dJdW1"] =  self.grads["dJdW2"] @ self.parameters["W2"] @ dz2dz1.t() @ self.cache["x"]
        # self.grads["dJdb1"] =  self.grads["dJdb2"] @ self.parameters["W2"] @ dz2dz1.t() @ torch.ones_like(self.parameters["b1"]).unsqueeze(dim = 1)        self.grads["dJdb1"] =  dJdy_hat @ dy_hatdz3.t() @ self.parameters["W2"] @ dz2dz1.t() @ torch.ones_like(self.parameters["b1"]).unsqueeze(dim = 1)

        dict1 = {"dJdy_hat" : dJdy_hat, "dy_hatdz3" : dy_hatdz3, "dz3dz2" : self.parameters["W2"], "dz2dz1" : dz2dz1, "dz1db1" : torch.ones_like(self.parameters["b1"]).unsqueeze(dim = 1) }

        for key, val in dict1.items():
          print(key, "shape:", val.shape)
        self.grads["dJdb1"] =  dJdy_hat @ dy_hatdz3.t() @ self.parameters["W2"] @ dz2dz1.t() @ torch.ones_like(self.parameters["b1"]).unsqueeze(dim = 1)
    
        # Update weights:
        self.parameters["W1"] = self.parameters["W1"] - self.grads["dJdW1"]
        self.parameters["b1"] = self.parameters["b1"] - self.grads["dJdb1"]
        self.parameters["W2"] = self.parameters["W2"] - self.grads["dJdW2"]
        self.parameters["b2"] = self.parameters["b2"] - self.grads["dJdb2"]

        # clear cache and set gradients to zero :
        self.clear_grad_and_cache()

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    
    loss = np.square(y_hat.numpy() - y.numpy()).mean()
    dJdy_hat = 2 / y.size()[0] * (y_hat - y) 
    
    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    loss = y.numpy() * np.log(y_hat.numpy()) + (1 - y.numpy())*np.log(1 - y_hat.numpy())
    loss = loss.mean()

    dJdy_hat = 1 / y.size()[0] * (y / y_hat + (y - 1)/(1 - y_hat))

    return loss, dJdy_hat











