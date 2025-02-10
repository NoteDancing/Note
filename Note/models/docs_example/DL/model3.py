"""
This example demonstrates how to use Note's Adahessian optimizer 
by modifying the train_step function inherited from the Model class.
"""
import tensorflow as tf
from Note import nn

class Model(nn.Model):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add(nn.conv2d(32, 3, activation='relu'))
        self.layers.add(nn.max_pool2d())
        self.layers.add(nn.conv2d(64, 3, activation='relu'))
        self.layers.add(nn.max_pool2d())
        self.layers.add(nn.flatten())
        self.layers.add(nn.dense(64, activation='relu'))
        self.layers.add(nn.dense(10))
    
    def __call__(self, x):
        return self.layers(x)

    @tf.function(jit_compile=True)
    def train_step(self, train_data, labels, loss_object, train_loss, train_accuracy, optimizer):
        with tf.GradientTape() as tape:
            output = self.__call__(train_data)
            loss = loss_object(labels, output)
        gradients = tape.gradient(loss, self.param)
        optimizer.apply_gradients(zip(gradients, self.param), tape)
        train_loss(loss)
        if train_accuracy!=None:
            acc=train_accuracy(labels, output)
            return loss,acc
        return loss,None