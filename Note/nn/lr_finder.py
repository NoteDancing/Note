from matplotlib import pyplot as plt
import math
import tensorflow as tf
from Note import nn
import keras.backend as K
from tensorflow.python.util import nest
import numpy as np


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        if type(self.model.optimizer)!=list:
            lr = K.get_value(self.model.optimizer.lr)
        else:
            lr = K.get_value(self.model.optimizer[-1].lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, N=None, train_ds=None, loss_object=None, train_loss=None, global_batch_size=None, dataset_fn=None, num_epochs=None, num_steps_per_epoch=None, strategy=None, start_lr=None, end_lr=None, batch_size=64, epochs=1, **kw_fit):
        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        # Save weights into a file
        initial_weights = [tf.Variable(param.read_value()) for param in nest.flatten(self.model.param)]

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = nn.LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        if strategy == None:
            self.model.train(train_ds=train_ds,
                           loss_object=loss_object, 
                           train_loss=train_loss, 
                           epochs=epochs,
                           callbacks=[callback],
                           **kw_fit)
        else:
            if isinstance(strategy,tf.distribute.MirroredStrategy):
                self.model.distributed_training(train_dataset=train_ds,
                               loss_object=loss_object, 
                               global_batch_size=global_batch_size, 
                               epochs=epochs,
                               strategy=strategy,
                               callbacks=[callback],
                               **kw_fit)
            elif isinstance(strategy,tf.distribute.MultiWorkerMirroredStrategy):
                self.model.distributed_training(train_dataset=train_ds,
                               loss_object=loss_object, 
                               global_batch_size=global_batch_size, 
                               num_epochs=num_epochs, 
                               num_steps_per_epoch=num_steps_per_epoch,
                               strategy=strategy,
                               callbacks=[callback],
                               **kw_fit)
            elif isinstance(strategy,tf.distribute.ParameterServerStrategy):
                self.model.distributed_training(dataset_fn=dataset_fn,
                               loss_object=loss_object, 
                               num_epochs=num_epochs, 
                               num_steps_per_epoch=num_steps_per_epoch,
                               strategy=strategy,
                               callbacks=[callback],
                               **kw_fit)

        # Restore the weights to the state before model fitting
        nn.assign_param(self.model.param, initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        plt.show()

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.ylim(y_lim)
        plt.show()

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]