import tensorflow as tf
from keras import backend
from keras import ops
from tensorflow.keras.layers import Layer
from Note.nn.initializer import initializer
from multiprocessing import Manager
from Note.nn.Module import Module


class batch_norm:
    def __init__(self, input_size=None, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', keepdims=True, trainable=True, parallel=True, dtype='float32'):
        self.input_size=input_size
        self.axis=axis
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.keepdims=keepdims
        self.trainable=trainable
        self.parallel=parallel
        self.dtype=dtype
        self.train_flag=True
        if input_size!=None:
            self.output_size=input_size
            self.moving_mean=tf.zeros([input_size],dtype)
            self.moving_var=tf.ones([input_size],dtype)
            if parallel:
                manager=Manager()
                self.moving_mean=manager.list([self.moving_mean])
                self.moving_var=manager.list([self.moving_var])
                Module.ctl_list.append(self.convert_to_list)
                Module.ctsl_list.append(self.convert_to_shared_list)
            self.param=[]
            if center==True:
                self.beta=initializer([input_size], beta_initializer, dtype)
                if trainable==True:
                    self.param.append(self.beta)
            else:
                self.beta=None
            if scale==True:
                self.gamma=initializer([input_size], gamma_initializer, dtype)
                if trainable==True:
                    self.param.append(self.gamma)
            else:
                self.gamma=None
            Module.param.extend(self.param)
    
    
    def build(self):
        self.output_size=self.input_size
        self.moving_mean=tf.zeros([self.input_size],self.dtype)
        self.moving_var=tf.ones([self.input_size],self.dtype)
        if self.parallel:
            manager=Manager()
            self.moving_mean=manager.list([self.moving_mean])
            self.moving_var=manager.list([self.moving_var])
            Module.ctl_list.append(self.convert_to_list)
            Module.ctsl_list.append(self.convert_to_shared_list)
        self.param=[]
        if self.center==True:
            self.beta=initializer([self.input_size], self.beta_initializer, self.dtype)
            if self.trainable==True:
                self.param.append(self.beta)
        else:
            self.beta=None
        if self.scale==True:
            self.gamma=initializer([self.input_size], self.gamma_initializer, self.dtype)
            if self.trainable==True:
                self.param.append(self.gamma)
        else:
            self.gamma=None
        Module.param.extend(self.param)
        return
    
    
    def __call__(self, data, train_flag=True):
        if data.dtype!=self.dtype:
            data=tf.cast(data,self.dtype)
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        self.train_flag=train_flag
        if self.train_flag:
            mean, var = tf.nn.moments(data, self.axis, keepdims=self.keepdims)
            if self.parallel:
                self.moving_mean[0]=self.moving_mean[0] * self.momentum + mean * (1 - self.momentum)
                self.moving_var[0]=self.moving_var[0] * self.momentum + var * (1 - self.momentum)
            else:
                self.moving_mean=self.moving_mean * self.momentum + mean * (1 - self.momentum)
                self.moving_var=self.moving_var * self.momentum + var * (1 - self.momentum)
            output = tf.nn.batch_normalization(data,
                                               mean=mean,
                                               variance=var,
                                               offset=self.beta,
                                               scale=self.gamma,
                                               variance_epsilon=self.epsilon)
        else:
            if self.parallel:
                output = tf.nn.batch_normalization(data,
                                   mean=self.moving_mean[0],
                                   variance=self.moving_var[0],
                                   offset=self.beta,
                                   scale=self.gamma,
                                   variance_epsilon=self.epsilon)
            else:
                output = tf.nn.batch_normalization(data,
                                                   mean=self.moving_mean,
                                                   variance=self.moving_var,
                                                   offset=self.beta,
                                                   scale=self.gamma,
                                                   variance_epsilon=self.epsilon)
        return output
    
    
    def convert_to_list(self):
        self.moving_mean=list(self.moving_mean)
        self.moving_var=list(self.moving_var)
        return
    
    
    def convert_to_shared_list(self,manager):
        self.moving_mean=manager.list(self.moving_mean)
        self.moving_var=manager.list(self.moving_var)
        return


class batch_norm_(Layer):
    def __init__(self, input_size=None, axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', trainable=True, synchronized=False):
        self.input_size=input_size
        self.axis=axis
        self.momentum=momentum
        self.epsilon=epsilon
        self.center=center
        self.scale=scale
        self.beta_initializer=beta_initializer
        self.gamma_initializer=gamma_initializer
        self.trainable=trainable
        self.synchronized=synchronized
        self.train_flag=True
        if input_size!=None:
            self.output_size=input_size
            self.moving_mean=tf.Variable(tf.zeros([input_size]))
            self.moving_variance=tf.Variable(tf.ones([input_size]))
            self.param=[]
            if center==True:
                self.beta=initializer([input_size], beta_initializer)
                if trainable==True:
                    self.param.append(self.beta)
            else:
                self.beta=None
            if scale==True:
                self.gamma=initializer([input_size], gamma_initializer)
                if trainable==True:
                    self.param.append(self.gamma)
            else:
                self.gamma=None
            Module.param.extend(self.param)
    
    
    def build(self):
        self.output_size=self.input_size
        self.moving_mean=tf.Variable(tf.zeros([self.input_size]))
        self.moving_variance=tf.Variable(tf.ones([self.input_size]))
        self.param=[]
        if self.center==True:
            self.beta=initializer([self.input_size], self.beta_initializer)
            if self.trainable==True:
                self.param.append(self.beta)
        else:
            self.beta=None
        if self.scale==True:
            self.gamma=initializer([self.input_size], self.gamma_initializer)
            if self.trainable==True:
                self.param.append(self.gamma)
        else:
            self.gamma=None
        Module.param.extend(self.param)
        return
    
    
    def __call__(self, data, train_flag=True, mask=None):
        data=tf.cast(data,'float32')
        if self.input_size==None:
            self.input_size=data.shape[-1]
            self.build()
        self.train_flag=train_flag
        if train_flag and self.trainable:
            mean, variance = self._moments(
                data,
                mask,
            )
            moving_mean = ops.cast(self.moving_mean, data.dtype)
            moving_variance = ops.cast(self.moving_variance, data.dtype)
            self.moving_mean.assign(
                ops.cast(
                    moving_mean * self.momentum + mean * (1.0 - self.momentum),
                    data.dtype,
                )
            )
            self.moving_variance.assign(
                ops.cast(
                    moving_variance * self.momentum
                    + variance * (1.0 - self.momentum),
                    data.dtype,
                )
            )
        else:
            moving_mean = ops.cast(self.moving_mean, data.dtype)
            moving_variance = ops.cast(self.moving_variance, data.dtype)
            mean = moving_mean
            variance = moving_variance
    
        if self.scale:
            gamma = ops.cast(self.gamma, data.dtype)
        else:
            gamma = None
    
        if self.center:
            beta = ops.cast(self.beta, data.dtype)
        else:
            beta = None
    
        outputs = ops.batch_normalization(
            x=data,
            mean=mean,
            variance=variance,
            axis=self.axis,
            offset=beta,
            scale=gamma,
            epsilon=self.epsilon,
        )
        return outputs
    
    
    def _moments(self, inputs, mask):
        reduction_axes = list(range(len(inputs.shape)))
        del reduction_axes[self.axis]
        _reduction_axes = reduction_axes
        if mask is None:
            return ops.moments(
                inputs,
                axes=_reduction_axes,
                synchronized=self.synchronized,
            )

        mask_weights = ops.cast(
            mask,
            inputs.dtype,
        )
        mask_weights_broadcasted = ops.expand_dims(
            mask_weights,
            axis=-1,
        )
        weighted_inputs = mask_weights_broadcasted * inputs

        weighted_input_sum = ops.sum(
            weighted_inputs,
            _reduction_axes,
            keepdims=True,
        )
        sum_of_weights = ops.sum(
            mask_weights_broadcasted,
            _reduction_axes,
            keepdims=True,
        )
        mean = weighted_input_sum / (sum_of_weights + backend.config.epsilon())

        difference = weighted_inputs - mean
        squared_difference = ops.square(difference)
        weighted_distsq = ops.sum(
            mask_weights_broadcasted * squared_difference,
            _reduction_axes,
            keepdims=True,
        )
        variance = weighted_distsq / (sum_of_weights + backend.config.epsilon())

        return ops.squeeze(mean), ops.squeeze(variance)
