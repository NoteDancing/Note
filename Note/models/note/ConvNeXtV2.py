import tensorflow as tf
import numpy as np
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.depthwise_conv2d import depthwise_conv2d
from Note.nn.layer.dense import dense
from Note.nn.layer.layer_norm import layer_norm
from Note.nn.layer.stochastic_depth import stochastic_depth
from Note.nn.layer.identity import identity
from Note.nn.initializer import initializer
from Note.nn.activation import activation_dict
from Note.nn.Sequential import Sequential
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Model import Model


class GRN:
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        self.gamma = initializer([1, 1, 1, dim], 'zeros')
        self.beta = initializer([1, 1, 1, dim], 'zeros')
    
    
    def __call__(self, x):
        Gx = tf.norm(x, ord=2, axis=(1,2), keepdims=True)
        Nx = tf.math.divide(Gx, tf.math.add(tf.reduce_mean(Gx, axis=-1, keepdims=True), 1e-6))
        return self.gamma * (x * Nx) + self.beta + x


class Block:
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        self.layers=Sequential()
        self.layers.add(depthwise_conv2d(7,input_size=dim,padding='SAME',weight_initializer=['truncated_normal',.02]))
        self.layers.add(layer_norm(epsilon=1e-6))
        self.layers.add(dense(4*dim,weight_initializer=['truncated_normal',.02]))
        self.layers.add(activation_dict['gelu'])
        self.layers.add(GRN(4*dim))
        self.layers.add(dense(dim,weight_initializer=['truncated_normal',.02]))
        self.layers.add(stochastic_depth(drop_path)) if drop_path > 0. else self.layers.add(identity())


    def __call__(self, x):
        input = x
        x = self.layers(x)

        x = input + x
        return x


class ConvNeXtV2(Model):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, model_type='tiny', classes=1000,
                 drop_path_rate=0., head_init_scale=1., include_top=True,
                 pooling=None, device='GPU'
                 ):
        super().__init__()
        self.in_chans = 3
        self.classes = classes
        self.depths = MODEL_CONFIGS[model_type]['depths']
        self.dims = MODEL_CONFIGS[model_type]['dims']
        self.drop_path_rate = drop_path_rate
        self.head_init_scale = tf.constant(head_init_scale)
        self.downsample_layers = [] # stem and 3 intermediate downsampling conv layers
        self.stages = [] # 4 feature resolution stages, each consisting of multiple residual blocks
        self.include_top=include_top
        self.pooling=pooling
        
        stem=Sequential()
        stem.add(conv2d(self.dims[0],kernel_size=4,input_size=self.in_chans,strides=4,
                        weight_initializer=['truncated_normal',.02]))
        stem.add(layer_norm(epsilon=1e-6))
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer=Sequential()
            downsample_layer.add(layer_norm(input_size=self.dims[i],epsilon=1e-6))
            downsample_layer.add(conv2d(self.dims[i+1],kernel_size=2,input_size=self.dims[i],strides=2,
                            weight_initializer=['truncated_normal',.02]))
            self.downsample_layers.append(downsample_layer)
        
        dp_rates = [
            float(x) for x in np.linspace(0.0, self.drop_path_rate, sum(self.depths))
            ]
        cur = 0
        for i in range(4):
            layers=Sequential()
            for j in range(self.depths[i]):
                layers.add(Block(dim=self.dims[i], drop_path=dp_rates[cur + j]))
            stage=layers
            self.stages.append(stage)
            cur += self.depths[i]
        
        self.layer_norm=layer_norm(self.dims[-1],epsilon=1e-6)
        self.head=self.dense(self.classes,self.dims[-1],weight_initializer=['truncated_normal',.02])
        self.head.weight.assign(tf.cast(self.head_init_scale,self.dense.weight.dtype)*self.dense.weight)
        self.head.bias.assign(tf.cast(self.head_init_scale,self.dense.bias.dtype)*self.dense.bias)
        
        self.device=device
        self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer=Adam()
        self.km=0


    def fp(self,data,p=None):
        if self.km==1:
            with tf.device(assign_device(p,self.device)):
                x = data
                for i in range(4):
                    x = self.downsample_layers[i](x)
                    for j in range(self.depths[i]):
                        x = self.stages[i](x)
                if self.include_top:
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                    x = self.layer_norm(x)
                    x = self.head(x)
                else:
                    if self.pooling=="avg":
                        x = tf.math.reduce_mean(x, axis=[1, 2])
                    else:
                        x = tf.math.reduce_max(x, axis=[1, 2])
                return x
        else:
            x = data
            for i in range(4):
                x = self.downsample_layers[i](x,self.km)
                for j in range(self.depths[i]):
                    x = self.stages[i](x,self.km)
            if self.include_top:
                x = tf.math.reduce_mean(x, axis=[1, 2])
                x = self.layer_norm(x)
                x = self.head(x)
            else:
                if self.pooling=="avg":
                    x = tf.math.reduce_mean(x, axis=[1, 2])
                else:
                    x = tf.math.reduce_max(x, axis=[1, 2])
            return x
    
    
    def loss(self,output,labels,p):
        with tf.device(assign_device(p,self.device)):
            loss=self.loss_object(labels,output)
            return loss
    
    
    def GradientTape(self,data,labels,p):
        with tf.device(assign_device(p,self.device)):
            with tf.GradientTape(persistent=True) as tape:
                output=self.fp(data,p)
                loss=self.loss(output,labels,p)
            return tape,output,loss
    
    
    def opt(self,gradient,p):
        with tf.device(assign_device(p,self.device)):
            param=self.optimizer(gradient,self.param,self.bc[0])
            return param


MODEL_CONFIGS = {
    "atto": {
        "depths": [2, 2, 6, 2],
        "dims": [40, 80, 160, 320],
    },
    "femto": {
        "depths": [2, 2, 6, 2],
        "dims": [48, 96, 192, 384],
    },
    "pico": {
        "depths": [2, 2, 6, 2],
        "dims": [64, 128, 256, 512],
    },
    "nano": {
        "depths": [2, 2, 8, 2],
        "dims": [80, 160, 320, 640],
    },
    "tiny": {
        "depths": [3, 3, 9, 3],
        "dims": [96, 192, 384, 768],
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024],
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "dims": [192, 384, 768, 1536],
    },
    "huge": {
        "depths": [3, 3, 27, 3],
        "dims": [352, 704, 1408, 2816],
    },
}