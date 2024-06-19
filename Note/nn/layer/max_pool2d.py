import tensorflow as tf
from Note import nn


class max_pool2d:
    def __init__(self,kernel_size,strides,padding=0):
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        if not isinstance(padding,str):
            self.zeropadding2d=nn.zeropadding2d(padding=padding)
    
    
    def __call__(self,data):
        if not isinstance(self.padding,str):
            data=self.zeropadding2d(data)
            padding='VALID'
        else:
            padding=self.padding
        return tf.nn.max_pool2d(data,ksize=self.kernel_size,strides=self.strides,padding=padding)
