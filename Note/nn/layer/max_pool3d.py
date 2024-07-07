import tensorflow as tf
from Note import nn


class max_pool3d:
    def __init__(self,kernel_size=(2,2,2),strides=None,padding=0):
        self.kernel_size=kernel_size
        self.strides=strides if strides!=None else kernel_size
        self.padding=padding
        if not isinstance(padding,str):
            self.zeropadding3d=nn.zeropadding3d(padding=padding)
    
    
    def __call__(self,data):
        if not isinstance(self.padding,str):
            data=self.zeropadding3d(data)
            padding='VALID'
        else:
            padding=self.padding
        return tf.nn.max_pool3d(data,ksize=self.kernel_size,strides=self.strides,padding=padding)
