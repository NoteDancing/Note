import tensorflow as tf


class Module:
    param=[]
    param_dict=dict()
    param_dict['dense_weight']=[]
    param_dict['dense_bias']=[]
    param_dict['conv2d_weight']=[]
    param_dict['conv2d_bias']=[]
    layer_dict=dict()
    layer_param=dict()
    ctl_list=[]
    ctsl_list=[]
    name=None
    name_=None
    
    
    def __init__(self):
        Module.init()
        self.param=Module.param
        self.param_dict=Module.param_dict
        self.layer_dict=Module.layer_dict
        self.layer_param=Module.layer_param
    
    
    def apply(func):
        for layer in Module.layer_dict[Module.name]:
            func(layer)
        return
    
    
    def cast_param(key,dtype):
        for param in Module.param_dict[key]:
            param.assign(tf.cast(param,dtype))
        return
    
    
    def freeze(key):
        for param in Module.layer_param[key]:
            param.trainable=False
        return
    
    
    def unfreeze(key):
        for param in Module.layer_param[key]:
            param.trainable=True
        return
    
    
    def convert_to_list():
        for ctl in Module.ctl_list:
            ctl()
        return
    
    
    def convert_to_shared_list(manager):
        for ctsl in Module.ctsl_list:
            ctsl(manager)
        return
    
    
    def init_():
        Module.name=None
        Module.name_=None
        return
    
    
    def init():
        Module.param.clear()
        Module.param_dict['dense_weight'].clear()
        Module.param_dict['dense_bias'].clear()
        Module.param_dict['conv2d_weight'].clear()
        Module.param_dict['conv2d_bias'].clear()
        Module.layer_dict=dict()
        Module.layer_param=dict()
        Module.ctl_list.clear()
        Module.ctsl_list.clear()
        Module.name=None
        Module.name_=None
        return
