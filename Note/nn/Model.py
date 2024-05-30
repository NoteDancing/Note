import tensorflow as tf


class Model:
    param=[]
    param_dict=dict()
    param_dict['dense_weight']=[]
    param_dict['dense_bias']=[]
    param_dict['conv2d_weight']=[]
    param_dict['conv2d_bias']=[]
    layer_dict=dict()
    layer_param=dict()
    layer_list=[]
    counter=0
    name_list=[]
    ctl_list=[]
    ctsl_list=[]
    name=None
    name_=None
    
    
    def __init__(self):
        Model.init()
        self.param=Model.param
        self.param_dict=Model.param_dict
        self.layer_dict=Model.layer_dict
        self.layer_param=Model.layer_param
        self.layer_list=Model.layer_list
        
    
    def add():
        Model.counter+=1
        Model.name_list.append('layer'+str(Model.counter))
        return
    
    
    def apply(func):
        for layer in Model.layer_dict[Model.name]:
            func(layer)
        if len(Model.name_list)>0:
            Model.name_list.pop()
            if len(Model.name_list)==0:
                Model.name=None
        return
    
    
    def training(self,flag=False):
        for layer in self.layer_list:
            layer.train_flag=flag
        return
    
    
    def apply_decay(self,str,weight_decay,flag=True):
        if flag==True:
            for param in self.param_dict[str]:
                param.assign(weight_decay * param)
        else:
            for param in self.param_dict[str]:
                param.assign(param / weight_decay)
        return
    
    
    def cast_param(self,key,dtype):
        for param in self.param_dict[key]:
            param.assign(tf.cast(param,dtype))
        return
    
    
    def freeze(self,key):
        for param in self.layer_param[key]:
            param.trainable=False
        return
    
    
    def unfreeze(self,key):
        for param in self.layer_param[key]:
            param.trainable=True
        return
    
    
    def convert_to_list():
        for ctl in Model.ctl_list:
            ctl()
        return
    
    
    def convert_to_shared_list(manager):
        for ctsl in Model.ctsl_list:
            ctsl(manager)
        return
    
    
    def init():
        Model.param.clear()
        Model.param_dict['dense_weight'].clear()
        Model.param_dict['dense_bias'].clear()
        Model.param_dict['conv2d_weight'].clear()
        Model.param_dict['conv2d_bias'].clear()
        Model.layer_dict=dict()
        Model.layer_param=dict()
        Model.layer_list.clear()
        Model.counter=0
        Model.name_list=[]
        Model.ctl_list.clear()
        Model.ctsl_list.clear()
        Model.name=None
        Model.name_=None
        return
