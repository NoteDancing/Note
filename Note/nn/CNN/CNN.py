import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from keras.preprocessing.image import ImageDataGenerator
import time


def write_data(data,path):
    output_file=open(path,'wb')
    pickle.dump(data,output_file)
    output_file.close()
    return


def read_data(path,dtype=np.float32):
    input_file=open(path,'rb')
    data=pickle.load(input_file)
    return np.array(data,dtype=dtype)


def write_data_csv(data,path,dtype=None,index=False,header=False):
    if dtype==None:
        data=pd.DataFrame(data)
        data.to_csv(path,index=index,header=header)
    else:
        data=np.array(data,dtype=dtype)
        data=pd.DataFrame(data)
        data.to_csv(path,index=index,header=header)
    return
        

def read_data_csv(path,dtype=None,header=None):
    if dtype==None:
        data=pd.read_csv(path,header=header)
        return np.array(data)
    else:
        data=pd.read_csv(path,header=header)
        return np.array(data,dtype=dtype)


class cnn:
    def __init__(self,train_data=None,train_labels=None):
        self.graph=tf.Graph()
        self.train_data=train_data
        self.train_labels=train_labels
        with self.graph.as_default():
            if type(train_data)==np.ndarray:
                self.shape0=train_data.shape[0]
                self.data_shape=train_data.shape
                self.labels_shape=train_labels.shape
                self.data=tf.placeholder(dtype=train_data.dtype,shape=[None,None,None,None],name='data')
                self.labels=tf.placeholder(dtype=train_labels.dtype,shape=[None,None],name='labels')
                self.data_dtype=train_data.dtype
                self.labels_dtype=train_labels.dtype
        self.conv=[]
        self.max_pool=None
        self.avg_pool=None
        self.fc=[]
        self.function=[]
        self.weight_conv=[]
        self.weight_fc=[]
        self.bias_conv=[]
        self.bias_fc=[]
        self.last_weight_conv=[]
        self.last_weight_fc=[]
        self.last_bias_conv=[]
        self.last_bias_fc=[]
        self.activation=[]
        self.activation_fc=[]
        self.flattened_len=None
        self.batch=None
        self.epoch=None
        self.l2=None
        self.dropout=None
        self.optimizer=None
        self.lr=None
        self.train_loss=None
        self.train_accuracy=None
        self.train_loss_list=[]
        self.train_accuracy_list=[]
        self.test_loss=None
        self.test_accuracy=None
        self.continue_train=None
        self.continue_flag=None
        self.flag=None
        self.end_flag=False
        self.test_flag=None
        self.time=None
        self.cpu_gpu='/gpu:0'
        self.use_cpu_gpu='/gpu:0'
    
    
    def data_enhance(self,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,
                     shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest'):
        datagen=ImageDataGenerator(rotation_range=rotation_range,width_shift_range=width_shift_range,height_shift_range=height_shift_range,
                                   shear_range=shear_range,zoom_range=zoom_range,horizontal_flip=horizontal_flip,fill_mode=fill_mode)
        for data in datagen.flow(self.train_data,batch_size=self.train_data.shape[0]):
            self.train_data=data
            break
        return
    
    
    def weight_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
            
    
    def bias_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
        
        
    def conv_f(self,data,weight,i):
        return tf.nn.conv2d(data,weight,strides=[1,self.conv[i][2],self.conv[i][2],1],padding=self.conv[i][3])
    
    
    def max_pool_f(self,data,i):
        return tf.nn.max_pool(data,ksize=[1,self.max_pool[i][0],self.max_pool[i][0],1],strides=[1,self.max_pool[i][1],self.max_pool[i][1],1],padding=self.max_pool[i][2])
    
    
    def avg_pool_f(self,data,i):
        return tf.nn.avg_pool(data,ksize=[1,self.avg_pool[i][0],self.avg_pool[i][0],1],strides=[1,self.avg_pool[i][1],self.avg_pool[i][1],1],padding=self.avg_pool[i][2])
    
    
    def structure(self,conv=None,max_pool=None,avg_pool=None,fc=None,function=None,mean=0,stddev=0.07,dtype=tf.float32):
        with self.graph.as_default():
            self.continue_train=False
            self.total_epoch=0
            self.flag=None
            self.end_flag=False
            self.test_flag=False
            self.train_loss_list.clear()
            self.train_accuracy_list.clear()
            self.weight_conv.clear()
            self.bias_conv.clear()
            self.weight_fc.clear()
            self.bias_fc.clear()
            self.conv=conv
            self.max_pool=max_pool
            self.avg_pool=avg_pool
            self.fc=fc
            self.function=function
            self.mean=mean
            self.stddev=stddev
            self.dtype=dtype
            self.time=None
            with tf.name_scope('parameter_initialization'):
                for i in range(len(self.conv)):
                    if i==0:
                        self.weight_conv.append(self.weight_init([self.conv[i][0],self.conv[i][0],self.data_shape[3],self.conv[i][1]],mean=mean,stddev=stddev,name='conv_{0}_weight'.format(i+1)))
                        self.bias_conv.append(self.bias_init([self.conv[i][1]],mean=mean,stddev=stddev,name='conv_{0}_bias'.format(i+1)))
                    else:                       
                        self.weight_conv.append(self.weight_init([self.conv[i][0],self.conv[i][0],self.conv[i-1][1],self.conv[i][1]],mean=mean,stddev=stddev,name='conv_{0}_weight'.format(i+1)))
                        self.bias_conv.append(self.bias_init([self.conv[i][1]],mean=mean,stddev=stddev,name='conv_{0}_bias'.format(i+1)))
                return
                    
    
    def forward_propagation_fc(self,data,dropout,shape,use_nn):
        with self.graph.as_default():
            for i in range(len(self.fc)):
                if type(self.cpu_gpu)==str:
                    self.forward_cpu_gpu[1].append(self.cpu_gpu)
                elif len(self.cpu_gpu)!=len(self.fc):
                    self.forward_cpu_gpu[1].append(self.cpu_gpu[1][0])
                else:
                    self.forward_cpu_gpu[1].append(self.cpu_gpu[1][i])
            if use_nn==True:
                for i in range(len(self.fc)):
                    if type(self.use_cpu_gpu)==str:
                        self.forward_cpu_gpu[1].append(self.use_cpu_gpu)
                    else:
                        self.forward_cpu_gpu[1].append(self.use_cpu_gpu[1][i])
            if use_nn==False:
                weight_fc=self.weight_fc
                bias_fc=self.bias_fc
            else:
                weight_fc=[]
                bias_fc=[]
                for i in range(len(self.last_weight_fc)):
                    weight_fc.append(tf.constant(self.last_weight_fc[i]))
                    bias_fc.append(tf.constant(self.last_bias_fc[i]))
            if self.continue_train==True and self.flag==1:
                self.weight_fc=[x for x in range(len(self.fc)+1)]
                self.bias_fc=[x for x in range(len(self.fc)+1)]
                for i in range(len(self.fc)+1):
                    if i==len(self.fc):
                        self.weight_fc[i]=tf.Variable(self.last_weight_fc[i],name='out_weight')
                        self.bias_fc[i]=tf.Variable(self.last_bias_fc[i],name='out_bias')
                    else:
                        self.weight_fc[i]=tf.Variable(self.last_weight_fc[i],name='fc_{0}_weight'.format(i+1))
                        self.bias_fc[i]=tf.Variable(self.last_bias_fc[i],name='fc_{0}_bias'.format(i+1))
            self.flag=0
            self.activation_fc=[x for x in range(len(self.fc))]
            if use_nn!=True and len(self.weight_fc)!=len(self.fc)+1:
                for i in range(len(self.fc)+1):
                    if i==0:
                        self.weight_fc.append(self.weight_init([shape,self.fc[i][0]],mean=self.mean,stddev=self.stddev,name='fc_{0}_weight'.format(i+1)))
                        self.bias_fc.append(self.bias_init([self.fc[i][0]],mean=self.mean,stddev=self.stddev,name='fc_{0}_bias'.format(i+1)))
                    elif i==len(self.fc):
                        self.weight_fc.append(self.weight_init([self.fc[i-1][0],self.train_labels.shape[1]],mean=self.mean,stddev=self.stddev,name='output_weight'))
                        self.bias_fc.append(self.bias_init([self.train_labels.shape[1]],mean=self.mean,stddev=self.stddev,name='output_bias'))
                    else:
                        self.weight_fc.append(self.weight_init([self.fc[i-1][0],self.fc[i][0]],mean=self.mean,stddev=self.stddev,name='fc_{0}_weight'.format(i+1)))
                        self.bias_fc.append(self.bias_init([self.fc[i][0]],mean=self.mean,stddev=self.stddev,name='fc_{0}_bias'.format(i+1)))
            if type(dropout)==list:
                data=tf.nn.dropout(data,dropout[0])
            for i in range(len(self.fc)):
                with tf.device(self.forward_cpu_gpu[1][i]):
                    if self.fc[i][1]=='sigmoid':
                        if i==0:
                            self.activation_fc[i]=tf.nn.sigmoid(tf.matmul(data,weight_fc[i])+bias_fc[i])
                            if type(dropout)==list:
                                self.activation_fc[i]=tf.nn.dropout(self.activation_fc[i],dropout[i+1])
                        else:
                            self.activation_fc[i]=tf.nn.sigmoid(tf.matmul(self.activation_fc[i-1],weight_fc[i])+bias_fc[i])
                            if type(dropout)==list:
                                self.activation_fc[i]=tf.nn.dropout(self.activation_fc[i],dropout[i+1])
                    if self.fc[i][1]=='tanh':
                        if i==0:
                            self.activation_fc[i]=tf.nn.tanh(tf.matmul(data,weight_fc[i])+bias_fc[i])
                            if type(dropout)==list:
                                self.activation_fc[i]=tf.nn.dropout(self.activation_fc[i],dropout[i+1])
                        else:
                            self.activation_fc[i]=tf.nn.tanh(tf.matmul(self.activation_fc[i-1],weight_fc[i])+bias_fc[i])
                            if type(dropout)==list:
                                self.activation_fc[i]=tf.nn.dropout(self.activation_fc[i],dropout[i+1])
                    if self.fc[i][1]=='relu':
                        if i==0:
                            self.activation_fc[i]=tf.nn.relu(tf.matmul(data,weight_fc[i])+bias_fc[i])
                            if type(dropout)==list:
                                self.activation_fc[i]=tf.nn.dropout(self.activation_fc[i],dropout[i+1])
                        else:
                            self.activation_fc[i]=tf.nn.relu(tf.matmul(self.activation_fc[i-1],weight_fc[i])+bias_fc[i])
                            if type(dropout)==list:
                                self.activation_fc[i]=tf.nn.dropout(self.activation_fc[i],dropout[i+1])
                    if self.fc[i][1]=='elu':
                        if i==0:
                            self.activation_fc[i]=tf.nn.elu(tf.matmul(data,weight_fc[i])+bias_fc[i])
                            if type(dropout)==list:
                                self.activation_fc[i]=tf.nn.dropout(self.activation_fc[i],dropout[i+1])
                        else:
                            self.activation_fc[i]=tf.nn.elu(tf.matmul(self.activation_fc[i-1],weight_fc[i])+bias_fc[i])
                            if type(dropout)==list:
                                self.activation_fc[i]=tf.nn.dropout(self.activation_fc[i],dropout[i+1])
            if dropout!=None and type(dropout)!=list:
                self.activation_fc[-1]=tf.nn.dropout(self.activation_fc[-1],dropout)
            return weight_fc,bias_fc
                
                
    def forward_propagation(self,data,dropout=None,use_nn=False):
        with self.graph.as_default():
            self.forward_cpu_gpu=[[],[]]
            for i in range(len(self.conv)):
                if type(self.cpu_gpu)==str:
                    self.forward_cpu_gpu[0].append(self.cpu_gpu)
                elif len(self.cpu_gpu[0][0])!=len(self.conv):
                    self.forward_cpu_gpu[0].append(self.cpu_gpu[0][0])
                else:
                    self.forward_cpu_gpu[0].append(self.cpu_gpu[0][i])
            if use_nn==True:
                for i in range(len(self.conv)):
                    if type(self.use_cpu_gpu)==str:
                        self.forward_cpu_gpu.append(self.use_cpu_gpu)
                    else:
                        self.forward_cpu_gpu.append(self.use_cpu_gpu[0][i])
            if use_nn==False:
                weight_conv=self.weight_conv
                bias_conv=self.bias_conv
            else:
                weight_conv=[]
                bias_conv=[]
                for i in range(len(self.last_weight_conv)):
                    weight_conv.append(tf.constant(self.last_weight_conv[i]))
                    bias_conv.append(tf.constant(self.last_bias_conv[i]))
            self.activation=[x for x in range(len(self.conv))]
            with tf.name_scope('forward_propagation'):
                for i in range(len(self.conv)):
                    with tf.device(self.forward_cpu_gpu[0][i]):
                        if type(self.function)==list:
                            if self.function[i]=='sigmoid':
                                if i==0:
                                    self.activation[i]=tf.nn.sigmoid(self.conv_f(data,weight_conv[i],i)+bias_conv[i])                           
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                                else:
                                    self.activation[i]=tf.nn.sigmoid(self.conv_f(self.activation[i-1],weight_conv[i],i)+bias_conv[i])                             
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                            if self.function[i]=='tanh':
                                if i==0:
                                    self.activation[i]=tf.nn.tanh(self.conv_f(data,weight_conv[i],i)+bias_conv[i])                          
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                                else:
                                    self.activation[i]=tf.nn.tanh(self.conv_f(self.activation[i-1],weight_conv[i],i)+bias_conv[i])                    
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                            if self.function[i]=='relu':
                                if i==0:
                                    self.activation[i]=tf.nn.relu(self.conv_f(data,weight_conv[i],i)+bias_conv[i])       
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                                else:
                                    self.activation[i]=tf.nn.relu(self.conv_f(self.activation[i-1],weight_conv[i],i)+bias_conv[i])
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                            if self.function[i]=='elu':
                                if i==0:
                                    self.activation[i]=tf.nn.elu(self.conv_f(data,weight_conv[i],i)+bias_conv[i])          
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                                else:
                                    self.activation[i]=tf.nn.elu(self.conv_f(self.activation[i-1],weight_conv[i],i)+bias_conv[i])
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                        elif type(self.function)==str:
                            if self.function=='sigmoid':
                                if i==0:
                                    self.activation[i]=tf.nn.sigmoid(self.conv_f(data,weight_conv[i],i)+bias_conv[i])
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                                else:
                                    self.activation[i]=tf.nn.sigmoid(self.conv_f(self.activation[i-1],weight_conv[i],i)+bias_conv[i])                    
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                            if self.function=='tanh':
                                if i==0:
                                    self.activation[i]=tf.nn.tanh(self.conv_f(data,weight_conv[i],i)+bias_conv[i])                        
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                                else:
                                    self.activation[i]=tf.nn.tanh(self.conv_f(self.activation[i-1],weight_conv[i],i)+bias_conv[i])                              
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                            if self.function=='relu':
                                if i==0:
                                    self.activation[i]=tf.nn.relu(self.conv_f(data,weight_conv[i],i)+bias_conv[i])                               
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                                else:
                                    self.activation[i]=tf.nn.relu(self.conv_f(self.activation[i-1],weight_conv[i],i)+bias_conv[i])                               
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                            if self.function=='elu':
                                if i==0:
                                    self.activation[i]=tf.nn.elu(self.conv_f(data,weight_conv[i],i)+bias_conv[i])                                
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                                else:
                                    self.activation[i]=tf.nn.elu(self.conv_f(self.activation[i-1],weight_conv[i],i)+bias_conv[i])                         
                                    if type(self.max_pool)==list and self.max_pool[i][0]!=0:
                                        self.activation[i]=self.max_pool_f(self.activation[i],i)
                                    if type(self.avg_pool)==list and self.avg_pool[i][0]!=0:
                                        self.activation[i]=self.avg_pool_f(self.activation[i],i)
                flattened_layer=tf.reshape(self.activation[-1],[-1,self.activation[-1].shape[1]*self.activation[-1].shape[2]*self.activation[-1].shape[3]])
                shape=flattened_layer.shape
                shape=np.array(shape[1])
                shape=shape.astype(np.int)
                self.flattened_len=shape
                weight_fc,bias_fc=self.forward_propagation_fc(flattened_layer,dropout,shape,use_nn)             
                output=tf.matmul(self.activation_fc[-1],weight_fc[-1])+bias_fc[-1]
                return output
            
            
    def train(self,batch=None,epoch=None,optimizer='Adam',lr=0.001,l2=None,dropout=None,acc=True,train_summary_path=None,model_path=None,one=True,continue_train=False,cpu_gpu=None):
        t1=time.time()
        with self.graph.as_default():
            self.batch=batch
            self.l2=l2
            self.dropout=dropout
            self.optimizer=optimizer
            self.lr=lr
            self.acc=acc
            if continue_train!=True:
                if self.continue_train==True:
                    continue_train=True
                else:
                    self.train_loss_list.clear()
                    self.train_accuracy_list.clear()
            if self.continue_train==False and continue_train==True:
                if self.end_flag==False and self.flag==0:
                    self.epoch=None
                self.train_loss_list.clear()
                self.train_accuracy_list.clear()
                self.continue_train=True
            if cpu_gpu!=None:
                self.cpu_gpu=cpu_gpu
            if type(self.cpu_gpu)==list and len(self.cpu_gpu)!=3:
                train_cpu_gpu='/gpu:0'
            if type(self.cpu_gpu)==str:
                train_cpu_gpu=self.cpu_gpu
            with tf.device(train_cpu_gpu):
                if continue_train==True and self.end_flag==True:
                    self.end_flag=False
                    self.weight_conv=[x for x in range(len(self.conv))]
                    self.bias_conv=[x for x in range(len(self.conv))]
                    for i in range(len(self.conv)):
                        self.weight_conv[i]=tf.Variable(self.last_weight_conv[i],name='conv_{0}_weight'.format(i+1))
                        self.bias_conv[i]=tf.Variable(self.last_bias_conv[i],name='conv_{0}_bias'.format(i+1))
                    self.last_weight_conv.clear()
                    self.last_bias_conv.clear()
                    self.last_weight_fc.clear()
                    self.last_bias_fc.clear()
                if continue_train==True and self.flag==1:
                    self.weight_conv=[x for x in range(len(self.conv))]
                    self.bias_conv=[x for x in range(len(self.conv))]
                    for i in range(len(self.conv)):
                        self.weight_conv[i]=tf.Variable(self.last_weight_conv[i],name='conv_{0}_weight'.format(i+1))
                        self.bias_conv[i]=tf.Variable(self.last_bias_conv[i],name='conv_{0}_bias'.format(i+1))
                    self.last_weight_conv.clear()
                    self.last_bias_conv.clear()
                    self.last_weight_fc.clear()
                    self.last_bias_fc.clear()
                    self.flag=0
#     －－－－－－－－－－－－－－－forward propagation－－－－－－－－－－－－－－－
                train_output=self.forward_propagation(self.data,self.dropout)
#     －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
                with tf.name_scope('train_loss'):
                    if self.labels_shape[1]==1:
                        if self.l2==None:
                            train_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=train_output,labels=self.labels))
                        else:
                            train_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=train_output,labels=self.labels)
                            train_loss=tf.reduce_mean(train_loss+self.l2/2*(sum([tf.reduce_sum(x**2) for x in self.weight_conv])+sum([tf.reduce_sum(x**2) for x in self.weight_fc])))
                    else:
                        if self.l2==None:
                            train_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_output,labels=self.labels))
                        else:
                            train_loss=tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_output,labels=self.labels)
                            train_loss=tf.reduce_mean(train_loss+self.l2/2*(sum([tf.reduce_sum(x**2) for x in self.weight_conv])+sum([tf.reduce_sum(x**2) for x in self.weight_fc])))
                if self.optimizer=='Gradient':
                    opt=tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(train_loss)
                if self.optimizer=='RMSprop':
                    opt=tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(train_loss)
                if self.optimizer=='Momentum':
                    opt=tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.99).minimize(train_loss)
                if self.optimizer=='Adam':
                    opt=tf.train.AdamOptimizer(learning_rate=self.lr).minimize(train_loss)
                if self.acc==True:
                    with tf.name_scope('train_accuracy'):
                        equal=tf.equal(tf.argmax(train_output,1),tf.argmax(self.labels,1))
                        train_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
                if train_summary_path!=None:
                    train_loss_scalar=tf.summary.scalar('train_loss',train_loss)
                    train_merging=tf.summary.merge([train_loss_scalar])
                    if self.acc==True:
                        train_accuracy_scalar=tf.summary.scalar('train_accuracy',train_accuracy)
                        train_merging=tf.summary.merge([ train_accuracy_scalar])
                    train_writer=tf.summary.FileWriter(train_summary_path)
                config=tf.ConfigProto()
                config.gpu_options.allow_growth=True
                config.allow_soft_placement=True
                sess=tf.Session(config=config)
                sess.run(tf.global_variables_initializer())
                self.sess=sess
                if self.total_epoch==0:
                    epoch=epoch+1
                for i in range(epoch):
                    if self.batch!=None:
                        batches=int((self.shape0-self.shape0%self.batch)/self.batch)
                        total_loss=0
                        total_acc=0
                        random=np.arange(self.shape0)
                        np.random.shuffle(random)
                        train_data=self.train_data[random]
                        train_labels=self.train_labels[random]
                        for j in range(batches):
                            index1=j*self.batch
                            index2=(j+1)*self.batch
                            train_data_batch=train_data[index1:index2]
                            train_labels_batch=train_labels[index1:index2]
                            feed_dict={self.data:train_data_batch,self.labels:train_labels_batch}
                            if i==0 and self.total_epoch==0:
                                batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                            else:
                                batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                            total_loss+=batch_loss
                            if self.acc==True:
                                batch_acc=sess.run(train_accuracy,feed_dict=feed_dict)
                                total_acc+=batch_acc
                        if self.shape0%self.batch!=0:
                            batches+=1
                            index1=batches*self.batch
                            index2=self.batch-(self.shape0-batches*self.batch)
                            train_data_batch=np.concatenate([train_data[index1:],train_data[:index2]])
                            train_labels_batch=np.concatenate([train_labels[index1:],train_labels[:index2]])
                            feed_dict={self.data:train_data_batch,self.labels:train_labels_batch}
                            if i==0 and self.total_epoch==0:
                                batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                            else:
                                batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                            total_loss+=batch_loss
                            if self.acc==True:
                                batch_acc=sess.run(train_accuracy,feed_dict=feed_dict)
                                total_acc+=batch_acc
                        loss=total_loss/batches
                        train_acc=total_acc/batches
                        self.train_loss_list.append(loss.astype(np.float32))
                        self.train_loss=loss
                        self.train_loss=self.train_loss.astype(np.float32)
                        if self.acc==True:
                            self.train_accuracy_list.append(train_acc.astype(np.float32))
                            self.train_accuracy=train_acc
                            self.train_accuracy=self.train_accuracy.astype(np.float32)
                    else:
                        random=np.arange(self.shape0)
                        np.random.shuffle(random)
                        train_data=self.train_data[random]
                        train_labels=self.train_labels[random]
                        feed_dict={self.data:train_data,self.labels:train_labels}
                        if i==0 and self.total_epoch==0:
                            loss=sess.run(train_loss,feed_dict=feed_dict)
                        else:
                            loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                        self.train_loss_list.append(loss.astype(np.float32))
                        self.train_loss=loss
                        self.train_loss=self.train_loss.astype(np.float32)
                        if self.acc==True:
                            accuracy=sess.run(train_accuracy,feed_dict=feed_dict)
                            self.train_accuracy_list.append(accuracy.astype(np.float32))
                            self.train_accuracy=accuracy
                            self.train_accuracy=self.train_accuracy.astype(np.float32)
                    if epoch%10!=0:
                        temp_epoch=epoch-epoch%10
                        temp_epoch=int(temp_epoch/10)
                    else:
                        temp_epoch=epoch/10
                    if temp_epoch==0:
                        temp_epoch=1
                    if i%temp_epoch==0:
                        if continue_train==True:
                            if self.epoch!=None:
                                self.total_epoch=self.epoch+i+1
                            else:
                                self.total_epoch=i
                        if continue_train==True:
                            print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                        else:
                            print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                        if model_path!=None and i%epoch*2==0:
                            self.save(model_path,i,one)
                        if train_summary_path!=None:
                            train_summary=sess.run(train_merging,feed_dict=feed_dict)
                            train_writer.add_summary(train_summary,i)
                print()
                print('last loss:{0:.6f}'.format(self.train_loss))
                if self.acc==True:
                    print('accuracy:{0:.3f}%'.format(self.train_accuracy*100))
                if train_summary_path!=None:
                    train_writer.close()
                if continue_train==True:
                    self.last_weight_conv=sess.run(self.weight_conv)
                    self.last_bias_conv=sess.run(self.bias_conv)
                    self.last_weight_fc=sess.run(self.weight_fc)
                    self.last_bias_fc=sess.run(self.bias_fc)
                    for i in range(len(self.conv)):
                        self.weight_conv[i]=tf.Variable(self.last_weight_conv[i],name='conv_{0}_weight'.format(i+1))
                        self.bias_conv[i]=tf.Variable(self.last_bias_conv[i],name='conv_{0}_bias'.format(i+1))
                    for i in range(len(self.fc)+1):
                        if i==len(self.fc):
                            self.weight_fc[i]=tf.Variable(self.last_weight_fc[i],name='output_weight')
                            self.bias_fc[i]=tf.Variable(self.last_bias_fc[i],name='output_bias')
                        else:
                            self.weight_fc[i]=tf.Variable(self.last_weight_fc[i],name='fc_{0}_weight'.format(i+1))
                            self.bias_fc[i]=tf.Variable(self.last_bias_fc[i],name='fc_{0}_weight'.format(i+1))
                    self.last_weight_conv.clear()
                    self.last_bias_conv.clear()
                    self.last_weight_fc.clear()
                    self.last_bias_fc.clear()
                    sess.run(tf.global_variables_initializer())
                if continue_train==True:
                    if self.epoch!=None:
                        self.total_epoch=self.epoch+epoch
                    else:
                        self.total_epoch=epoch-1
                    self.epoch=self.total_epoch
                if continue_train!=True:
                    self.epoch=epoch-1
                t2=time.time()
                _time=t2-t1
                if continue_train!=True or self.time==None:
                    self.time=_time
                else:
                    self.time+=_time
                print('time:{0:.3f}s'.format(self.time))
                return
    
    
    def end(self):
        with self.graph.as_default():
            self.end_flag=True
            self.last_weight_conv=self.sess.run(self.weight_conv)
            self.last_bias_conv=self.sess.run(self.bias_conv)
            self.last_weight_fc=self.sess.run(self.weight_fc)
            self.last_bias_fc=self.sess.run(self.bias_fc)
            self.weight_conv.clear()
            self.bias_conv.clear()
            self.weight_fc.clear()
            self.bias_fc.clear()
            self.total_epoch=self.epoch
            self.sess.close()
            return
    
    
    def test(self,test_data,test_labels,batch=None):
        with self.graph.as_default():
            if len(self.last_weight)==0 or self.test_flag==False:
                use_nn=False
            elif len(self.last_weight)!=0 and self.test_flag!=False:
                use_nn=True
            self.test_flag=True
            shape=test_labels.shape
            test_data_placeholder=tf.placeholder(dtype=test_data.dtype,shape=[None,test_data.shape[1],test_data.shape[2],test_data.shape[3]])
            test_labels_placeholder=tf.placeholder(dtype=test_labels.dtype,shape=[None,shape[1]])
            test_output=self.forward_propagation(test_data_placeholder,use_nn=use_nn)
            if shape[1]==1:
                test_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=test_output,labels=test_labels_placeholder))
            else:
                test_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=test_output,labels=test_labels_placeholder))
            equal=tf.equal(tf.argmax(test_output,1),tf.argmax(test_labels_placeholder,1))
            test_accuracy=tf.reduce_mean(tf.cast(equal,tf.float32))
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            sess=tf.Session(config=config)
            if batch!=None:
                total_test_loss=0
                total_test_acc=0
                test_batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
                for j in range(test_batches):
                    test_data_batch=test_data[j*batch:(j+1)*batch]
                    test_labels_batch=test_labels[j*batch:(j+1)*batch]
                    batch_test_loss=sess.run(test_loss,feed_dict={test_data_placeholder:test_data_batch,test_labels_placeholder:test_labels_batch})
                    total_test_loss+=batch_test_loss
                    batch_test_acc=sess.run(test_accuracy,feed_dict={test_data_placeholder:test_data_batch,test_labels_placeholder:test_labels_batch})
                    total_test_acc+=batch_test_acc
                if test_data.shape[0]%batch!=0:
                    test_batches+=1
                    test_data_batch=np.concatenate([test_data[test_batches*batch:],test_data[:batch-(test_data.shape[0]-test_batches*batch)]])
                    test_labels_batch=np.concatenate([test_labels[test_batches*batch:],test_labels[:batch-(test_labels.shape[0]-test_batches*batch)]])
                    batch_test_loss=sess.run(test_loss,feed_dict={test_data_placeholder:test_data_batch,test_labels_placeholder:test_labels_batch})
                    total_test_loss+=batch_test_loss
                    batch_test_acc=sess.run(test_accuracy,feed_dict={test_data_placeholder:test_data_batch,test_labels_placeholder:test_labels_batch})
                    total_test_acc+=batch_test_acc
                test_loss=total_test_loss/test_batches
                test_acc=total_test_acc/test_batches
                self.test_loss=test_loss
                self.test_accuracy=test_acc
                self.test_loss=self.test_loss.astype(np.float32)
                self.test_accuracy=self.test_accuracy.astype(np.float32)
            else:
                self.test_loss=sess.run(test_loss,feed_dict={test_data_placeholder:test_data,test_labels_placeholder:test_labels})
                self.test_accuracy=sess.run(test_accuracy,feed_dict={test_data_placeholder:test_data,test_labels_placeholder:test_labels})
                self.test_loss=self.test_loss.astype(np.float32)
                self.test_accuracy=self.test_accuracy.astype(np.float32)
            print('test loss:{0:.6f}'.format(self.test_loss))
            print('test accuracy:{0:.3f}%'.format(self.test_accuracy*100))
            sess.close()
            return
        
        
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.epoch))
        print()
        print('l2:{0}'.format(self.l2))
        print()
        print('dropout:{0}'.format(self.dropout))
        print()
        print('optimizer:{0}'.format(self.optimizer))
        print()
        print('learning rate:{0}'.format(self.lr))
        print()
        print('time:{0:.3f}s'.format(self.time))
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0:.6f}'.format(self.train_loss))
        if self.acc==True:
            print()
            print('train accuracy:{0:.3f}%'.format(self.train_accuracy*100))
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        print()
        print('test accuracy:{0:.3f}%'.format(self.test_accuracy*100))
        return
		
    
    def info(self):
        self.train_info()
        if self.test_flag==True:
            print()
            print('-------------------------------------')
            self.test_info()
        return


    def train_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.epoch+1),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.epoch+1),self.train_accuracy_list)
        plt.title('train accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        print('train loss:{0:.6f}'.format(self.train_loss))
        if self.acc==True:
            print()
            print('train accuracy:{0:.3f}%'.format(self.train_accuracy*100))
        return
    
    
    def comparison(self):
        print()
        print('train loss:{0:.6f}'.format(self.train_loss))
        print()
        print('train accuracy:{0:.3f}%'.format(self.train_accuracy*100))
        print()
        print('-------------------------------------')
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        print()
        print('test accuracy:{0:.3f}%'.format(self.test_accuracy*100))
        return
    
    
    def network(self):
        print()
        total_params=0
        for i in range(len(self.conv)+1):
            if i==0:
                print('input layer\t{0}\t{1}'.format(self.data_shape[1],self.data_shape[3]))
                print()
            if type(self.function)==list:
                if i==1:
                    if self.conv[i-1][3]=='SAME':
                        print('conv_layer_{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(i,self.conv[i-1][0],self.data_shape[1],self.conv[i-1][1],np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3],self.function[i-1]))
                        total_params+=np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3]
                        print()
                    else:
                        conv_output_shape=int((self.data_shape[1]-self.conv[i-1][0])/self.conv[i-1][2]+1)
                        print('conv_layer_{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(i,self.conv[i-1][0],conv_output_shape,self.conv[i-1][1],np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3],self.function[i-1]))
                        total_params+=np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3]
                        print()
                    if type(self.max_pool)==list and self.max_pool[i-1][0]!=0:
                        if self.max_pool[i-1][2]=='SAME':
                            print('max_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.max_pool[i-1][0],conv_output_shape,self.conv[i-1][1]))
                            print()
                        else:
                            max_pool_output_shape=int((conv_output_shape-self.max_pool[i-1][0])/self.max_pool[i-1][1]+1)
                            print('max_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.max_pool[i-1][0],max_pool_output_shape,self.conv[i-1][1]))
                            print()
                    if type(self.avg_pool)==list and self.avg_pool[i-1][0]!=0:
                        if self.avg_pool[i-1][2]=='SAME':
                            print('avg_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.avg_pool[i-1][0],conv_output_shape,self.conv[i-1][1]))
                            print()
                        else:
                            avg_pool_output_shape=int((conv_output_shape-self.avg_pool[i-1][0])/self.avg_pool[i-1][1]+1)
                            print('avg_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.avg_pool[i-1][0],avg_pool_output_shape,self.conv[i-1][1]))
                            print()
                elif i>0 and i<len(self.conv):
                    if self.conv[i-1][3]=='SAME':
                        print('conv_layer_{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(i,self.conv[i-1][0],self.data_shape[1],self.conv[i-1][1],np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3],self.function[i-1]))
                        total_params+=np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3]
                        print()
                    else:
                        conv_output_shape=int((np.array(self.activation[i-2].shape[1]).astype(int)-self.conv[i-1][0])/self.conv[i-1][2]+1)
                        print('conv_layer_{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(i,self.conv[i-1][0],conv_output_shape,self.conv[i-1][1],np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3],self.function[i-1]))
                        total_params+=np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3]
                        print()
                    if type(self.max_pool)==list and self.max_pool[i-1][0]!=0:
                        if self.max_pool[i-1][2]=='SAME':
                            print('max_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.max_pool[i-1][0],conv_output_shape,self.conv[i-1][1]))
                            print()
                        else:
                            max_pool_output_shape=int((conv_output_shape-self.max_pool[i-1][0])/self.max_pool[i-1][1]+1)
                            print('max_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.max_pool[i-1][0],max_pool_output_shape,self.conv[i-1][1]))
                            print()
                    if type(self.avg_pool)==list and self.avg_pool[i-1][0]!=0:
                        if self.avg_pool[i-1][2]=='SAME':
                            print('avg_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.avg_pool[i-1][0],conv_output_shape,self.conv[i-1][1]))
                            print()
                        else:
                            avg_pool_output_shape=int((conv_output_shape-self.avg_pool[i-1][0])/self.avg_pool[i-1][1]+1)
                            print('avg_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.avg_pool[i-1][0],avg_pool_output_shape,self.conv[i-1][1]))
                            print()
            else:
                if i==1:
                    if self.conv[i-1][3]=='SAME':
                        print('conv_layer_{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(i,self.conv[i-1][0],self.data_shape[1],self.conv[i-1][1],np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3],self.function))
                        total_params+=np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3]
                        print()
                    else:
                        conv_output_shape=int((self.data_shape[1]-self.conv[i-1][0])/self.conv[i-1][2]+1)
                        print('conv_layer_{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(i,self.conv[i-1][0],conv_output_shape,self.conv[i-1][1],np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3],self.function))
                        total_params+=np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3]
                        print()
                    if type(self.max_pool)==list and self.max_pool[i-1][0]!=0:
                        if self.max_pool[i-1][2]=='SAME':
                            print('max_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.max_pool[i-1][0],conv_output_shape,self.conv[i-1][1]))
                            print()
                        else:
                            max_pool_output_shape=int((conv_output_shape-self.max_pool[i-1][0])/self.max_pool[i-1][1]+1)
                            print('max_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.max_pool[i-1][0],max_pool_output_shape,self.conv[i-1][1]))
                            print()
                    if type(self.avg_pool)==list and self.avg_pool[i-1][0]!=0:
                        if self.avg_pool[i-1][2]=='SAME':
                            print('avg_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.avg_pool[i-1][0],conv_output_shape,self.conv[i-1][1]))
                            print()
                        else:
                            avg_pool_output_shape=int((conv_output_shape-self.avg_pool[i-1][0])/self.avg_pool[i-1][1]+1)
                            print('avg_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.avg_pool[i-1][0],avg_pool_output_shape,self.conv[i-1][1]))
                            print()
                elif i>0 and i<len(self.conv):               
                    if self.conv[i-1][3]=='SAME':
                        print('conv_layer_{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(i,self.conv[i-1][0],self.data_shape[1],self.conv[i-1][1],np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3],self.function))
                        total_params+=np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3]
                        print()
                    else:
                        conv_output_shape=int((np.array(self.activation[i-2].shape[1]).astype(int)-self.conv[i-1][0])/self.conv[i-1][2]+1)
                        print('conv_layer_{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(i,self.conv[i-1][0],conv_output_shape,self.conv[i-1][1],np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3],self.function))
                        total_params+=np.prod(self.weight_conv[i-1].shape)+self.weight_conv[i-1].shape[3]
                        print()
                    if type(self.max_pool)==list and self.max_pool[i-1][0]!=0:
                        if self.max_pool[i-1][2]=='SAME':
                            print('max_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.max_pool[i-1][0],conv_output_shape,self.conv[i-1][1]))
                            print()
                        else:
                            max_pool_output_shape=int((conv_output_shape-self.max_pool[i-1][0])/self.max_pool[i-1][1]+1)
                            print('max_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.max_pool[i-1][0],max_pool_output_shape,self.conv[i-1][1]))
                            print()
                    if type(self.avg_pool)==list and self.avg_pool[i-1][0]!=0:
                        if self.avg_pool[i-1][2]=='SAME':
                            print('avg_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.avg_pool[i-1][0],conv_output_shape,self.conv[i-1][1]))
                            print()
                        else:
                            avg_pool_output_shape=int((conv_output_shape-self.avg_pool[i-1][0])/self.avg_pool[i-1][1]+1)
                            print('avg_pool_layer_{0}\t{1}\t{2}\t{3}'.format(i,self.avg_pool[i-1][0],avg_pool_output_shape,self.conv[i-1][1]))
                            print()
        for i in range(len(self.fc)+1):
            if i==0:
                print('fc_layer_{0}\t{1}\t{2}\t{3}'.format(i+1,self.fc[i][0],self.flattened_len*self.fc[i][0]+self.fc[i][0],self.fc[i][1]))
                total_params+=self.flattened_len*self.fc[i][0]+self.fc[i][0]
                print()
            elif i==len(self.fc):
                if self.labels_shape[1]==1:
                    print('output layer\t{0}\t{1}\t{2}'.format(self.labels_shape[1],self.fc[i-1][0]*self.labels_shape[1]+self.labels_shape[1],'sigmoid'))
                    total_params+=self.fc[i-1][0]*self.labels_shape[1]+self.labels_shape[1]
                    print()
                else:
                    print('output layer\t{0}\t{1}\t{2}'.format(self.labels_shape[1],self.fc[i-1][0]*self.labels_shape[1]+self.labels_shape[1],'softmax'))
                    total_params+=self.fc[i-1][0]*self.labels_shape[1]+self.labels_shape[1]
                    print()
            else:
                print('fc_layer_{0}\t{1}\t{2}\t{3}'.format(i+1,self.fc[i][0],self.fc[i-1][0]*self.fc[i][0]+self.fc[i][0],self.fc[i][1]))
                total_params+=self.fc[i-1][0]*self.fc[i][0]+self.fc[i][0]
                print()
        print('total params:{0}'.format(total_params))
        return
    
    
    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.last_weight_conv,output_file)
        pickle.dump(self.last_bias_conv,output_file)
        pickle.dump(self.last_weight_fc,output_file)
        pickle.dump(self.last_bias_fc,output_file)
        pickle.dump(self.data_dtype,output_file)
        pickle.dump(self.labels_dtype,output_file)
        pickle.dump(self.shape0,output_file)
        pickle.dump(self.data_shape,output_file)
        pickle.dump(self.labels_shape,output_file)
        pickle.dump(self.conv,output_file)
        pickle.dump(self.max_pool,output_file)
        pickle.dump(self.avg_pool,output_file)
        pickle.dump(self.fc,output_file)
        pickle.dump(self.function,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.l2,output_file)
        pickle.dump(self.dropout,output_file)
        pickle.dump(self.acc,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_accuracy,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_accuracy,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_accuracy_list,output_file)
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.time,output_file)
        pickle.dump(self.cpu_gpu,output_file)
        pickle.dump(self.use_cpu_gpu,output_file)
        output_file.close()
        return
    
    
    def restore(self,model_path):
        input_file=open(model_path,'rb')
        self.last_weight_conv=pickle.load(input_file)
        self.last_bias_conv=pickle.load(input_file)
        self.last_weight_fc=pickle.load(input_file)
        self.last_bias_fc=pickle.load(input_file)
        self.data_dtype=pickle.load(input_file)
        self.labels_dtype=pickle.load(input_file)
        self.shape0=pickle.load(input_file)
        self.data_shape=pickle.load(input_file)
        self.labels_shape=pickle.load(input_file)
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.data=tf.placeholder(dtype=self.data_dtype,shape=[None,None,None,None],name='data')
            self.labels=tf.placeholder(dtype=self.labels_dtype.dtype,shape=[None,None],name='labels')
        self.conv=pickle.load(input_file)
        self.max_pool=pickle.load(input_file)
        self.avg_pool=pickle.load(input_file)
        self.fc=pickle.load(input_file)
        self.function=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.epoch=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.l2=pickle.load(input_file)
        self.dropout=pickle.load(input_file)
        self.acc=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_accuracy=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss=pickle.load(input_file)
            self.test_accuracy=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_accuracy_list=pickle.load(input_file)
        self.epoch=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.time=pickle.load(input_file)
        self.cpu_gpu=pickle.load(input_file)
        self.use_cpu_gpu=pickle.load(input_file)
        self.flag=1
        input_file.close()
        return


    def classify(self,data,one_hot=False,save_path=None,save_csv=None,cpu_gpu=None):
        with self.graph.as_default():
            if cpu_gpu!=None:
                self.use_cpu_gpu=cpu_gpu
            if type(self.use_cpu_gpu)==str:
                use_cpu_gpu=self.use_cpu_gpu
            else:
                use_cpu_gpu=self.use_cpu_gpu[-1]
            with tf.device(use_cpu_gpu):
                data=tf.constant(data)
                output=self.forward_propagation(data,use_nn=True)
                config=tf.ConfigProto()
                config.gpu_options.allow_growth=True
                config.allow_soft_placement=True
                with tf.Session(config=config) as sess:
                    output=sess.run(output)
                    if one_hot==True:
                        softmax=np.sum(np.exp(output),axis=1).reshape(output.shape[0],1)
                        softmax=np.exp(output)/softmax
                        index=np.argmax(softmax,axis=1)
                        output=np.zeros([output.shape[0],output.shape[1]])
                        for i in range(output.shape[0]):
                            output[i][index[i]]+=1
                        if save_path!=None:
                            output_file=open(save_path,'wb')
                            pickle.dump(output,output_file)
                            output_file.close()
                        elif save_csv!=None:
                            data=pd.DataFrame(output)
                            data.to_csv(save_csv,index=False,header=False)
                        return output
                    else:
                        softmax=np.sum(np.exp(output),axis=1).reshape(output.shape[0],1)
                        softmax=np.exp(output)/softmax
                        output=np.argmax(softmax,axis=1)+1
                        if save_path!=None:
                            output_file=open(save_path,'wb')
                            pickle.dump(output,output_file)
                            output_file.close()
                        elif save_csv!=None:
                            data=pd.DataFrame(output)
                            data.to_csv(save_csv,index=False,header=False)
                        return output
