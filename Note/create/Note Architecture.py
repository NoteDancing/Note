import tensorflow as tf
from tensorflow.python.ops import state_ops
import tensorflow.keras.optimizers as optimizer
import Note.create.optimizer as optimizern
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
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


class unnamed:
    def __init__():
        with tf.name_scope('data'):     
           
            
        with tf.name_scope('parameter'):
            
        
        self.batch=None
        self.epoch=0
        self.optimizer=None
        self.lr=None
        with tf.name_scope('regulation'):
            
            
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.test_flag=False
        self.time=None
        self.processor='GPU:0'
        self.use_processor='GPU:0'
        
    
    def weight_init(self,shape,mean,stddev,name=None):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
            
    
    def bias_init(self,shape,mean,stddev,name=None):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
    
    
    def structure():
        self.epoch=0
        self.total_epoch=0
        self.test_flag=False
        self.train_loss_list.clear()
        self.train_acc_list.clear()
        self.dtype=dtype
        with tf.name_scope('hyperparameter'):
            
            
        self.time=None
        with tf.name_scope('parameter_initialization'):
            
           
            
    @tf.function       
    def forward_propagation():
        with tf.name_scope('processor_allocation'):
               
               
        with tf.name_scope('forward_propagation'):
            
            
            
    def apply_gradient(self,tape,optimizer,loss,variable):
        gradient=tape.gradient(loss,variable)
        optimizer.apply_gradients(zip(gradient,variable))
        return
    
    
    def train(self,batch=None,epoch=None,lr=None,model_path=None,one=True,processor=None):
        t1=time.time()
        self.batch=batch
        self.lr=lr
        with tf.name_scope('regulation'):
            
            
        self.train_loss_list.clear()
        self.train_acc_list.clear()
        if processor!=None:
            self.processor=processor
        with tf.name_scope('processor_allocation'):
            
        
        with tf.device(train_processor):
            with tf.name_scope('variable'):
                
                
            with tf.name_scope('optimizer'):
                
                
            if self.total_epoch==0:
                epoch=epoch+1
            for i in range(epoch):
                if self.batch!=None:
                    batches=int((self.shape0-self.shape0%self.batch)/self.batch)
                    total_loss=0
                    total_acc=0
                    random=np.arange(self.shape0)
                    np.random.shuffle(random)
                    with tf.name_scope('randomize_data'):
                        
                    
                    for j in range(batches):
                        index1=j*self.batch
                        index2=(j+1)*self.batch
                        with tf.name_scope('data_batch'):
                            
                        
                        with tf.GradientTape() as tape:
                            with tf.name_scope('forward_propagation/loss'):
                                
                        
                            if i==0 and self.total_epoch==0:
                                batch_loss=batch_loss.numpy()
                            else:
                                with tf.name_scope('apply_gradient'):
                                    
                                    
                        total_loss+=batch_loss
                        with tf.name_scope('accuracy'):
                     
                        
                        batch_acc=batch_acc.numpy()
                        total_acc+=batch_acc
                    if self.shape0%self.batch!=0:
                        batches+=1
                        index1=batches*self.batch
                        index2=self.batch-(self.shape0-batches*self.batch)
                        with tf.name_scope('data_batch'):
                            
                        
                        with tf.GradientTape() as tape:
                            with tf.name_scope('forward_propagation/loss'):
                                
                            
                            if i==0 and self.total_epoch==0:
                                batch_loss=batch_loss.numpy()
                            else:
                                with tf.name_scope('apply_gradient'):
                                    
                                    
                        total_loss+=batch_loss
                        with tf.name_scope('accuracy'):
                     
                        
                        batch_acc=batch_acc.numpy()
                        total_acc+=batch_acc
                    loss=total_loss/batches
                    train_acc=total_acc/batches
                    self.train_loss_list.append(loss.astype(np.float32))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float32)
                    self.train_acc_list.append(float(train_acc))
                    self.train_acc=train_acc
                    self.train_acc=self.train_acc.astype(np.float32)
                else:
                    random=np.arange(self.shape0)
                    np.random.shuffle(random)
                    with tf.name_scope('randomize_data'):
                        

                    with tf.GradientTape() as tape:
                        with tf.name_scope('forward_propagation/loss'):
                            
                        
                        if i==0 and self.total_epoch==0:
                            loss=train_loss.numpy()
                        else:
                           with tf.name_scope('apply_gradient'):
                                
                                
                    self.train_loss_list.append(loss.astype(np.float32))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float32)
                    with tf.name_scope('accuracy'):
                     
                      
                    acc=train_acc.numpy()
                    self.train_acc_list.append(float(acc))
                    self.train_acc=acc
                    self.train_acc=self.train_acc.astype(np.float32)
                if epoch%10!=0:
                    temp_epoch=epoch-epoch%10
                    temp_epoch=int(temp_epoch/10)
                else:
                    temp_epoch=epoch/10
                if temp_epoch==0:
                    temp_epoch=1
                if i%temp_epoch==0:
                    if self.epoch==0:
                        self.total_epoch=i
                    else:
                        self.total_epoch=self.epoch+i+1
                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch,self.train_loss))
                    if model_path!=None and i%epoch*2==0:
                        self.save(model_path,i,one)
            print()
            print('last loss:{0:.6f}'.format(self.train_loss))
            with tf.name_scope('print_accuracy'):
                    
                
            self.epoch=self.total_epoch
            t2=time.time()
            _time=t2-t1
            if continue_train!=True or self.time==None:
                self.total_time=_time
            else:
                self.total_time+=_time
            print('time:{0:.3f}s'.format(self.time))
            return
    
    
    def test(self,test_data,test_labels,batch=None):
        self.test_flag=True
        if batch!=None:
            total_loss=0
            total_acc=0
            test_batches=int((test_data.shape[0]-test_data.shape[0]%batch)/batch)
            for j in range(test_batches):
                test_data_batch=test_data[j*batch:(j+1)*batch]
                test_labels_batch=test_labels[j*batch:(j+1)*batch]
                with tf.name_scope('loss'):
                
                    
                total_test_loss+=batch_test_loss.numpy()
                with tf.name_scope('accuracy'):
                    
                
                total_acc+=batch_acc.numpy()
            if test_data.shape[0]%batch!=0:
                test_batches+=1
                test_data_batch=np.concatenate([test_data[batches*batch:],test_data[:batch-(test_data.shape[0]-batches*batch)]])
                test_labels_batch=np.concatenate([test_labels[batches*batch:],test_labels[:batch-(test_labels.shape[0]-batches*batch)]])
                with tf.name_scope('loss'):
                    
                
                total_loss+=batch_loss.numpy()
                with tf.name_scope('accuracy'):
                    
                
                total_acc+=batch_acc.numpy()
            test_loss=total_loss/test_batches
            test_acc=total_acc/test_batches
            self.test_loss=test_loss
            self.test_acc=test_acc
            self.test_loss=self.test_loss.astype(np.float32)
            self.test_acc=self.test_acc.astype(np.float32)
        else:
            with tf.name_scope('loss'):
                
                
            with tf.name_scope('accuracy'):
                
                
            self.test_loss=test_loss.numpy().astype(np.float32)
            self.test_acc=test_acc.numpy().astype(np.float32)
        print('test loss:{0:.6f}'.format(self.test_loss))
        with tf.name_scope('print_accuracy'):
            
            
        return
        
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.epoch))
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
        with tf.name_scope('print_accuracy'):
                
                
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        with tf.name_scope('print_accuracy'):
            
        
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
        plt.plot(np.arange(self.epoch+1),self.train_acc_list)
        plt.title('train accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        print('train loss:{0:.6f}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
                
                
        return
    
        
    def comparison(self):
        print()
        print('train loss:{0}'.format(self.train_loss))
        with tf.name_scope('print_accuracy'):
                
            
        if self.test_flag==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            with tf.name_scope('print_accuracy'):
                
                
        return
    
    
    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        with tf.name_scope('save_parameter'):  
            
            
        with tf.name_scope('save_shape0'):
            
            
        pickle.dump(self.batch,output_file)
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.lr,output_file)
        with tf.name_scope('save_hyperparameter'):
            
            
        with tf.name_scope('save_regularization'):
            
            
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_acc,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_acc,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_acc_list,output_file)
        pickle.dump(self.epoch,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.time,output_file)
        pickle.dump(self.processor,output_file)
        pickle.dump(self.use_processor,output_file)
        output_file.close()
        return
    

    def restore(self,model_path):
        input_file=open(model_path,'rb')
        with tf.name_scope('restore_parameter'):
            
            
        with tf.name_scope('restore_shape0'):
            
            
        self.batch=pickle.load(input_file)
        self.epoch=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        with tf.name_scope('restore_regularization'):
            
            
        with tf.name_scope('restore_hyperparameter'):
            
            
        self.total_time=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_acc=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss=pickle.load(input_file)
            self.test_acc=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_acc_list=pickle.load(input_file)
        self.epoch=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        self.use_processor=pickle.load(input_file)
        input_file.close()
        return
