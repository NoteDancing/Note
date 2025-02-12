import tensorflow as tf
from Note import nn
import multiprocessing
import numpy as np
import numpy.ctypeslib as npc
import math
import matplotlib.pyplot as plt
import pickle
import os
import time


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
    layer_eval=dict()
    counter=0
    name_list=[]
    name_list_=[]
    ctl_list=[]
    ctsl_list=[]
    name=None
    name_=None
    train_flag=True
    
    
    def __init__(self):
        Model.init()
        self.param=Model.param
        self.param_dict=Model.param_dict
        self.layer_dict=Model.layer_dict
        self.layer_param=Model.layer_param
        self.layer_list=Model.layer_list
        self.layer_eval=Model.layer_eval
        self.name_list=Model.name_list_
        self.head=None
        self.head_=None
        self.ft_flag=0
        self.ctl_list=Model.ctl_list
        self.ctsl_list=Model.ctsl_list
        self.optimizer=None
        self.path=None
        self.save_freq=1
        self.save_freq_=None
        self.max_save_files=None
        self.steps_per_execution=None
        self.monitor='val_loss'
        self.val_loss=0
        self.val_accuracy=1
        self.save_best_only=False
        self.save_param_only=False
        self.callbacks=[]
        self.stop_training=False
        self.info=dict()
        self.batch_counter=0
        self.path_list=[]
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.shared_test_loss_array=None
        self.shared_test_acc_array=None
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.end_loss=None
        self.end_acc=None
        self.end_test_loss=None
        self.end_test_acc=None
        self.total_epoch=0
        self.time=0
        self.total_time=0
    
    
    def get_info(self):
        self.info['path']=self.path
        self.info['save_freq']=self.save_freq
        self.info['save_freq_']=self.save_freq_
        self.info['max_save_files']=self.max_save_files
        self.info['steps_per_execution']=self.steps_per_execution
        self.info['monitor']=self.monitor
        self.info['val_loss']=self.val_loss
        self.info['val_accuracy']=self.val_accuracy
        self.info['save_best_only']=self.save_best_only
        self.info['save_param_only']=self.save_param_only
        self.info['end_loss']=self.end_loss
        self.info['end_acc']=self.end_acc
        self.info['end_test_loss']=self.end_test_loss
        self.info['end_test_acc']=self.end_test_acc
        self.info['total_epoch']=self.total_epoch
        self.info['time']=self.time
        self.info['total_time']=self.total_time
        if self.info_flag==0:
            try:
                self.info['batch_size']=self.batch_size
                self.info['loss_object']=self.loss_object
                self.info['train_loss']=self.train_loss
                if type(self.optimizer)==list:
                    self.info['optimizer']=[tf.keras.optimizers.serialize(optimizer) for optimizer in self.optimzer]
                else:
                    self.info['optimizer']=tf.keras.optimizers.serialize(self.optimizer)
                self.info['epochs']=self.epochs
                self.info['train_accuracy']=self.train_accuracy
                self.info['test_loss']=self.test_loss
                self.info['test_accuracy']=self.test_accuracy
                self.info['test_batch_size']=self.test_batch_size
                self.info['processes']=self.processes
                self.info['parallel_test']=self.parallel_test_
                self.info['jit_compile']=self.jit_compile
                self.info['p']=self.p
            except Exception:
                pass
        else:
            try:
                self.info['loss_object']=self.loss_object
                self.info['global_batch_size']=self.global_batch_size
                if type(self.optimizer)==list:
                    self.info['optimizer']=[tf.keras.optimizers.serialize(optimizer) for optimizer in self.optimzer]
                else:
                    self.info['optimizer']=tf.keras.optimizers.serialize(self.optimizer)
                self.info['strategy']=self.strategy
                self.info['epochs']=self.epochs
                self.info['num_epochs']=self.num_epochs
                self.info['num_steps_per_epoch']=self.num_steps_per_epoch
                self.info['train_accuracy']=self.train_accuracy
                self.info['test_loss']=self.test_loss
                self.info['test_accuracy']=self.test_accuracy
                self.info['global_test_batch_size']=self.global_test_batch_size
                self.info['eval_steps_per_epoch']=self.eval_steps_per_epoch
                self.info['jit_compile']=self.jit_compile
                self.info['p']=self.p
            except Exception:
                pass
        return self.info
        
    
    def add():
        Model.counter+=1
        Model.name_list.append('layer'+str(Model.counter))
        return
    
    
    def apply(func):
        for layer in Model.layer_dict[Model.name_]:
            if layer.input_size!=None:
                func(layer)
            else:
                layer.init_weights=func
        if len(Model.name_list)>0:
            Model.name_list.pop()
            if len(Model.name_list)==0:
                Model.name_=None
        return
    
    
    def training(self,flag=False):
        Model.train_flag=flag
        for layer in self.layer_list:
            if hasattr(layer,'train_flag'):
                layer.train_flag=flag
            else:
                layer.training=flag
        return
    
    
    def dense(self,num_classes,dim,weight_initializer='Xavier',use_bias=True):
        self.head=nn.dense(num_classes,dim,weight_initializer,use_bias=use_bias)
        return self.head
    
    
    def conv2d(self,num_classes,dim,kernel_size=1,weight_initializer='Xavier',padding='SAME',use_bias=True):
        self.head=nn.conv2d(num_classes,kernel_size,dim,weight_initializer=weight_initializer,padding=padding,use_bias=use_bias)
        return self.head
    
    
    def fine_tuning(self,num_classes,flag=0):
        self.ft_flag=flag
        if flag==0:
            self.head_=self.head
            if isinstance(self.head,nn.dense):
                self.head=nn.dense(num_classes,self.head.input_size,self.head.weight_initializer,use_bias=self.head.use_bias)
            elif isinstance(self.head,nn.conv2d):
                self.head=nn.conv2d(num_classes,self.head.kernel_size,self.head.input_size,weight_initializer=self.head.weight_initializer,padding=self.head.padding,use_bias=self.head.use_bias)
            self.param[-len(self.head.param):]=self.head.param
            for param in self.param[:-len(self.head.param)]:
                param._trainable=False
        elif flag==1:
            for param in self.param[:-len(self.head.param)]:
                param._trainable=True
        else:
            self.head,self.head_=self.head_,self.head
            self.param[-len(self.head.param):]=self.head.param
            for param in self.param[:-len(self.head.param)]:
                param._trainable=True
        return
    
    
    def apply_decay(self,str,weight_decay,flag=True):
        if flag==True:
            for param in self.param_dict[str]:
                param.assign(weight_decay * param)
        else:
            for param in self.param_dict[str]:
                param.assign(param / weight_decay)
        return
    
    
    def cast_param(self,key=None,dtype=None):
        if key is None:
            for param in self.param:
                param.assign(tf.cast(param,dtype))
        else:
            for param in self.param_dict[key]:
                param.assign(tf.cast(param,dtype))
        return
    
    
    def namespace(name=None):
        Model.name=name
        if name!=None:
            Model.name_list_.append(name)
        return
    
    
    def freeze(self,name=None):
        if name==None:
            for name in self.name_list:
                for param in self.layer_param[name]:
                    param._trainable=False
        else:
            for param in self.layer_param[name]:
                param._trainable=False
        return
    
    
    def unfreeze(self,name=None):
        if name==None:
            for name in self.name_list:
                for param in self.layer_param[name]:
                    param._trainable=True
        else:
            for param in self.layer_param[name]:
                param._trainable=True
        return
    
    
    def eval(self,name=None,flag=True):
        if name==None:
            for name in self.name_list:
                if flag:
                    for layer in self.layer_eval[name]:
                        layer.train_flag=False
                else:
                    for name in self.layer_eval.keys():
                        for layer in self.layer_eval[name]:
                            layer.train_flag=True
        else:
            if flag:
                for layer in self.layer_eval[name]:
                    layer.train_flag=False
            else:
                for name in self.layer_eval.keys():
                    for layer in self.layer_eval[name]:
                        layer.train_flag=True
        return
    
    
    def summary(self):
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        total_memory = 0  # Memory usage in bytes

        for param in self.param:
            param_count = tf.size(param).numpy()
            param_dtype_size = param.dtype.size
            param_memory = param_count * param_dtype_size

            total_params += param_count
            total_memory += param_memory

            if param.trainable:
                trainable_params += param_count
            else:
                non_trainable_params += param_count

        def format_memory(bytes_size):
            units = ['Bytes', 'KB', 'MB', 'GB']
            index = 0
            while bytes_size >= 1024 and index < len(units) - 1:
                bytes_size /= 1024
                index += 1
            return f"{bytes_size:.2f} {units[index]}"

        # Print the summary with formatted memory usage
        print("Model Summary")
        print("-------------")
        print(f"Total params: {total_params} ({format_memory(total_memory)})")
        print(f"Trainable params: {trainable_params} ({format_memory(trainable_params * param_dtype_size)})")
        print(f"Non-trainable params: {non_trainable_params} ({format_memory(non_trainable_params * param_dtype_size)})")
        return
    
    
    def convert_to_list(self):
        for ctl in self.ctl_list:
            ctl()
        return
    
    
    def convert_to_shared_list(self,manager):
        for ctsl in self.ctsl_list:
            ctsl(manager)
        return
    
    
    def end(self):
        if self.end_acc!=None and self.train_acc!=None and self.train_acc>self.end_acc:
            return True
        elif self.end_loss!=None and self.train_loss!=None and self.train_loss<self.end_loss:
            return True
        elif self.end_test_acc!=None and self.test_acc!=None and self.test_acc>self.end_test_acc:
            return True
        elif self.end_test_loss!=None and self.test_loss!=None and self.test_loss<self.end_test_loss:
            return True
        elif self.end_acc!=None and self.end_test_acc!=None:
            if self.train_acc!=None and self.test_acc!=None and self.train_acc>self.end_acc and self.test_acc>self.end_test_acc:
                return True
        elif self.end_loss!=None and self.end_test_loss!=None:
            if self.train_loss!=None and self.test_loss!=None and self.train_loss<self.end_loss and self.test_loss<self.end_test_loss:
                return True
    
    
    def segment_data(self, data, labels, processes):
        data=np.array_split(data, processes)
        labels=np.array_split(labels, processes)
        return data,labels
    
    
    def parallel_test(self, test_ds, loss_object, test_loss, test_accuracy, jit_compile, p):
        for test_data, labels in test_ds:
            if jit_compile==True:
                self.test_step(test_data, labels, loss_object, test_loss, test_accuracy)
            else:
                self.test_step_(test_data, labels, loss_object, test_loss, test_accuracy)
        if test_accuracy!=None:
            self.shared_test_loss_array[p]=test_loss.result()
            self.shared_test_acc_array[p]=test_accuracy.result()
        else:
            self.shared_test_loss_array[p]=test_loss.result()
        return
    
    
    @tf.function(jit_compile=True)
    def train_step(self, train_data, labels, loss_object, train_loss, train_accuracy, optimizer):
        with tf.GradientTape() as tape:
            output = self.__call__(train_data)
            loss = loss_object(labels, output)
        if type(optimizer)!=list:
            gradients = tape.gradient(loss, self.param)
            optimizer.apply_gradients(zip(gradients, self.param))
        else:
            for i in range(len(optimizer)):
                gradients = tape.gradient(loss, self.param[i])
                optimizer[i].apply_gradients(zip(gradients, self.param[i]))
        train_loss(loss)
        if train_accuracy!=None:
            acc=train_accuracy(labels, output)
            return loss,acc
        return loss,None
      
      
    @tf.function
    def train_step_(self, train_data, labels, loss_object, train_loss, train_accuracy, optimizer):
        with tf.GradientTape() as tape:
            output = self.__call__(train_data)
            loss = loss_object(labels, output)
        if type(optimizer)!=list:
            gradients = tape.gradient(loss, self.param)
            optimizer.apply_gradients(zip(gradients, self.param))
        else:
            for i in range(len(optimizer)):
                gradients = tape.gradient(loss, self.param[i])
                optimizer[i].apply_gradients(zip(gradients, self.param[i]))
        train_loss(loss)
        if train_accuracy!=None:
            acc=train_accuracy(labels, output)
            return loss,acc
        return loss,None
        
    
    @tf.function(jit_compile=True)
    def test_step(self, test_data, labels, loss_object, test_loss, test_accuracy):
        output = self.__call__(test_data)
        loss = loss_object(labels, output)
        test_loss(loss)
        if test_accuracy!=None:
            test_accuracy(labels, output)
        return
      
      
    @tf.function
    def test_step_(self, test_data, labels, loss_object, test_loss, test_accuracy):
        output = self.__call__(test_data)
        loss = loss_object(labels, output)
        test_loss(loss)
        if test_accuracy!=None:
            test_accuracy(labels, output)
        return
    
    
    def _train_step(self, inputs, optimizer, train_accuracy):
        data, labels = inputs
    
        with tf.GradientTape() as tape:
            output = self.__call__(data)
            loss = self.compute_loss(labels, output)
        
        if type(optimizer)!=list:
            gradients = tape.gradient(loss, self.param)
            optimizer.apply_gradients(zip(gradients, self.param))
        else:
            for i in range(len(optimizer)):
                gradients = tape.gradient(loss, self.param[i])
                optimizer[i].apply_gradients(zip(gradients, self.param[i]))
        
        if train_accuracy!=None:
            acc=train_accuracy.update_state(labels, output)
            return loss,acc
        return loss,None
    
    
    def _test_step(self, inputs, loss_object, test_loss, test_accuracy):
        data, labels = inputs
    
        predictions = self.__call__(data, training=False)
        t_loss = loss_object(labels, predictions)
    
        test_loss.update_state(t_loss)
        if test_accuracy!=None:
            test_accuracy.update_state(labels, predictions)
        return
    
    
    @tf.function(jit_compile=True)
    def distributed_train_step(self, dataset_inputs, optimizer, train_accuracy, strategy):
        per_replica_losses,acc = strategy.run(self._train_step, args=(dataset_inputs, optimizer, train_accuracy))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None),acc
    
    
    @tf.function(jit_compile=True)
    def distributed_test_step(self, dataset_inputs, loss_object, test_loss, test_accuracy, strategy):
        return strategy.run(self._test_step, args=(dataset_inputs, loss_object, test_loss, test_accuracy))
    
    
    @tf.function
    def distributed_train_step_(self, dataset_inputs, optimizer, train_accuracy, strategy):
        per_replica_losses,acc = strategy.run(self._train_step, args=(dataset_inputs, optimizer, train_accuracy))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None),acc
    
    
    @tf.function
    def distributed_test_step_(self, dataset_inputs, loss_object, test_loss, test_accuracy, strategy):
        return strategy.run(self._test_step, args=(dataset_inputs, loss_object, test_loss, test_accuracy))
    
    
    def test(self, test_ds, loss_object, test_loss, test_accuracy=None, processes=None, mp=None, jit_compile=True):
        if mp==None:
            self.training()
            for test_data, labels in test_ds:
                if jit_compile==True:
                    self.test_step(test_data, labels)
                else:
                    self.test_step_(test_data, labels)
            self.training(True)
            
            test_loss=test_loss.result().numpy()
            if test_accuracy!=None:
                test_acc=test_accuracy.result().numpy()
                return test_loss,test_acc
            else:
                return test_loss
        else:
            self.training()
            self.shared_test_loss_array=mp.Array('f',np.zeros([processes],dtype='float32'))
            if test_accuracy!=None:
                self.shared_test_acc_array=mp.Array('f',np.zeros([processes],dtype='float32'))
            
            process_list=[]
            for p in range(processes):
                test_loss_=test_loss[p]
                if test_accuracy!=None:
                    test_accuracy_=test_accuracy[p]
                process=mp.Process(target=self.parallel_test,args=(test_ds[p], loss_object, test_loss_, test_accuracy_, jit_compile, p))
                process.start()
                process_list.append(process)
            for process in process_list:
                test_loss[p].reset_states()
                if test_accuracy!=None:
                    test_accuracy[p].reset_states()
                process.join()
            self.training(True)
                
            if test_accuracy!=None:
                test_loss,test_acc=np.sum(npc.as_array(self.shared_test_loss_array.get_obj()))/processes,np.sum(npc.as_array(self.shared_test_acc_array.get_obj()))/processes
                return test_loss,test_acc
            else:
                test_loss=np.sum(npc.as_array(self.shared_test_loss_array.get_obj()))/processes
                return test_loss
    
    
    def test_(self,test_ds, loss_object, test_loss, test_accuracy, processes, mp, jit_compile):
        if mp==None:
            self.training()
            if test_loss!=None:
                test_loss.reset_states()
            if test_accuracy!=None:
                test_accuracy.reset_states()
            for test_data, labels in test_ds:
                if jit_compile==True:
                    self.test_step(test_data, labels)
                else:
                    self.test_step_(test_data, labels)
                
            self.test_loss=test_loss.result().numpy()
            if test_accuracy!=None:
                self.test_acc=test_accuracy.result().numpy()
            self.training(True)
        else:
            self.training()
            if not isinstance(self.shared_test_loss_array, mp.sharedctypes.SynchronizedArray):
                self.shared_test_loss_array=mp.Array('f',np.zeros([processes],dtype='float32'))
            if test_accuracy!=None:
                if not isinstance(self.shared_test_acc_array, mp.sharedctypes.SynchronizedArray):
                    self.shared_test_acc_array=mp.Array('f',np.zeros([processes],dtype='float32'))
            
            process_list=[]
            for p in range(processes):
                test_loss_=test_loss[p]
                if test_accuracy!=None:
                    test_accuracy_=test_accuracy[p]
                process=mp.Process(target=self.parallel_test,args=(test_ds[p], loss_object, test_loss_, test_accuracy_, jit_compile, p))
                process.start()
                process_list.append(process)
            for process in process_list:
                test_loss[p].reset_states()
                if test_accuracy!=None:
                    test_accuracy[p].reset_states()
                process.join()
                
            if test_accuracy!=None:
                self.test_loss,self.test_acc=np.sum(npc.as_array(self.shared_test_loss_array.get_obj()))/processes,np.sum(npc.as_array(self.shared_test_acc_array.get_obj()))/processes
            else:
                self.test_loss=np.sum(npc.as_array(self.shared_test_loss_array.get_obj()))/processes
            self.training(True)
        return
    
    
    def train(self, train_ds, loss_object, train_loss, optimizer=None, epochs=None, train_accuracy=None, test_ds=None, test_loss=None, test_accuracy=None, processes=None, parallel_test=None, jit_compile=True, callbacks=None, p=None):
        if p==None:
            p_=9
        else:
            p_=p-1
        if epochs%10!=0:
            p=epochs-epochs%p_
            p=int(p/p_)
        else:
            p=epochs/(p_+1)
            p=int(p)
        if p==0:
            p=1
        if parallel_test==True:
            mp=multiprocessing
        else:
            mp=None
        try:
            self.batch_size=train_ds._batch_size.numpy()
        except Exception:
            self.batch_size=None
        self.loss_object=loss_object
        self.train_loss=train_loss
        if self.optimizer==None:
            self.optimizer=optimizer
        self.epochs=epochs
        self.train_accuracy=train_accuracy
        self.test_loss=test_loss
        self.test_accuracy=test_accuracy
        try:
            self.test_batch_size=test_ds._batch_size.numpy()
        except Exception:
            self.test_batch_size=None
        self.processes=processes
        self.parallel_test_=parallel_test
        self.jit_compile=jit_compile
        self.p=p
        self.info_flag=0
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(logs={})
        if epochs!=None:
            for epoch in range(epochs):
                t1=time.time()
                if self.stop_training==True:
                    return
                if self.steps_per_execution==None and self.end():
                    break
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_begin'):
                        callback.on_epoch_begin(epoch, logs={})
                train_loss.reset_states()
                if train_accuracy!=None:
                    train_accuracy.reset_states()
            
                batch = 0
                for train_data, labels in train_ds:
                    if self.stop_training==True:
                        return
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_batch_begin'):
                            callback.on_batch_begin(batch, logs={})
                    if jit_compile==True:
                        loss,acc=self.train_step(train_data, labels, loss_object, train_loss, train_accuracy, self.optimizer)
                    else:
                        loss,acc=self.train_step_(train_data, labels, loss_object, train_loss, train_accuracy, self.optimizer)
                    batch_logs = {'loss': loss.numpy()}
                    if train_accuracy != None:
                        batch_logs['accuracy'] = acc.numpy()
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_batch_end'):
                            callback.on_batch_end(batch, logs=batch_logs)
                    self.batch_counter+=1
                    batch += 1
                    if self.steps_per_execution!=None and self.batch_counter%self.steps_per_execution==0:
                        self.train_loss=train_loss.result().numpy()
                        if train_accuracy!=None:
                            self.train_acc=train_accuracy.result().numpy()
                        if test_ds!=None:
                            self.test_(test_ds, loss_object, test_loss, test_accuracy, processes, mp, jit_compile)
                        if self.end():
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    if self.save_freq_!=None and self.path!=None and self.batch_counter%self.save_freq_==0:
                        if self.save_param_only==False:
                            self.save_(self.path)
                        else:
                            self.save_param_(self.path)
                
                if test_ds!=None:
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_test_begin'):
                            callback.on_test_begin(epoch, logs={})
                    self.test_(test_ds, loss_object, test_loss, test_accuracy, processes, mp, jit_compile)
                self.test_loss_list.append(self.test_loss)
                if test_accuracy!=None:
                    self.test_acc_list.append(self.test_acc)
                
                self.train_loss=train_loss.result().numpy()
                self.train_loss_list.append(self.train_loss)
                if train_accuracy!=None:
                    self.train_acc=train_accuracy.result().numpy()
                    self.train_acc_list.append(self.train_acc)
                    
                epoch_logs = {'loss': self.train_loss}
                if train_accuracy != None:
                    epoch_logs['accuracy'] = self.train_acc
                if self.test_loss != None:
                    epoch_logs['val_loss'] = self.test_loss
                if test_accuracy != None:
                    epoch_logs['val_accuracy'] = self.test_acc
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(epoch, logs=epoch_logs)
                for callback in self.callbacks:
                    if hasattr(callback, 'on_test_end'):
                        callback.on_test_end(epoch, logs=epoch_logs)
                self.total_epoch+=1   
                if epoch%p==0:
                    if self.test_ds==None:
                        if train_accuracy!=None:
                            print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                            print('epoch:{0}   accuracy:{1:.4f}'.format(epoch+1, self.train_acc))
                            print()
                        else:
                            print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                            print()
                    else:
                        if test_accuracy!=None:
                            print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                            print('epoch:{0}   accuracy:{1:.4f},test accuracy:{2:.4f}'.format(epoch+1,self.train_acc,self.test_acc))
                            print()
                        else:
                            print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                            print()
                if self.save_freq_==None:
                    if self.path!=None and epoch%self.save_freq==0:
                        if self.save_param_only==False:
                            self.save_(self.path)
                        else:
                            self.save_param_(self.path)
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                if self.stop_training==True:
                    return
                if self.steps_per_execution==None and self.end():
                    break
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_begin'):
                        callback.on_epoch_begin(i, logs={})
                train_loss.reset_states()
                if train_accuracy!=None:
                    train_accuracy.reset_states()
            
                batch = 0
                for train_data, labels in train_ds:
                    if self.stop_training==True:
                        return
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_batch_begin'):
                            callback.on_batch_begin(batch, logs={})
                    if jit_compile==True:
                        loss,acc=self.train_step(train_data, labels, loss_object, train_loss, train_accuracy, self.optimizer)
                    else:
                        loss,acc=self.train_step_(train_data, labels, loss_object, train_loss, train_accuracy, self.optimizer)
                    batch_logs = {'loss': loss.numpy()}
                    if train_accuracy != None:
                        batch_logs['accuracy'] = acc.numpy()
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_batch_end'):
                            callback.on_batch_end(batch, logs=batch_logs)
                    self.batch_counter+=1
                    batch += 1
                    if self.steps_per_execution!=None and self.batch_counter%self.steps_per_execution==0:
                        self.train_loss=train_loss.result().numpy()
                        if train_accuracy!=None:
                            self.train_acc=train_accuracy.result().numpy()
                        if test_ds!=None:
                            self.test_(test_ds, loss_object, test_loss, test_accuracy, processes, mp, jit_compile)
                        if self.end():
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    if self.save_freq_!=None and self.path!=None and self.batch_counter%self.save_freq_==0:
                        if self.save_param_only==False:
                            self.save_(self.path)
                        else:
                            self.save_param_(self.path)
                
                if test_ds!=None:
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_test_begin'):
                            callback.on_test_begin(i, logs={})
                    self.test_(test_ds, loss_object, test_loss, test_accuracy, processes, mp, jit_compile)
                self.test_loss_list.append(self.test_loss)
                if test_accuracy!=None:
                    self.test_acc_list.append(self.test_acc)
            
                self.train_loss=train_loss.result().numpy()
                self.train_loss_list.append(self.train_loss)
                if train_accuracy!=None:
                    self.train_acc=train_accuracy.result().numpy()
                    self.train_acc_list.append(self.train_acc)
                
                epoch_logs = {'loss': self.train_loss}
                if train_accuracy != None:
                    epoch_logs['accuracy'] = self.train_acc
                if self.test_loss != None:
                    epoch_logs['val_loss'] = self.test_loss
                if test_accuracy != None:
                    epoch_logs['val_accuracy'] = self.test_acc
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(i, logs=epoch_logs)
                for callback in self.callbacks:
                    if hasattr(callback, 'on_test_end'):
                        callback.on_test_end(i, logs=epoch_logs)
                i+=1
                self.total_epoch+=1
                if i%p==0:
                    if self.test_ds==None:
                        if train_accuracy!=None:
                            print('epoch:{0}   loss:{1:.4f}'.format(i+1, self.train_loss))
                            print('epoch:{0}   accuracy:{1:.4f}'.format(i+1, self.train_acc))
                            print()
                        else:
                            print('epoch:{0}   loss:{1:.4f}'.format(i+1, self.train_loss))
                            print()
                    else:
                        if test_accuracy!=None:
                            print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(i+1,self.train_loss,self.test_loss))
                            print('epoch:{0}   accuracy:{1:.4f},test accuracy:{2:.4f}'.format(i+1,self.train_acc,self.test_acc))
                            print()
                        else:
                            print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(i+1,self.train_loss,self.test_loss))
                            print()
                if self.save_freq_==None:
                    if self.path!=None and i%self.save_freq==0:
                        if self.save_param_only==False:
                            self.save_(self.path)
                        else:
                            self.save_param_(self.path)
                t2=time.time()
                self.time+=(t2-t1)
        self.shared_test_loss_array=None
        self.shared_test_acc_array=None
        self._time=self.time-int(self.time)
        if self._time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        print('time:{0}s'.format(self.time))
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(logs={})
        return
    
    
    def distributed_training(self, train_dataset=None, loss_object=None, global_batch_size=None, optimizer=None, strategy=None, epochs=None, num_epochs=None, num_steps_per_epoch=None, train_accuracy=None, test_dataset=None, test_loss=None, test_accuracy=None, dataset_fn=None, test_dataset_fn=None, global_test_batch_size=None, eval_steps_per_epoch=None, jit_compile=True, callbacks=None, p=None):
        if num_epochs!=None:
            epochs=num_epochs
        if p==None:
            p_=9
        else:
            p_=p-1
        if epochs%10!=0:
            p=epochs-epochs%p_
            p=int(p/p_)
        else:
            p=epochs/(p_+1)
            p=int(p)
        if p==0:
            p=1
        self.loss_object=loss_object
        self.global_batch_size=global_batch_size
        if self.optimizer==None:
            self.optimizer=optimizer
        self.strategy=strategy
        self.epochs=epochs
        self.num_epochs=num_epochs
        self.num_steps_per_epoch=num_steps_per_epoch
        self.train_accuracy=train_accuracy
        self.test_loss=test_loss
        self.test_accuracy=test_accuracy
        self.global_test_batch_size=global_test_batch_size
        self.eval_steps_per_epoch=eval_steps_per_epoch
        self.jit_compile=jit_compile
        self.p=p
        self.info_flag=1
        with strategy.scope():
            def compute_loss(self, labels, output):
                per_example_loss = loss_object(labels, output)
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(logs={})
        if isinstance(strategy,tf.distribute.MirroredStrategy):
            train_dist_dataset=strategy.experimental_distribute_dataset(train_dataset)
            if test_dataset!=None:
                test_dist_dataset=strategy.experimental_distribute_dataset(test_dataset)    
            if epochs!=None:
                for epoch in range(epochs):
                    t1=time.time()
                    if self.stop_training==True:
                        return
                    if self.steps_per_execution==None and self.end():
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_begin'):
                            callback.on_epoch_begin(epoch, logs={})
                    if train_accuracy!=None:
                        train_accuracy.reset_states()
                    if test_loss!=None:
                        test_loss.reset_states()
                    if test_accuracy!=None:
                        test_accuracy.reset_states()
                
                    total_loss = 0.0
                    num_batches = 0
                    batch = 0
                    for x in train_dist_dataset:
                        if self.stop_training==True:
                            return
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_batch_begin'):
                                callback.on_batch_begin(batch, logs={})
                        if jit_compile==True:
                            loss,acc = self.distributed_train_step(x, self.optimizer, train_accuracy, strategy)
                        else:
                            loss,acc = self.distributed_train_step_(x, self.optimizer, train_accuracy, strategy)
                        total_loss += loss
                        
                        batch_logs = {'loss': loss.numpy()}
                        if train_accuracy != None:
                            batch_logs['accuracy'] = acc.numpy()
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_batch_end'):
                                callback.on_batch_end(batch, logs=batch_logs)
                        num_batches += 1
                        self.batch_counter+=1
                        if self.steps_per_execution!=None and self.batch_counter%self.steps_per_execution==0:
                            self.train_loss=(total_loss / num_batches).numpy()
                            if train_accuracy!=None:
                                self.train_acc=train_accuracy.result().numpy()
                            if test_dist_dataset!=None:
                                self.training()
                                for x in test_dist_dataset:
                                    if jit_compile==True:
                                        self.distributed_test_step(x, loss_object, test_loss, test_accuracy, strategy)
                                    else:
                                        self.distributed_test_step_(x, loss_object, test_loss, test_accuracy, strategy)
                                    
                                self.test_loss=test_loss.result().numpy()
                                if test_accuracy!=None:
                                    self.test_acc=test_accuracy.result().numpy()
                                self.training(True)
                            if self.end():
                                if self.save_param_only==False:
                                    self.save_(self.path)
                                else:
                                    self.save_param_(self.path)
                        if self.save_freq_!=None and self.path!=None and self.batch_counter%self.save_freq_==0:
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    
                    if test_loss!=None:
                        test_loss.reset_states()
                    if test_accuracy!=None:
                        test_accuracy.reset_states()
                    if test_dist_dataset!=None:
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_test_begin'):
                                callback.on_test_begin(epoch, logs={})
                        self.training()
                        for x in test_dist_dataset:
                            if jit_compile==True:
                                self.distributed_test_step(x, loss_object, test_loss, test_accuracy, strategy)
                            else:
                                self.distributed_test_step_(x, loss_object, test_loss, test_accuracy, strategy)
                            
                        self.test_loss=test_loss.result().numpy()
                        self.test_loss_list.append(self.test_loss)
                        if test_accuracy!=None:
                            self.test_acc=test_accuracy.result().numpy()
                            self.test_acc_list.append(self.test_acc)
                        self.training(True)
                    
                    self.train_loss=(total_loss / num_batches).numpy()
                    self.train_loss_list.append(self.train_loss)
                    if train_accuracy!=None:
                        self.train_acc=train_accuracy.result().numpy()
                        self.train_acc_list.append(self.train_acc)
                    
                    epoch_logs = {'loss': self.train_loss}
                    if train_accuracy != None:
                        epoch_logs['accuracy'] = self.train_acc
                    if self.test_loss != None:
                        epoch_logs['val_loss'] = self.test_loss
                    if test_accuracy != None:
                        epoch_logs['val_accuracy'] = self.test_acc
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_end'):
                            callback.on_epoch_end(epoch, logs=epoch_logs)
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_test_end'):
                            callback.on_test_end(epoch, logs=epoch_logs)
                    self.total_epoch+=1     
                    if epoch%p==0:
                        if self.test_ds==None:
                            if train_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                print('epoch:{0}   accuracy:{1:.4f}'.format(epoch+1, self.train_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                print()
                        else:
                            if test_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                print('epoch:{0}   accuracy:{1:.4f},test accuracy:{2:.4f}'.format(epoch+1,self.train_acc,self.test_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                print()
                    if self.save_freq_==None:
                        if self.path!=None and epoch%self.save_freq==0:
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    t2=time.time()
                    self.time+=(t2-t1)
            else:
                i=0
                while True:
                    t1=time.time()
                    if self.stop_training==True:
                        return
                    if self.steps_per_execution==None and self.end():
                        break
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_begin'):
                            callback.on_epoch_begin(epoch, logs={})
                    if train_accuracy!=None:
                        train_accuracy.reset_states()
                    if test_loss!=None:
                        test_loss.reset_states()
                    if test_accuracy!=None:
                        test_accuracy.reset_states()
                
                    total_loss = 0.0
                    num_batches = 0
                    batch = 0
                    for x in train_dist_dataset:
                        if self.stop_training==True:
                            return
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_batch_begin'):
                                callback.on_batch_begin(batch, logs={})
                        if jit_compile==True:
                            loss,acc = self.distributed_train_step(x, self.optimizer, train_accuracy, strategy)
                        else:
                            loss,acc = self.distributed_train_step_(x, self.optimizer, train_accuracy, strategy)
                        total_loss += loss
                        
                        batch_logs = {'loss': loss.numpy()}
                        if train_accuracy != None:
                            batch_logs['accuracy'] = acc.numpy()
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_batch_end'):
                                callback.on_batch_end(batch, logs=batch_logs)
                        num_batches += 1
                        self.batch_counter+=1
                        batch +=1
                        if self.steps_per_execution!=None and self.batch_counter%self.steps_per_execution==0:
                            self.train_loss=(total_loss / num_batches).numpy()
                            if train_accuracy!=None:
                                self.train_acc=train_accuracy.result().numpy()
                            if test_dist_dataset!=None:
                                self.training()
                                for x in test_dist_dataset:
                                    if jit_compile==True:
                                        self.distributed_test_step(x, loss_object, test_loss, test_accuracy, strategy)
                                    else:
                                        self.distributed_test_step_(x, loss_object, test_loss, test_accuracy, strategy)
                                    
                                self.test_loss=test_loss.result().numpy()
                                if test_accuracy!=None:
                                    self.test_acc=test_accuracy.result().numpy()
                                self.training(True)
                            if self.end():
                                if self.save_param_only==False:
                                    self.save_(self.path)
                                else:
                                    self.save_param_(self.path)
                        if self.save_freq_!=None and self.path!=None and self.batch_counter%self.save_freq_==0:
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    
                    if test_loss!=None:
                        test_loss.reset_states()
                    if test_accuracy!=None:
                        test_accuracy.reset_states()
                    if test_dist_dataset!=None:
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_test_begin'):
                                callback.on_test_begin(epoch, logs={})
                        self.training()
                        for x in test_dist_dataset:
                            if jit_compile==True:
                                self.distributed_test_step(x, loss_object, test_loss, test_accuracy, strategy)
                            else:
                                self.distributed_test_step_(x, loss_object, test_loss, test_accuracy, strategy)
                            
                        self.test_loss=test_loss.result().numpy()
                        self.test_loss_list.append(self.test_loss)
                        if test_accuracy!=None:
                            self.test_acc=test_accuracy.result().numpy()
                            self.test_acc_list.append(self.test_acc)
                        self.training(True)
                
                    self.train_loss=(total_loss / num_batches).numpy()
                    self.train_loss_list.append(self.train_loss)
                    if train_accuracy!=None:
                        self.train_acc=train_accuracy.result().numpy()
                        self.train_acc_list.append(self.train_acc)
                    
                    epoch_logs = {'loss': self.train_loss}
                    if train_accuracy != None:
                        epoch_logs['accuracy'] = self.train_acc
                    if self.test_loss != None:
                        epoch_logs['val_loss'] = self.test_loss
                    if test_accuracy != None:
                        epoch_logs['val_accuracy'] = self.test_acc
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_end'):
                            callback.on_epoch_end(i, logs=epoch_logs)
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_test_end'):
                            callback.on_test_end(i, logs=epoch_logs)
                    i+=1
                    self.total_epoch+=1
                    if i%p==0:
                        if self.test_ds==None:
                            if train_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f}'.format(i+1, self.train_loss))
                                print('epoch:{0}   accuracy:{1:.4f}'.format(i+1, self.train_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f}'.format(i+1, self.train_loss))
                                print()
                        else:
                            if test_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(i+1,self.train_loss,self.test_loss))
                                print('epoch:{0}   accuracy:{1:.4f},test accuracy:{2:.4f}'.format(i+1,self.train_acc,self.test_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(i+1,self.train_loss,self.test_loss))
                                print()
                    if self.save_freq_==None:
                        if self.path!=None and i%self.save_freq==0:
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    t2=time.time()
                    self.time+=(t2-t1)
        elif isinstance(strategy,tf.distribute.MultiWorkerMirroredStrategy):
            if num_epochs!=None:
                epoch = 0
                self.step_in_epoch = 0
                with strategy.scope():
                    multi_worker_dataset = strategy.distribute_datasets_from_function(
                            lambda input_context: self.dataset_fn(train_dataset, global_batch_size, input_context))
                if test_dataset!=None:
                    with strategy.scope():
                        multi_worker_test_dataset = strategy.distribute_datasets_from_function(
                                lambda input_context: self.dataset_fn(test_dataset, global_test_batch_size, input_context))
                while epoch < num_epochs:
                    t1=time.time()
                    
                    if self.stop_training==True:
                        return
                    
                    if self.steps_per_execution==None and self.end():
                        break
                    
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_begin'):
                            callback.on_epoch_begin(epoch, logs={})
                    
                    train_loss=self.CTL(multi_worker_dataset, num_steps_per_epoch, train_accuracy, strategy, jit_compile)
                    if test_dataset!=None:
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_test_begin'):
                                callback.on_test_begin(epoch, logs={})
                        self.training()
                        iterator = iter(multi_worker_test_dataset)
                        for _ in math.ceil(len(test_dataset)/global_test_batch_size):
                            if jit_compile==True:
                                self.distributed_test_step(next(iterator), loss_object, test_loss, test_accuracy, strategy)
                            else:
                                self.distributed_test_step_(next(iterator), loss_object, test_loss, test_accuracy, strategy)
                        self.test_loss=test_loss.result().numpy()
                        self.test_loss_list.append(self.test_loss)
                        if test_accuracy!=None:
                            self.test_acc=test_accuracy.result().numpy()
                            self.test_acc_list.append(self.test_acc)
                        self.training(True)
                    
                    self.train_loss=train_loss.numpy()
                    self.train_loss_list.append(self.train_loss)
                    self.train_acc=train_accuracy.result().numpy()
                    self.train_acc_list.append(self.train_acc)
                        
                    epoch_logs = {'loss': self.train_loss}
                    if train_accuracy != None:
                        epoch_logs['accuracy'] = self.train_acc
                    if self.test_loss != None:
                        epoch_logs['val_loss'] = self.test_loss
                    if test_accuracy != None:
                        epoch_logs['val_accuracy'] = self.test_acc
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_end'):
                            callback.on_epoch_end(epoch, logs=epoch_logs)
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_test_end'):
                            callback.on_test_end(epoch, logs=epoch_logs)
                    self.total_epoch+=1     
                    if epoch%p==0:
                        if self.test_ds==None:
                            if train_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                print('epoch:{0}   accuracy:{1:.4f}'.format(epoch+1, self.train_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                print()
                        else:
                            if test_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                print('epoch:{0}   accuracy:{1:.4f},test accuracy:{2:.4f}'.format(epoch+1,self.train_acc,self.test_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                print()
                    if self.save_freq_==None:
                        if self.path!=None and epoch%self.save_freq==0:
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    
                    if train_accuracy!=None:
                        train_accuracy.reset_states()
                    if test_loss!=None:
                        test_loss.reset_states()
                    if test_accuracy!=None:
                        test_accuracy.reset_states()
                    epoch += 1
                    self.step_in_epoch = 0
                                
                    t2=time.time()
                    self.time+=(t2-t1)
            else:
                epoch = 0
                self.step_in_epoch = 0
                with strategy.scope():
                    multi_worker_dataset = strategy.distribute_datasets_from_function(
                            lambda input_context: self.dataset_fn(train_dataset, global_batch_size, input_context))
                if test_dataset!=None:
                    with strategy.scope():
                        multi_worker_test_dataset = strategy.distribute_datasets_from_function(
                                lambda input_context: self.dataset_fn(test_dataset, global_test_batch_size, input_context))
                while True:
                    t1=time.time()
                    
                    if self.stop_training==True:
                        return
                    
                    if self.steps_per_execution==None and self.end():
                        break
                    
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_begin'):
                            callback.on_epoch_begin(epoch, logs={})
                    
                    train_loss=self.CTL(multi_worker_dataset, num_steps_per_epoch, train_accuracy, strategy, jit_compile)
                    if test_dataset!=None:
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_test_begin'):
                                callback.on_test_begin(epoch, logs={})
                        self.training()
                        iterator = iter(multi_worker_test_dataset)
                        for _ in math.ceil(len(test_dataset)/global_test_batch_size):
                            if jit_compile==True:
                                self.distributed_test_step(next(iterator), loss_object, test_loss, test_accuracy, strategy)
                            else:
                                self.distributed_test_step_(next(iterator), loss_object, test_loss, test_accuracy, strategy)
                        self.test_loss=test_loss.result().numpy()
                        self.test_loss_list.append(self.test_loss)
                        if test_accuracy!=None:
                            self.test_acc=test_accuracy.result().numpy()
                            self.test_acc_list.append(self.test_acc)
                        self.training(True)
                    
                    self.train_loss=train_loss.numpy()
                    self.train_loss_list.append(self.train_loss)
                    self.train_acc=train_accuracy.result().numpy()
                    self.train_acc_list.append(self.train_acc)
                    
                    epoch_logs = {'loss': self.train_loss}
                    if train_accuracy != None:
                        epoch_logs['accuracy'] = self.train_acc
                    if self.test_loss != None:
                        epoch_logs['val_loss'] = self.test_loss
                    if test_accuracy != None:
                        epoch_logs['val_accuracy'] = self.test_acc
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_end'):
                            callback.on_epoch_end(epoch, logs=epoch_logs)
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_test_end'):
                            callback.on_test_end(epoch, logs=epoch_logs)
                    self.total_epoch+=1     
                    if epoch%p==0:
                        if self.test_ds==None:
                            if train_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                print('epoch:{0}   accuracy:{1:.4f}'.format(epoch+1, self.train_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                print()
                        else:
                            if test_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                print('epoch:{0}   accuracy:{1:.4f},test accuracy:{2:.4f}'.format(epoch+1,self.train_acc,self.test_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                print()
                    if self.save_freq_==None:
                        if self.path!=None and epoch%self.save_freq==0:
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    
                    if train_accuracy!=None:
                        train_accuracy.reset_states()
                    if test_loss!=None:
                        test_loss.reset_states()
                    if test_accuracy!=None:
                        test_accuracy.reset_states()
                    epoch += 1
                    self.step_in_epoch = 0
                                
                    t2=time.time()
                    self.time+=(t2-t1)
        elif isinstance(strategy,tf.distribute.ParameterServerStrategy):
            coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)
            self.dataset_fn = dataset_fn
            self.test_dataset_fn = test_dataset_fn
            self.strategy = strategy
            if num_epochs!=None:
                epoch = 0
                self.step_in_epoch = 0
                while epoch < num_epochs:
                    t1=time.time()
                    
                    if self.stop_training==True:
                        return
                    
                    if self.steps_per_execution==None and self.end():
                        break
                    
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_begin'):
                            callback.on_epoch_begin(epoch, logs={})
                    
                    train_loss=self.CTL_param(coordinator, num_steps_per_epoch, train_accuracy, strategy, jit_compile)
                    if eval_steps_per_epoch!=None:
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_test_begin'):
                                callback.on_test_begin(epoch, logs={})
                        self.training()
                        if jit_compile==True:
                            per_worker_dataset = coordinator.create_per_worker_dataset(self.per_worker_test_dataset_fn)
                        else:
                            per_worker_dataset = coordinator.create_per_worker_dataset(self.per_worker_test_dataset_fn_)
                        per_worker_iterator = iter(per_worker_dataset)
                        for _ in eval_steps_per_epoch:
                            if jit_compile==True:
                                coordinator.schedule(self.distributed_test_step, args=(next(per_worker_iterator), loss_object, test_loss, test_accuracy, strategy))
                            else:
                                coordinator.schedule(self.distributed_test_step_, args=(next(per_worker_iterator), loss_object, test_loss, test_accuracy, strategy))
                        coordinator.join()
                        self.test_loss=test_loss.result().numpy()
                        self.test_loss_list.append(self.test_loss)
                        if test_accuracy!=None:
                            self.test_acc=test_accuracy.result().numpy()
                            self.test_acc_list.append(self.test_acc)
                        self.training(True)
                    
                    self.train_loss=train_loss
                    self.train_loss_list.append(self.train_loss)
                    self.train_acc=train_accuracy.result().numpy()
                    self.train_acc_list.append(self.train_acc)
                        
                    epoch_logs = {'loss': self.train_loss}
                    if train_accuracy != None:
                        epoch_logs['accuracy'] = self.train_acc
                    if self.test_loss != None:
                        epoch_logs['val_loss'] = self.test_loss
                    if test_accuracy != None:
                        epoch_logs['val_accuracy'] = self.test_acc
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_epoch_end'):
                            callback.on_epoch_end(epoch, logs=epoch_logs)
                    for callback in self.callbacks:
                        if hasattr(callback, 'on_test_end'):
                            callback.on_test_end(epoch, logs=epoch_logs)
                    self.total_epoch+=1     
                    if epoch%p==0:
                        if self.test_ds==None:
                            if train_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                print('epoch:{0}   accuracy:{1:.4f}'.format(epoch+1, self.train_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                print()
                        else:
                            if test_accuracy!=None:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                print('epoch:{0}   accuracy:{1:.4f},test accuracy:{2:.4f}'.format(epoch+1,self.train_acc,self.test_acc))
                                print()
                            else:
                                print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                print()
                    if self.save_freq_==None:
                        if self.path!=None and epoch%self.save_freq==0:
                            if self.save_param_only==False:
                                self.save_(self.path)
                            else:
                                self.save_param_(self.path)
                    
                    if train_accuracy!=None:
                        train_accuracy.reset_states()
                    if test_loss!=None:
                        test_loss.reset_states()
                    if test_accuracy!=None:
                        test_accuracy.reset_states()
                    epoch += 1
                    self.step_in_epoch = 0
                                
                    t2=time.time()
                    self.time+=(t2-t1)
            else:
                coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)
                self.dataset_fn = dataset_fn
                self.test_dataset_fn = test_dataset_fn
                self.strategy = strategy
                if num_epochs!=None:
                    epoch = 0
                    self.step_in_epoch = 0
                    while True:
                        t1=time.time()
                        
                        if self.stop_training==True:
                            return
                        
                        if self.steps_per_execution==None and self.end():
                            break
                        
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_epoch_begin'):
                                callback.on_epoch_begin(epoch, logs={})
                        
                        train_loss=self.CTL_param(coordinator, num_steps_per_epoch, train_accuracy, strategy, jit_compile)
                        if eval_steps_per_epoch!=None:
                            for callback in self.callbacks:
                                if hasattr(callback, 'on_test_begin'):
                                    callback.on_test_begin(epoch, logs={})
                            self.training()
                            if jit_compile==True:
                                per_worker_dataset = coordinator.create_per_worker_dataset(self.per_worker_test_dataset_fn)
                            else:
                                per_worker_dataset = coordinator.create_per_worker_dataset(self.per_worker_test_dataset_fn_)
                            per_worker_iterator = iter(per_worker_dataset)
                            for _ in eval_steps_per_epoch:
                                if jit_compile==True:
                                    coordinator.schedule(self.distributed_test_step, args=(next(per_worker_iterator), loss_object, test_loss, test_accuracy, strategy))
                                else:
                                    coordinator.schedule(self.distributed_test_step_, args=(next(per_worker_iterator), loss_object, test_loss, test_accuracy, strategy))
                            coordinator.join()
                            self.test_loss=test_loss.result().numpy()
                            self.test_loss_list.append(self.test_loss)
                            if test_accuracy!=None:
                                self.test_acc=test_accuracy.result().numpy()
                                self.test_acc_list.append(self.test_acc)
                            self.training(True)
                        
                        self.train_loss=train_loss
                        self.train_loss_list.append(self.train_loss)
                        self.train_acc=train_accuracy.result().numpy()
                        self.train_acc_list.append(self.train_acc)
                            
                        epoch_logs = {'loss': self.train_loss}
                        if train_accuracy != None:
                            epoch_logs['accuracy'] = self.train_acc
                        if self.test_loss != None:
                            epoch_logs['val_loss'] = self.test_loss
                        if test_accuracy != None:
                            epoch_logs['val_accuracy'] = self.test_acc
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_epoch_end'):
                                callback.on_epoch_end(epoch, logs=epoch_logs)
                        for callback in self.callbacks:
                            if hasattr(callback, 'on_test_end'):
                                callback.on_test_end(epoch, logs=epoch_logs)
                        self.total_epoch+=1     
                        if epoch%p==0:
                            if self.test_ds==None:
                                if train_accuracy!=None:
                                    print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                    print('epoch:{0}   accuracy:{1:.4f}'.format(epoch+1, self.train_acc))
                                    print()
                                else:
                                    print('epoch:{0}   loss:{1:.4f}'.format(epoch+1, self.train_loss))
                                    print()
                            else:
                                if test_accuracy!=None:
                                    print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                    print('epoch:{0}   accuracy:{1:.4f},test accuracy:{2:.4f}'.format(epoch+1,self.train_acc,self.test_acc))
                                    print()
                                else:
                                    print('epoch:{0}   loss:{1:.4f},test loss:{2:.4f}'.format(epoch+1,self.train_loss,self.test_loss))
                                    print()
                        if self.save_freq_==None:
                            if self.path!=None and epoch%self.save_freq==0:
                                if self.save_param_only==False:
                                    self.save_(self.path)
                                else:
                                    self.save_param_(self.path)
                        
                        if train_accuracy!=None:
                            train_accuracy.reset_states()
                        if test_loss!=None:
                            test_loss.reset_states()
                        if test_accuracy!=None:
                            test_accuracy.reset_states()
                        epoch += 1
                        self.step_in_epoch = 0
                                    
                        t2=time.time()
                        self.time+=(t2-t1)
        self._time=self.time-int(self.time)
        if self._time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        print('time:{0}s'.format(self.time))
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(logs={})
        return
    
    
    def dataset_fn(self, dataset, global_batch_size, input_context):
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        dataset = dataset.shard(input_context.num_input_pipelines,
                                input_context.input_pipeline_id)
        dataset = dataset.batch(batch_size)
        return dataset
    
    
    def CTL(self, multi_worker_dataset, num_steps_per_epoch, train_accuracy, strategy, jit_compile):
        iterator = iter(multi_worker_dataset)
        total_loss = 0.0
        num_batches = 0
        batch = 0
        
        while self.step_in_epoch < num_steps_per_epoch:
            for callback in self.callbacks:
                if hasattr(callback, 'on_batch_begin'):
                    callback.on_batch_begin(batch, logs={})
            if jit_compile==True:
                loss,acc = self.distributed_train_step(next(iterator), self.optimizer, train_accuracy, strategy)
            else:
                loss,acc = self.distributed_train_step_(next(iterator), self.optimizer, train_accuracy, strategy)
            total_loss += loss
            batch_logs = {'loss': loss.numpy()}
            if train_accuracy != None:
                batch_logs['accuracy'] = acc.numpy()
            for callback in self.callbacks:
                if hasattr(callback, 'on_batch_end'):
                    callback.on_batch_end(batch, logs=batch_logs)
            num_batches += 1
            self.step_in_epoch += 1
            self.batch_counter += 1
            batch += 1
            if self.steps_per_execution!=None and self.batch_counter%self.steps_per_execution==0:
                self.train_loss = total_loss / num_batches
                if self.end():
                    if self.save_param_only==False:
                        self.save_(self.path)
                    else:
                        self.save_param_(self.path)
            if self.save_freq_!=None and self.path!=None and self.batch_counter%self.save_freq_==0:
                if self.save_param_only==False:
                    self.save_(self.path)
                else:
                    self.save_param_(self.path)
            if self.stop_training==True:
                return total_loss / num_batches
      
        train_loss = total_loss / num_batches
        return train_loss
    
    
    @tf.function(jit_compile=True)
    def per_worker_dataset_fn(self):
        return self.strategy.distribute_datasets_from_function(self.dataset_fn)
  
    
    @tf.function
    def per_worker_dataset_fn_(self):
        return self.strategy.distribute_datasets_from_function(self.dataset_fn)
    
    
    @tf.function(jit_compile=True)
    def per_worker_test_dataset_fn(self):
        return self.strategy.distribute_datasets_from_function(self.test_dataset_fn)
  
    
    @tf.function
    def per_worker_test_dataset_fn_(self):
        return self.strategy.distribute_datasets_from_function(self.test_dataset_fn)
    
    
    def CTL_param(self, coordinator, num_steps_per_epoch, train_accuracy, strategy, jit_compile):
        if jit_compile==True:
            per_worker_dataset = coordinator.create_per_worker_dataset(self.per_worker_dataset_fn)
        else:
            per_worker_dataset = coordinator.create_per_worker_dataset(self.per_worker_dataset_fn_)
        per_worker_iterator = iter(per_worker_dataset)
        total_loss = 0.0
        num_batches = 0
        batch = 0
        
        while self.step_in_epoch < num_steps_per_epoch:
            for callback in self.callbacks:
                if hasattr(callback, 'on_batch_begin'):
                    callback.on_batch_begin(batch, logs={})
            if jit_compile==True:
                loss,acc = coordinator.schedule(self.distributed_train_step, args=(next(per_worker_iterator), self.optimizer, train_accuracy, strategy))
            else:
                loss,acc = coordinator.schedule(self.distributed_train_step_, args=(next(per_worker_iterator), self.optimizer, train_accuracy, strategy))
            total_loss += loss
            batch_logs = {'loss': loss.fetch()}
            if train_accuracy != None:
                batch_logs['accuracy'] = acc.numpy()
            for callback in self.callbacks:
                if hasattr(callback, 'on_batch_end'):
                    callback.on_batch_end(batch, logs=batch_logs)
            num_batches += 1
            self.step_in_epoch += 1
            self.batch_counter += 1
            batch += 1
            if self.steps_per_execution!=None and self.batch_counter%self.steps_per_execution==0:
                self.train_loss=total_loss.fetch() / num_batches
                if self.end():
                    if self.save_param_only==False:
                        self.save_(self.path)
                    else:
                        self.save_param_(self.path)
            if self.save_freq_!=None and self.path!=None and self.batch_counter%self.save_freq_==0:
                if self.save_param_only==False:
                    self.save_(self.path)
                else:
                    self.save_param_(self.path)
            if self.stop_training==True:
                coordinator.join()
                return total_loss.fetch() / num_batches
        coordinator.join()
      
        train_loss = total_loss.fetch() / num_batches
        return train_loss
        
    
    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_epoch+1),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
        print('train loss:{0:.4f}'.format(self.train_loss))
        if self.train_acc!=None:
            plt.figure(2)
            plt.plot(np.arange(1,self.total_epoch+1),self.train_acc_list)
            plt.title('train acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.xticks(np.arange(1,self.total_epoch+1))
            plt.show()
            print('train acc:{0:.4f}'.format(self.train_acc)) 
        return
    
    
    def visualize_test(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_epoch+1),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
        print('test loss:{0:.4f}'.format(self.test_loss))
        if self.test_acc!=None:
            plt.figure(2)
            plt.plot(np.arange(1,self.total_epoch+1),self.test_acc_list)
            plt.title('test acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.xticks(np.arange(1,self.total_epoch+1))
            plt.show()
            print('test acc:{0:.4f}'.format(self.test_acc))  
        return 
    
    
    def visualize_comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_epoch+1),self.train_loss_list,'b-',label='train loss')
        if self.test_loss!=None:
            plt.plot(np.arange(1,self.total_epoch+1),self.test_loss_list,'r-',label='test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
        print('train loss:{0:.4f}'.format(self.train_loss))
        if self.train_acc!=None:
            plt.figure(2)
            plt.plot(np.arange(1,self.total_epoch+1),self.train_acc_list,'b-',label='train acc')
            if self.test_acc!=None:
                plt.plot(np.arange(1,self.total_epoch+1),self.test_acc_list,'r-',label='test acc')
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.xticks(np.arange(1,self.total_epoch+1))
            plt.show()
            print('train acc:{0:.4f}'.format(self.train_acc))
        if self.test_loss!=None:   
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.4f}'.format(self.test_loss))
            if self.test_acc!=None:
                print('test acc:{0:.4f}'.format(self.test_acc)) 
        return
    
    
    def save_param_(self,path):
        if self.save_best_only==False:
            if self.max_save_files==None or self.max_save_files==1:
                output_file=open(path,'wb')
            else:
                if self.train_acc!=None and self.test_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}-{2:.4f}.dat'.format(self.total_epoch,self.train_acc,self.test_acc))
                elif self.train_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch,self.train_acc))
                else:
                    path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_epoch))
                output_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self.param,output_file)
            output_file.close()
        else:
            if self.monitor=='val_loss':
                if self.test_loss<self.val_loss:
                    self.val_loss=self.test_loss
                    self.save_param(path)
                if self.val_loss==0:
                    self.val_loss=self.test_loss
            elif self.monitor=='val_accuracy':
                if self.test_acc>self.val_accuracy:
                    self.val_accuracy=self.test_acc
                    self.save_param(path)
                if self.val_accuracy==1:
                    self.val_accuracy=self.test_acc
        return
    
    
    def save_param(self,path):
        output_file=open(path,'wb')
        pickle.dump(self.param,output_file)
        output_file.close()
        return
    
    
    def restore_param(self,path):
        input_file=open(path,'rb')
        param=pickle.load(input_file)
        nn.assign_param(self.param,param)
        input_file.close()
        return
    
    
    def save_(self,path):
        if self.save_best_only==False:
            if self.max_save_files==None or self.max_save_files==1:
                output_file=open(path,'wb')
            else:
                if self.train_acc!=None and self.test_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}-{2:.4f}.dat'.format(self.total_epoch,self.train_acc,self.test_acc))
                elif self.train_acc!=None:
                    path=path.replace(path[path.find('.'):],'-{0}-{1:.4f}.dat'.format(self.total_epoch,self.train_acc))
                else:
                    path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_epoch))
                output_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            optimizer_config=tf.keras.optimizers.serialize(self.optimizer)
            self.optimizer=None
            pickle.dump(self,output_file)
            pickle.dump(optimizer_config,output_file)
            output_file.close()
        else:
            if self.monitor=='val_loss':
                if self.test_loss<self.val_loss:
                    self.val_loss=self.test_loss
                    self.save(path)
                if self.val_loss==0:
                    self.val_loss=self.test_loss
            elif self.monitor=='val_accuracy':
                if self.test_acc>self.val_accuracy:
                    self.val_accuracy=self.test_acc
                    self.save(path)
                if self.val_accuracy==1:
                    self.val_accuracy=self.test_acc
        return
    
    
    def save(self,path):
        output_file=open(path,'wb')
        pickle.dump(self,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        model=pickle.load(input_file)
        self.__dict__.update(model.__dict__)
        input_file.close()
        return
    
    
    def init():
        Model.param=[]
        Model.param_dict=dict()
        Model.param_dict['dense_weight']=[]
        Model.param_dict['dense_bias']=[]
        Model.param_dict['conv2d_weight']=[]
        Model.param_dict['conv2d_bias']=[]
        Model.layer_dict=dict()
        Model.layer_param=dict()
        Model.layer_list=[]
        Model.layer_eval=dict()
        Model.counter=0
        Model.name_list=[]
        Model.name_list_=[]
        Model.ctl_list=[]
        Model.ctsl_list=[]
        Model.name=None
        Model.name_=None
        Model.train_flag=True
        return
