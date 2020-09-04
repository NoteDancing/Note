import tensorflow as tf
import Note.create.create as c
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class Note:
    def __init__(self,model):
        self.model=model
        self.tf2=c.tf2()
        with tf.name_scope('data/shape0'):
            self.train_data=self.model.train_data
            self.train_labels=self.model.train_labels
            if type(self.train_data)==list:
                self.data_batch=[x for x in range(len(self.train_data))]
            if type(self.train_labels)==list:
                self.labels_batch=[x for x in range(len(self.train_labels))]
            self.test_data=self.model.test_data
            self.test_labels=self.model.test_labels
            if type(self.train_data)==list:
                self.shape0=self.model.train_data[0].shape[0]
            else:
                self.shape0=self.model.train_data.shape[0]
        with tf.name_scope('parameter'):
            self.parameter=self.model.parameter
        with tf.name_scope('hyperparameter'):
            self.batch=None
            self.epoch=0
            self.lr=None
            self.l2=None
            self.dropout=None
            self.hyperparameter=self.model.hyperparameter
        with tf.name_scope('regulation'):
            self.regulation=self.model.regulation
        with tf.name_scope('optimizer'):
            self.optimizer=self.model.optimzier
        self.train_loss=None
        self.train_acc=None
        self.train_loss_list=[]
        self.train_acc_list=[]
        self.test_loss=None
        self.test_acc=None
        self.test_loss_list=[]
        self.test_acc_list=[]
        self.test_flag=False
        self.total_epoch=0
        self.time=0
        self.total_time=0
        self.processor='GPU:0'
    
    
    def init(self):
        self.model.flag=0
        self.parameter=[]
        self.model.parameter=[]
        self.train_loss_list.clear()
        self.train_acc_list.clear()
        self.test_loss_list.clear()
        self.test_acc_list.clear()
        self.test_flag=False
        self.total_epoch=0
        self.total_time=0
        return
    
    
    def train(self,batch=None,epoch=None,l2=None,dropout=None,optimizer=None,optimizern=None,lr=None,test=False,test_batch=None,model_path=None,one=True,processor=None):
        with tf.name_scope('parameter'):
            self.parameter=self.model.parameter
        with tf.name_scope('hyperparameter'):
            self.batch=batch
            self.epoch=0
            self.lr=lr
            self.l2=l2
            self.dropout=dropout
            self.hyperparameter=self.model.hyperparameter
        self.test_flag=test
        if processor!=None:
            self.processor=processor
        with tf.name_scope('optimizer'):
            self.optimizer=self.model.optimzier
            if optimizer!=None:
                optimizer=optimizer
            else:
                optimizer=optimizern
        if self.total_epoch==0:
            epoch=epoch+1
        for i in range(epoch):
            t1=time.time()
            if batch!=None:
                if type(self.train_data)==list:
                    train_data=[x for x in range(len(self.train_data))]
                if type(self.train_labels)==list:
                    train_labels=[x for x in range(len(self.train_labels))]
                batches=int((self.shape0-self.shape0%batch)/batch)
                self.tf2.batches=batches
                total_loss=0
                total_acc=0
                random=np.arange(self.shape0)
                np.random.shuffle(random)
                with tf.name_scope('randomize_data'):
                    if type(self.train_data)==list:
                        for i in range(len(self.train_data)):
                            train_data[i]=self.train_data[i]
                    else:
                        train_data=self.train_data
                    if type(self.train_labels)==list:
                        for i in range(len(self.train_labels)):
                            train_labels[i]=self.train_labels[i]
                    else:
                        train_labels=self.train_labels
                for j in range(batches):
                    self.tf2.index1=j*batch
                    self.tf2.index2=(j+1)*batch
                    with tf.name_scope('data_batch'):
                        if type(self.train_data)==list:
                            for i in range(len(self.train_data)):
                                self.data_batch[i]=self.tf2.batch(train_data[i])
                        else:
                            self.data_batch=self.tf2.batch(train_data)
                        if type(self.train_labels)==list:
                            for i in range(len(self.train_data)):
                                self.labels_batch[i]=self.tf2.batch(train_labels[i])
                        else:
                            self.labels_batch=self.tf2.batch(train_labels)
                    with tf.GradientTape() as tape:
                        with tf.name_scope('forward_propagation/loss'):
                            self.output=self.model.forward_propagation(self,self.data_batch,self.dropout)
                            batch_loss=self.model.loss(self,self.output,self.labels_batch,self.l2)
                        if i==0 and self.total_epoch==0:
                            batch_loss=batch_loss.numpy()
                        else:
                            with tf.name_scope('apply_gradient'):
                                if optimizer!=None:
                                    self.tf2.apply_gradient(tape,optimizer,batch_loss,self.parameter)
                                else:
                                    gradient=tape.gradient(batch_loss,self.parameter)
                                    optimizer(gradient,self.parameter)
                    total_loss+=batch_loss
                    if self.model.accuracy==1:
                        with tf.name_scope('accuracy'):
                            batch_acc=self.model.accuracy(self,self.output,self.labels_batch)
                        batch_acc=batch_acc.numpy()
                        total_acc+=batch_acc
                if self.shape0%batch!=0:
                    batches+=1
                    self.tf2.batches+=1
                    self.tf2.index1=batches*batch
                    self.tf2.index2=batch-(self.shape0-batches*batch)
                    with tf.name_scope('data_batch'):
                        if type(self.train_data)==list:
                            for i in range(len(self.train_data)):
                                self.data_batch[i]=self.tf2.batch(train_data[i])
                        else:
                            self.data_batch=self.tf2.batch(train_data)
                        if type(self.train_labels)==list:
                            for i in range(len(self.train_data)):
                                self.labels_batch[i]=self.tf2.batch(train_labels[i])
                        else:
                            self.labels_batch=self.tf2.batch(train_labels)
                    with tf.GradientTape() as tape:
                        with tf.name_scope('forward_propagation/loss'):
                            self.output=self.model.forward_propagation(self,self.data_batch,self.dropout)
                            batch_loss=self.model.loss(self,self.output,self.labels_batch,self.l2)
                        if i==0 and self.total_epoch==0:
                            batch_loss=batch_loss.numpy()
                        else:
                            with tf.name_scope('apply_gradient'):
                                if optimizer!=None:
                                    self.tf2.apply_gradient(tape,optimizer,batch_loss,self.parameter)
                                else:
                                    gradient=tape.gradient(batch_loss,self.parameter)
                                    optimizer(gradient,self.parameter)
                    total_loss+=batch_loss
                    if self.model.accuracy==1:
                        with tf.name_scope('accuracy'):
                            batch_acc=self.model.accuracy(self,self.output,self.labels_batch)
                        batch_acc=batch_acc.numpy()
                        total_acc+=batch_acc
                loss=total_loss/batches
                if self.model.accuracy==1:
                    train_acc=total_acc/batches
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
                if self.model.accuracy==1:
                    self.train_acc_list.append(float(train_acc))
                    self.train_acc=train_acc
                    self.train_acc=self.train_acc.astype(np.float32)
                if test==True:
                    with tf.name_scope('test'):
                        self.test_loss,self.test_acc=self.test(test_batch)
                        self.test_loss_list.append(self.test_loss)
                        if self.model.accuracy==1:
                            self.test_acc_list.append(self.test_acc)
            else:
                if type(self.train_data)==list:
                    train_data=[x for x in range(len(self.train_data))]
                if type(self.train_labels)==list:
                    train_labels=[x for x in range(len(self.train_labels))]
                random=np.arange(self.shape0)
                np.random.shuffle(random)
                with tf.name_scope('randomize_data'):
                    if type(self.train_data)==list:
                        for i in range(len(self.train_data)):
                            train_data[i]=self.train_data[i]
                    else:
                        train_data=self.train_data
                    if type(self.train_labels)==list:
                        for i in range(len(self.train_labels)):
                            train_labels[i]=self.train_labels[i]
                    else:
                        train_labels=self.train_labels
                with tf.GradientTape() as tape:
                    with tf.name_scope('forward_propagation/loss'):
                        self.output=self.model.forward_propagation(self,train_data,self.dropout)
                        train_loss=self.model.loss(self,self.output,train_labels,self.l2)
                    if i==0 and self.total_epoch==0:
                        loss=train_loss.numpy()
                    else:
                       with tf.name_scope('apply_gradient'):
                           if optimizer!=None:
                               self.tf2.apply_gradient(tape,optimizer,batch_loss,self.parameter)
                           else:
                               gradient=tape.gradient(batch_loss,self.parameter)
                               optimizer(gradient,self.parameter)  
                self.train_loss_list.append(loss.astype(np.float32))
                self.train_loss=loss
                self.train_loss=self.train_loss.astype(np.float32)
                if self.model.accuracy==1:
                    with tf.name_scope('accuracy'):
                        acc=self.model.accuracy(self,self.output,train_labels)
                    acc=train_acc.numpy()
                    self.train_acc_list.append(float(acc))
                    self.train_acc=acc
                    self.train_acc=self.train_acc.astype(np.float32)
                if test==True:
                    with tf.name_scope('test'):
                        self.test_loss,self.test_acc=self.test(test_batch)
                        self.test_loss_list.append(self.test_loss)
                        if self.model.accuracy==1:
                            self.test_acc_list.append(self.test_acc)
            self.epoch+=1
            self.total_epoch+=1
            if epoch%10!=0:
                temp=epoch-epoch%10
                temp=int(temp/10)
            else:
                temp=epoch/10
            if temp==0:
                temp=1
            if i%temp==0:
                if self.total_epoch==0:
                    print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                else:
                    print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch+i+1,self.train_loss))
                if model_path!=None and i%epoch*2==0:
                    self.save(model_path,i,one)
            t2=time.time()
            self.time+=(t2-t1)
        self.time=self.time-int(self.time)
        if self.time<0.5:
            self.time=int(self.time)
        else:
            self.time=int(self.time)+1
        self.total_time+=self.time
        print()
        print('last loss:{0:.6f}'.format(self.train_loss))
        if self.model.accuracy==1:
            if self.model.acc=='%':
                print('accuracy:{0:.1f}'.format(self.train_acc*100))
            else:
                print('accuracy:{0:.6f}'.format(self.train_acc))   
        print('time:{0}s'.format(self.time))
        return
    
    
    def test(self,batch=None):
        if batch!=None:
            total_loss=0
            total_acc=0
            if type(self.test_data)==list:
                batches=int((self.test_data[0].shape[0]-self.test_data[0].shape[0]%batch)/batch)
                shape0=self.test_data[0].shape[0]
            else:
                batches=int((self.test_data.shape[0]-self.test_data.shape[0]%batch)/batch)
                shape0=self.test_data.shape[0]
            self.tf2.batches=batches
            for j in range(batches):
                self.tf2.index1=j*batch
                self.tf2.index2=(j+1)*batch
                with tf.name_scope('data_batch'):
                    if type(self.train_data)==list:
                        for i in range(len(self.test_data)):
                            self.data_batch[i]=self.tf2.batch(self.test_data[i])
                    else:
                        self.data_batch=self.tf2.batch(self.test_data)
                    if type(self.test_labels)==list:
                        for i in range(len(self.test_labels)):
                            self.labels_batch[i]=self.tf2.batch(self.test_labels[i])
                    else:
                        self.labels_batch=self.tf2.batch(self.test_labels)
                with tf.name_scope('loss'):
                    batch_loss=self.model.loss(self)
                total_loss+=batch_loss.numpy()
                if self.model.accuracy==1:
                    with tf.name_scope('accuracy'):
                        batch_acc=self.model.accuracy(self)
                    total_acc+=batch_acc.numpy()
            if shape0%batch!=0:
                batches+=1
                self.tf2.batches+=1
                self.tf2.index1=batches*batch
                self.tf2.index2=batch-(self.shape0-batches*batch)
                with tf.name_scope('data_batch'):
                    if type(self.train_data)==list:
                        for i in range(len(self.test_data)):
                            self.data_batch[i]=self.tf2.batch(self.test_data[i])
                    else:
                        self.data_batch=self.tf2.batch(self.test_data)
                    if type(self.test_labels)==list:
                        for i in range(len(self.test_labels)):
                            self.labels_batch[i]=self.tf2.batch(self.test_labels[i])
                    else:
                        self.labels_batch=self.tf2.batch(self.test_labels)
                with tf.name_scope('loss'):
                    batch_loss=self.model.loss(self)
                total_loss+=batch_loss.numpy()
                if self.model.accuracy==1:
                    with tf.name_scope('accuracy'):
                        batch_acc=self.model.accuracy(self)
                    total_acc+=batch_acc.numpy()
            test_loss=total_loss/batches
            test_loss=test_loss
            test_loss=test_loss.astype(np.float32)
            if self.model.accuracy==1:
                test_acc=total_acc/batches
                test_acc=test_acc
                test_acc=test_acc.astype(np.float32)
        else:
            with tf.name_scope('loss'):
                test_loss=self.model.loss(self)
            if self.model.accuracy==1:
                with tf.name_scope('accuracy'):
                    test_acc=self.model.accuracy(self)
                test_loss=test_loss.numpy().astype(np.float32)
                test_acc=test_acc.numpy().astype(np.float32)
        print('test loss:{0:.6f}'.format(test_loss))
        if self.model.accuracy==1:
            if self.model.acc=='%':
                print('accuracy:{0:.1f}'.format(test_acc*100))
            else:
                print('accuracy:{0:.6f}'.format(test_acc))
            if self.model.acc=='%':
                return test_loss,test_acc*100
            else:
                return test_loss,test_acc
        else:
            return test_loss,None
    
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.total_epoch))
        if self.regulation!=None:
            print()
            print('regulation:{0}'.format(self.regulation))
        if self.optimizer!=None:
            print()
            print('optimizer:{0}'.format(self.optimizer))
        print()
        print('learning rate:{0}'.format(self.lr))
        print()
        print('time:{0:.3f}s'.format(self.total_time))
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0:.6f}'.format(self.train_loss))
        if self.model.acc=='%':
            print('train acc:{0:.1f}'.format(self.train_acc*100))
        else:
            print('train acc:{0:.6f}'.format(self.train_acc))       
        return
        
    
    def test_info(self):
        print()
        print('test loss:{0:.6f}'.format(self.test_loss))
        if self.model.acc=='%':
            print('test acc:{0:.1f}'.format(self.test_acc*100))
        else:
            print('test acc:{0:.6f}'.format(self.test_acc))      
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
        plt.plot(np.arange(self.total_epoch),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.train_acc_list)
        plt.title('train acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('train loss:{0:.6f}'.format(self.train_loss))
        if self.model.acc=='%':
            print('train acc:{0:.1f}'.format(self.train_acc*100))
        else:
            print('train acc:{0:.6f}'.format(self.train_acc))    
        return
    
    
    def test_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.test_loss_list)
        plt.title('test loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.test_acc_list)
        plt.title('test acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        print('test loss:{0:.6f}'.format(self.test_loss))
        if self.model.acc=='%':
            print('test acc:{0:.1f}'.format(self.test_acc*100))
        else:
            print('test acc:{0:.6f}'.format(self.test_acc))  
        return 
    
    
    def comparison(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_epoch),self.train_loss_list,'b-',label='train loss')
        if self.test_flag==True:
            plt.plot(np.arange(self.total_epoch),self.test_loss_list,'r-',label='test loss')
        plt.title('loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.figure(2)
        plt.plot(np.arange(self.total_epoch),self.train_acc_list,'b-',label='train acc')
        if self.test_flag==True:
            plt.plot(np.arange(self.total_epoch),self.test_acc_list,'r-',label='test acc')
        plt.title('accuracy')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        print('train loss:{0}'.format(self.train_loss))
        if self.model.acc=='%':
            print('train acc:{0:.1f}'.format(self.train_acc*100))
        else:
            print('train acc:{0:.6f}'.format(self.train_acc))     
        if self.test_flag==True:        
            print()
            print('-------------------------------------')
            print()
            print('test loss:{0:.6f}'.format(self.test_loss))
            if self.model.acc=='%':
                print('test acc:{0:.1f}'.format(self.test_acc*100))
            else:
                print('test acc:{0:.6f}'.format(self.test_acc)) 
        return
    
    
    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        with tf.name_scope('save_parameter'):  
            pickle.dump(self.parameter,output_file)
        with tf.name_scope('save_hyperparameter'):
            pickle.dump(self.batch,output_file)
            pickle.dump(self.lr,output_file)
            pickle.dump(self.l2,output_file)
            pickle.dump(self.dropout,output_file)
            pickle.dump(self.hyperparameter,output_file)
        with tf.name_scope('save_regulation'):
            pickle.dump(self.regulation,output_file)
        with tf.name_scope('save_optimizer'):
            pickle.dump(self.optimizer,output_file)
        pickle.dump(self.model.accuracy,output_file)
        pickle.dump(self.model.acc,output_file)
        pickle.dump(self.shape0,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_acc,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.train_acc_list,output_file)
        pickle.dump(self.test_flag,output_file)
        if self.test_flag==True:
            pickle.dump(self.test_loss,output_file)
            pickle.dump(self.test_acc,output_file)
            pickle.dump(self.test_loss_list,output_file)
            pickle.dump(self.test_acc_list,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        pickle.dump(self.processor,output_file)
        output_file.close()
        return
    

    def restore(self,model_path):
        self.model.flag=1
        input_file=open(model_path,'rb')
        with tf.name_scope('restore_parameter'):
            self.model.parameter=pickle.load(input_file)
        with tf.name_scope('restore_hyperparameter'):
            self.batch=pickle.load(input_file)
            self.lr=pickle.load(input_file)
            self.l2=pickle.load(input_file)
            self.dropout=pickle.load(input_file)
            self.hyperparameter=pickle.load(input_file)
        with tf.name_scope('restore_regulation'):
            self.regulation=pickle.load(input_file)
        with tf.name_scope('restore_optimizer'):
            self.optimizer=pickle.load(input_file)
        self.model.accuracy=pickle.load(input_file)
        self.model.acc=pickle.load(input_file)
        self.shape0=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_acc=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.train_acc_list=pickle.load(input_file)
        self.test_flag=pickle.load(input_file)
        if self.test_flag==True:
            self.test_loss=pickle.load(input_file)
            self.test_acc=pickle.load(input_file)
            self.test_loss_list=pickle.load(input_file)
            self.test_acc_list=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        input_file.close()
        return
