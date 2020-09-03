import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time


class skip_gram:
    def __init__(self,cword=None,bword=None,labels=None):
        self.graph=tf.Graph()
        self.cword=cword
        self.bword=bword
        self.labels=labels
        with self.graph.as_default():
            if type(cword)==np.ndarray:
                self.shape0=cword.shape[0]
                self.cword_place=tf.placeholder(dtype=cword.dtype,shape=[None,None],name='cword')
                self.bword_place=tf.placeholder(dtype=bword.dtype,shape=[None,None,None],name='bword')
                self.labels_place=tf.placeholder(dtype=labels.dtype,shape=[None,None],name='labels')
                self.cword_dtype=cword.dtype
                self.bword_dtype=bword.dtype
                self.labels_dtype=labels.dtype
        self.cword_weight=None
        self.bword_weight=None
        self.last_cword_weight=None
        self.last_bword_weight=None
        self.batch=None
        self.epoch=0
        self.lr=None
        self.optimizer=None
        self.train_loss=None
        self.train_loss_list=[]
        self.total_epoch=0
        self.time=0
        self.total_time=0
        self.processor='/gpu:0'
    
    
    def iet(self):
        self.total_epoch=self.total_epoch-self.epoch
        self.total_time=self.total_time-self.time
        return
    
    
    def weight_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
            
    
    def bias_init(self,shape,mean,stddev,name):
        return tf.Variable(tf.random.normal(shape=shape,mean=mean,stddev=stddev,dtype=self.dtype),name=name)
    
    
    def structure(self,d,vocab_size,mean=0,stddev=0.07,dtype=np.float32):
        with self.graph.as_default():
            self.d=d
            self.vocab_size=vocab_size
            self.continue_train=False
            self.flag=None
            self.end_flag=False
            self.train_loss_list.clear()
            self.dtype=dtype
            self.epoch=0
            self.total_epoch=0
            self.time=None
            self.total_time=None
            with tf.name_scope('parameter_initialization'):            
                self.cword_weight=self.weight_init(shape=[vocab_size,d],mean=mean,stddev=stddev,dtype=dtype,name='cword_weight')
                self.bword_weight=self.weight_init(shape=[vocab_size,d],mean=mean,stddev=stddev,dtype=dtype,name='bword_weight')
                return
    
    
    def forward_propagation(self,cword,bword):
        with self.graph.as_default():
            processor=self.processor
            with tf.device(processor):
                cword_vec=tf.matmul(cword,self.cword_weight)
                cword_vec=tf.reshape(cword_vec,shape=[cword_vec.shape[0],1,cword_vec.shape[1]])
                bword_vec=tf.einsum('ijk,kl->ijl',bword,self.bword_weight)
                cb=tf.reduce_sum(cword_vec*bword_vec,axis=2)
            return cb
            
        
    def train(self,batch=None,epoch=None,optimizer='Adam',lr=0.001,acc=True,train_summary_path=None,model_path=None,one=True,continue_train=False,processor=None):
        with self.graph.as_default():
            self.batch=batch
            self.epoch=0
            self.optimizer=optimizer
            self.lr=lr
            if continue_train!=True:
                if self.continue_train==True:
                    continue_train=True
                else:
                    self.train_loss_list.clear()
            if self.continue_train==False and continue_train==True:
                self.train_loss_list.clear()
                self.continue_train=True
            if processor!=None:
                self.processor=processor
            if continue_train==True and self.end_flag==True:
                self.end_flag=False
                self.cword_weight=tf.Variable(self.last_cword_weight,name='cword_weight')
                self.bword_weight=tf.Variable(self.last_bword_weight,name='bword_weight')
                self.last_cword_weight=None
                self.last_bword_weight=None
            if continue_train==True and self.flag==1:
                self.flag=0
                self.cword_weight=tf.Variable(self.last_cword_weight,name='cword_weight')
                self.bword_weight=tf.Variable(self.last_bword_weight,name='bword_weight')
                self.last_cword_weight=None
                self.last_bword_weight=None
#     －－－－－－－－－－－－－－－forward propagation－－－－－－－－－－－－－－－
            train_output=self.forward_propagation(self.cword_place,self.bword_place)
#     －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
            with tf.name_scope('train_loss'):
                train_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=train_output,labels=self.labels_place))   
            if self.optimizer=='Gradient':
                opt=tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(train_loss)
            if self.optimizer=='RMSprop':
                opt=tf.train.RMSPropOptimizer(learning_rate=lr).minimize(train_loss)
            if self.optimizer=='Momentum':
                opt=tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.99).minimize(train_loss)
            if self.optimizer=='Adam':
                opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(train_loss)
            if train_summary_path!=None:
                train_loss_scalar=tf.summary.scalar('train_loss',train_loss)
                train_merging=tf.summary.merge([train_loss_scalar])
                train_writer=tf.summary.FileWriter(train_summary_path)
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.allow_soft_placement=True
            sess=tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            self.sess=sess
            if self.total_epoch==0:
                epoch=epoch+1
            t1=time.time()
            for i in range(epoch):
                if batch!=None:
                    batches=int((self.shape0-self.shape0%batch)/batch)
                    total_loss=0
                    random=np.arange(self.shape0)
                    np.random.shuffle(random)
                    cword=self.cword[random]
                    bword=self.bword[random]
                    labels=self.labels[random]
                    for j in range(batches):
                        index1=j*batch
                        index2=(j+1)*batch
                        cword_batch=cword[index1:index2]
                        bword_batch=bword[index1:index2]
                        labels_batch=labels[index1:index2]
                        feed_dict={self.cword_place:cword_batch,self.bword_place:bword_batch,self.labels_place:labels_batch}
                        if i==0 and self.total_epoch==0:
                            batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                        else:
                            batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                        total_loss+=batch_loss
                    if self.shape0%batch!=0:
                        batches+=1
                        index1=batches*batch
                        index2=batch-(self.shape0-batches*batch)
                        cword_batch=np.concatenate([cword[index1:],cword[:index2]])
                        bword_batch=np.concatenate([bword[index1:],bword[:index2]])
                        labels_batch=np.concatenate([labels[index1:],labels[:index2]])
                        feed_dict={self.cword_place:cword_batch,self.bword_place:bword_batch,self.labels_place:labels_batch}
                        if i==0 and self.total_epoch==0:
                            batch_loss=sess.run(train_loss,feed_dict=feed_dict)
                        else:
                            batch_loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                        total_loss+=batch_loss
                    loss=total_loss/batches
                    self.train_loss_list.append(float(loss))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float16)
                else:
                    random=np.arange(self.shape0)
                    np.random.shuffle(random)
                    cword=self.cword[random]
                    bword=self.bword[random]
                    labels=self.labels[random]
                    feed_dict={self.cword_place:cword,self.bword_place:bword,self.labels_place:labels}
                    if i==0 and self.total_epoch==0:
                        loss=sess.run(train_loss,feed_dict=feed_dict)
                    else:
                        loss,_=sess.run([train_loss,opt],feed_dict=feed_dict)
                    self.train_loss_list.append(float(loss))
                    self.train_loss=loss
                    self.train_loss=self.train_loss.astype(np.float16)
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
                    if continue_train==True:
                        print('epoch:{0}   loss:{1:.6f}'.format(self.total_epoch+i+1,self.train_loss))
                    else:
                        print('epoch:{0}   loss:{1:.6f}'.format(i,self.train_loss))
                    if model_path!=None and i%epoch*2==0:
                        self.save(model_path,i,one)
                    if train_summary_path!=None:
                        train_summary=sess.run(train_merging,feed_dict=feed_dict)
                        train_writer.add_summary(train_summary,i)
            t2=time.time()
            _time=(t2-t1)-int(t2-t1)
            if _time<0.5:
                self.time=int(t2-t1)
            else:
                self.time=int(t2-t1)+1
            self.total+=self.time
            print()
            print('last loss:{0}'.format(self.train_loss))
            if train_summary_path!=None:
                train_writer.close()
            if continue_train==True:
                self.last_cword_weight=sess.run(self.cword_weight)
                self.last_bword_weight=sess.run(self.bword_weight)
                self.cword_weight=tf.Variable(self.last_cword_weight,name='cword_weight')
                self.bword_weight=tf.Variable(self.last_bword_weight,name='bword_weight')
                self.last_cword_weight=None
                self.last_bword_weight=None
                sess.run(tf.global_variables_initializer())
            print('time:{0}s'.format(self.time))
            return
    
    
    def end(self):
        with self.graph.as_default():
            self.end_flag=True
            self.continue_train=False
            self.last_cword_weight=self.sess.run(self.cword_weight)
            self.last_bword_weight=self.sess.run(self.bword_weight)
            self.cword_weight=None
            self.bword_weight=None
            self.sess.close()
            return
        
    
    def train_info(self):
        print()
        print('batch:{0}'.format(self.batch))
        print()
        print('epoch:{0}'.format(self.total_epoch))
        print()
        print('optimizer:{0}'.format(self.optimizer))
        print()
        print('learning rate:{0}'.format(self.lr))
        print()
        print('time:{0:.3f}s'.format(self.total_time))
        print()
        print('-------------------------------------')
        print()
        print('train loss:{0}'.format(self.train_loss))
        return
		
    
    def info(self):
        self.train_info()
        return


    def train_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.epoch+1),self.train_loss_list)
        plt.title('train loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        print('train loss:{0}'.format(self.train_loss))
        return
    
    
    def save(self,model_path,i=None,one=True):
        if one==True:
            output_file=open(model_path+'.dat','wb')
        else:
            output_file=open(model_path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.d)
        pickle.dump(self.vocab_size)
        pickle.dump(self.last_cword_weight,output_file)   
        pickle.dump(self.last_bword_weight,output_file) 
        pickle.dump(self.shape0,output_file)
        pickle.dump(self.cword_dtype,output_file)
        pickle.dump(self.bword_dtype,output_file)
        pickle.dump(self.labels_dtype,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.train_loss,output_file)
        pickle.dump(self.train_loss_list,output_file)
        pickle.dump(self.total_epoch,output_file)
        pickle.dump(self.total_time,output_file)
        pickle.dump(self.processor,output_file)
        output_file.close()
        return
    

    def restore(self,model_path):
        input_file=open(model_path,'rb')
        self.d=pickle.load(input_file)
        self.vocab_size=pickle.load(input_file)
        self.last_cword_weight=pickle.load(input_file)  
        self.last_bword_weight=pickle.load(input_file)
        self.shape0=pickle.load(input_file)
        self.cword_dtype=pickle.load(input_file)
        self.bword_dtype=pickle.load(input_file)
        self.labels_dtype=pickle.load(input_file)
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.cword_place=tf.placeholder(dtype=self.cword_dtype,shape=[None,None],name='cword')
            self.bword_place=tf.placeholder(dtype=self.bword_dtype,shape=[None,None,None],name='bword')
            self.labels=tf.placeholder(dtype=self.labels_dtype,shape=[None],name='labels')
        self.batch=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.train_loss=pickle.load(input_file)
        self.train_loss_list=pickle.load(input_file)
        self.total_epoch=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        self.processor=pickle.load(input_file)
        self.flag=1
        input_file.close()
        return
    
    
    def emb_weight(self):
        return self.last_cword_weight
