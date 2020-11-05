import tensorflow as tf
import numpy as np
import pickle
import time


class Q_learning:
    def __init__(self,net,net_p,state,state_name,action_name,search_space,epsilon=None,discount=None,episode_step=None,optimizer=None,lr=None,save_episode=True):
        self.net=net
        self.net_p=net_p
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.search_space=search_space
        self.state_len=len(self.state_name)
        self.action_len=len(self.action_name)
        self.epsilon=epsilon
        self.discount=discount
        self.episode_step=episode_step
        self.lr=lr
        self.optimizer=optimizer
        self.save_episode=save_episode
        self.opt_flag=False
        self.episode_num=0
        self.total_episode=0
        self.time=0
        self.total_time=0
       
        
    def init(self,dtype=np.int32):
        self.t3=time.time()
        if len(self.state_name)>self.state_len:
            self.state=np.concatenate(self.state,np.arange(len(self.state_name)-self.state_len,dtype=dtype)+len(self.state_len))
            self.state_one=np.concatenate(self.state_one,np.ones(len(self.state_name)-self.state_len,dtype=dtype))
            self.state_prob=self.state_one/len(self.state_name)
            self.action=np.concatenate(self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+len(self.action_len))
            self.action_p=np.concatenate(self.action_prob,np.ones(len(self.action_name)-self.action_len,dtype=dtype))
        else:
            self.state=np.arange(len(self.state_name),dtype=dtype)
            self.state_one=np.ones(len(self.state_name),dtype=dtype)
            self.state_prob=self.state_one/len(self.state_name)
            self.action=np.arange(len(self.action_name),dtype=dtype)
            self.action_p=np.ones(len(self.action_name),dtype=dtype)
        self.t4=time.time()
        return
    
    
    def epsilon_greedy_policy(self,s,action_p):
        action_prob=action_p
        action_prob=action_prob*self.epsilon/np.sum(action_p)
        best_action=np.argmax(self.predict_net(self.state[self.state_name[s]]).numpy())
        action_prob[best_action]+=1-self.epsilon
        return action_prob
    
    
    def loss(self,s,a,next_s,r):
        return (r+self.discount*tf.reduce_max(self.net(self.state[next_s]))-self.net(self.state[s])[a])**2
    
    
    def learn(self,episode_num,path=None,one=True):
        for i in range(episode_num):
            loss=0
            episode=[]
            s=np.random.choice(self.state,p=self.state_prob)
            if self.episode_step==None:
                while True:
                    t1=time.time()
                    action_prob=self.epsilon_greedy_policy(s,self.action_p)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                    if end:
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],r,end])
                        break
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.self.action_name[a],r])
                    loss+=self.loss(s,a,next_s,r)
                    s=next_s
                    with tf.GradientTape() as tape:
                        gradient=tape.gradient(1/2*loss,self.net_p)
                        if self.opt_flag==True:
                            self.optimizer(gradient,self.net_p)
                        else:
                            self.optimizer.apply_gradients(zip(gradient,self.net_p))
                    loss=loss.numpy()
                    t2=time.time()
                    self.time+=(t2-t1)
            else:
                for _ in range(self.episode_step):
                    t1=time.time()
                    action_prob=self.epsilon_greedy_policy(s,self.action_p)
                    a=np.random.choice(self.action,p=action_prob)
                    next_s,r,end=self.search_space[self.state_name[s]][self.action_name[a]]
                    if end:
                        if self.save_episode==True:
                            episode.append([self.state_name[s],self.action_name[a],r,end])
                        break
                    if self.save_episode==True:
                        episode.append([self.state_name[s],self.self.action_name[a],r])
                    loss+=self.loss(s,a,next_s,r)
                    s=next_s
                    with tf.GradientTape() as tape:
                        gradient=tape.gradient(1/2*loss,self.net_p)
                        if self.opt_flag==True:
                            self.optimizer(gradient,self.net_p)
                        else:
                            self.optimizer.apply_gradients(zip(gradient,self.net_p))
                    loss=loss.numpy()
                    t2=time.time()
                    self.time+=(t2-t1)
            if episode_num%10!=0:
                temp=episode_num-episode_num%10
                temp=int(temp/10)
            else:
                temp=episode_num/10
            if temp==0:
                temp=1
            if i%temp==0:
                print('episode num:{0}   loss:{1:.6f}'.format(i+1,loss))
                if path!=None and i%episode_num*2==0:
                    self.save(path,i,one)
            self.episode_num+=1
            self.total_episode+=1
        if self.save_episode==True:
            self.episode.append(episode)
        if self.time<0.5:
            self.time=int(self.time+(self.t4-self.t3))
        else:
            self.time=int(self.time+(self.t4-self.t3))+1
        self.total_time+=self.time
        print()
        print('last loss:{0:.6f}'.format(loss))
        print('time:{0}s'.format(self.time))
        return
    
    
    def save(self,path,i=None,one=True):
        if one==True:
            output_file=open(path+'.dat','wb')
        else:
            output_file=open(path+'-{0}.dat'.format(i+1),'wb')
        pickle.dump(self.state_len,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.state,output_file)
        pickle.dump(self.state_prob,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_p,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.lr,output_file)
        pickle.dump(self.optimizer,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.opt_flag,output_file)
        pickle.dump(self.state_one,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,path):
        input_file=open(path,'rb')
        self.state_len=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        if self.state_len==len(self.state_name):
            self.state=pickle.load(input_file)
            self.state_prob=pickle.load(input_file)
        if self.action_len==len(self.action_name):
            self.action=pickle.load(input_file)
            self.action_p=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.lr=pickle.load(input_file)
        self.optimizer=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.opt_flag=pickle.load(input_file)
        self.state_one=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=self.time
        input_file.close()
        return
