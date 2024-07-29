import tensorflow as tf
from Note import nn
from Note.RL.rl.prioritized_replay import pr
from multiprocessing import Array,Value
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import statistics
import pickle
import os
import time


class RL:
    def __init__(self):
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        self.epsilon=None
        self.reward_list=[]
        self.step_counter=0
        self.prioritized_replay=pr()
        self.seed=7
        self.optimizer_=None
        self.path=None
        self.save_freq=1
        self.save_freq_=None
        self.max_save_files=None
        self.save_best_only=False
        self.save_param_only=False
        self.path_list=[]
        self.loss=None
        self.loss_list=[]
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def action_vec(self):
        if self.epsilon_!=None:
            self.action_one=np.ones(self.env.action_space.n,dtype=np.int8)
        return
    
    
    def set_up(self,epsilon=None,pool_size=None,batch=None,update_batches=None,update_steps=None,trial_count=None,criterion=None,PPO=False,pr=False,HER=False,initial_TD=7,alpha=0.7):
        if pr==False:
            self.epsilon_=epsilon
        self.pool_size=pool_size
        self.batch=batch
        self.update_batches=update_batches
        self.update_steps=update_steps
        self.trial_count=trial_count
        self.criterion=criterion
        self.PPO=PPO
        self.pr=pr
        self.HER=HER
        if pr==True:
            self.epsilon_pr=epsilon
            self.initial_TD=initial_TD
            self.alpha=alpha
        self.action_vec()
        return
    
    
    def epsilon_greedy_policy(self,s):
        action_prob=self.action_one*self.epsilon_/len(self.action_one)
        best_a=np.argmax(self.action(s))
        action_prob[best_a]+=1-self.epsilon_
        return action_prob
    
    
    def pool(self,s,a,next_s,r,done):
        if self.mp_flag==True:
            if type(self.state_pool[7])!=np.ndarray and self.state_pool[7]==None:
                if type(s) in [int,float]:
                    s=np.array(s)
                    self.state_pool[7]=np.expand_dims(s,axis=0)
                elif type(s)==tuple:
                    s=np.array(s)
                    self.state_pool[7]=s
                else:
                    self.state_pool[7]=s
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool[7]=np.expand_dims(a,axis=0)
                else:
                    self.action_pool[7]=a
                self.next_state_pool[7]=np.expand_dims(next_s,axis=0)
                self.reward_pool[7]=np.expand_dims(r,axis=0)
                self.done_pool[7]=np.expand_dims(done,axis=0)
            else:
                self.state_pool[7]=np.concatenate((self.state_pool[7],s),0)
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool[7]=np.concatenate((self.action_pool[7],np.expand_dims(a,axis=0)),0)
                else:
                    self.action_pool[7]=np.concatenate((self.action_pool[7],a),0)
                self.next_state_pool[7]=np.concatenate((self.next_state_pool[7],np.expand_dims(next_s,axis=0)),0)
                self.reward_pool[7]=np.concatenate((self.reward_pool[7],np.expand_dims(r,axis=0)),0)
                self.done_pool[7]=np.concatenate((self.done_pool[7],np.expand_dims(done,axis=0)),0)
            if len(self.state_pool[7])>self.pool_size:
                self.state_pool[7]=self.state_pool[7][1:]
                self.action_pool[7]=self.action_pool[7][1:]
                self.next_state_pool[7]=self.next_state_pool[7][1:]
                self.reward_pool[7]=self.reward_pool[7][1:]
                self.done_pool[7]=self.done_pool[7][1:]
        else:
            if type(self.state_pool)!=np.ndarray and self.state_pool==None:
                if type(s) in [int,float]:
                    s=np.array(s)
                    self.state_pool=np.expand_dims(s,axis=0)
                elif type(s)==tuple:
                    s=np.array(s)
                    self.state_pool=s
                else:
                    self.state_pool=s
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool=np.expand_dims(a,axis=0)
                else:
                    self.action_pool=a
                self.next_state_pool=np.expand_dims(next_s,axis=0)
                self.reward_pool=np.expand_dims(r,axis=0)
                self.done_pool=np.expand_dims(done,axis=0)
            else:
                self.state_pool=np.concatenate((self.state_pool,s),0)
                if type(a)==int:
                    a=np.array(a)
                    self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0)
                else:
                    self.action_pool=np.concatenate((self.action_pool,a),0)
                self.next_state_pool=np.concatenate((self.next_state_pool,np.expand_dims(next_s,axis=0)),0)
                self.reward_pool=np.concatenate((self.reward_pool,np.expand_dims(r,axis=0)),0)
                self.done_pool=np.concatenate((self.done_pool,np.expand_dims(done,axis=0)),0)
            if len(self.state_pool)>self.pool_size:
                self.state_pool=self.state_pool[1:]
                self.action_pool=self.action_pool[1:]
                self.next_state_pool=self.next_state_pool[1:]
                self.reward_pool=self.reward_pool[1:]
                self.done_pool=self.done_pool[1:]
        return
    
    
    def choose_action(self,s):
        if self.epsilon!=None:
            self.epsilon_=self.epsilon(self.step_counter)
        if self.epsilon_==None:
            if hasattr(self, 'action'):
                action_prob=self.action(s)
                a=np.random.choice(self.action_count,p=action_prob)
            elif hasattr(self, 'noise'):
                a=(self.action(s)+self.noise()).numpy()
        else:
            action_prob=self.epsilon_greedy_policy(s)
            a=np.random.choice(self.action_count,p=action_prob)
        return a
    
    
    def env_(self,a=None,initial=None,p=None):
        if initial==True:
            if self.mp_flag==True:
                state=self.env[p].reset(seed=self.seed)
                return state
            else:
                state=self.env.reset(seed=self.seed)
                return state 
        else:
            if self.mp_flag==True:
                next_state,reward,done,_=self.env[p].step(a)
                return next_state,reward,done
            else:
                next_state,reward,done,_=self.env.step(a)
                return next_state,reward,done
    
    
    def data_func(self):
        if self.pr_:
            s,a,next_s,r,d=self.prioritized_replay.sample(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.epsilon_pr,self.alpha,self.batch)
        elif self.HER:
            s = []
            a = []
            next_s = []
            r = []
            d = []
            if self.mp_flag==True:
                length=min(len(self.state_pool[7]),len(self.action_pool[7]),len(self.next_state_pool[7]),len(self.reward_pool[7]),len(self.done_pool[7]))
                state_pool=self.state_pool[7][:length]
                action_pool=self.action_pool[7][:length]
                next_state_pool=self.next_state_pool[7][:length]
                done_pool=self.done_pool[7][:length]
            else:
                state_pool=self.state_pool
                action_pool=self.action_pool
                next_state_pool=self.next_state_pool
                done_pool=self.done_pool
            for _ in range(self.batch):
                step_state = np.random.randint(0, len(state_pool)-1)
                step_goal = np.random.randint(step_state+1, step_state+np.argmax(done_pool[step_state+1:])+2)
                state = state_pool[step_state]
                next_state = next_state_pool[step_state]
                action = action_pool[step_state]
                goal = state_pool[step_goal]
                reward, done = self.reward_done_func(next_state, goal)
                state = np.hstack((state, goal))
                next_state = np.hstack((next_state, goal))
                s.append(state)
                a.append(action)
                next_s.append(next_state)
                r.append(reward)
                d.append(done)
            s = np.array(s)
            a = np.array(a)
            next_s = np.array(next_s)
            r = np.array(r)
            d = np.array(d)
        return s,a,next_s,r,d
    
    
    @tf.function(jit_compile=True)
    def train_step(self, train_data, train_loss, optimizer):
        with tf.GradientTape() as tape:
            loss = self.__call__(*train_data)
        gradients = tape.gradient(loss, self.param)
        optimizer.apply_gradients(zip(gradients, self.param))
        train_loss(loss)
        return
      
      
    @tf.function
    def train_step_(self, train_data, train_loss, optimizer):
        with tf.GradientTape() as tape:
            loss = self.__call__(*train_data)
        gradients = tape.gradient(loss, self.param)
        optimizer.apply_gradients(zip(gradients, self.param))
        train_loss(loss)
        return
    
    
    def _train_step(self, train_data, optimizer):
        with tf.GradientTape() as tape:
            loss = self.__call__(*train_data)
            loss = self.compute_loss(loss)
        gradients = tape.gradient(loss, self.param)
        optimizer.apply_gradients(zip(gradients, self.param))
        return loss 
    
    
    @tf.function(jit_compile=True)
    def distributed_train_step(self, dataset_inputs, optimizer, strategy):
        per_replica_losses = strategy.run(self._train_step, args=(dataset_inputs, optimizer))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)
    
    
    @tf.function
    def distributed_train_step_(self, dataset_inputs, optimizer, strategy):
        per_replica_losses = strategy.run(self._train_step, args=(dataset_inputs, optimizer))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)
    
    
    def train1(self, train_loss, optimizer):
        if len(self.state_pool)<self.batch:
            if self.loss!=None:
                return self.loss
            else:
                if self.distributed_flag==True:
                    return np.array(0.)
                else:
                    return train_loss.result().numpy() 
        else:
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            if self.pr==True or self.HER==True:
                total_loss = 0.0
                num_batches = 0
                for j in range(batches):
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                    if self.distributed_flag==True:
                        if self.jit_compile==True:
                            total_loss+=self.distributed_train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.trategy)
                        else:
                            total_loss+=self.distributed_train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.trategy)
                        num_batches += 1
                        self.batch_counter+=1
                    else:
                        if self.jit_compile==True:
                            self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                        else:
                            self.train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                        self.batch_counter+=1
                if len(self.state_pool)%self.batch!=0:
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                    if self.distributed_flag==True:
                        if self.jit_compile==True:
                            total_loss+=self.distributed_train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.trategy)
                        else:
                            total_loss+=self.distributed_train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.trategy)
                        num_batches += 1
                        self.batch_counter+=1
                    else:
                        if self.jit_compile==True:
                            self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                        else:
                            self.train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                        self.batch_counter+=1
            else:
                if self.distributed_flag==True:
                    total_loss = 0.0
                    num_batches = 0
                    if self.mp_flag==True:
                        length=min(len(self.state_pool[7]),len(self.action_pool[7]),len(self.next_state_pool[7]),len(self.reward_pool[7]),len(self.done_pool[7]))
                        if self.shuffle!=True:
                            train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool[7][:length],self.action_pool[7][:length],self.next_state_pool[7][:length],self.reward_pool[7][:length],self.done_pool[7][:length])).batch(self.global_batch_size)
                        else:
                            train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool[7][:length],self.action_pool[7][:length],self.next_state_pool[7][:length],self.reward_pool[7][:length],self.done_pool[7][:length])).shuffle(len(self.state_pool)).batch(self.global_batch_size)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                    for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                        if self.jit_compile==True:
                            total_loss+=self.distributed_train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.trategy)
                        else:
                            total_loss+=self.distributed_train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],optimizer,self.trategy)
                        num_batches += 1
                        self.batch_counter+=1
                else:
                    if self.mp_flag==True:
                        length=min(len(self.state_pool[7]),len(self.action_pool[7]),len(self.next_state_pool[7]),len(self.reward_pool[7]),len(self.done_pool[7]))
                        if self.shuffle!=True:
                            train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool[7][:length],self.action_pool[7][:length],self.next_state_pool[7][:length],self.reward_pool[7][:length],self.done_pool[7][:length])).batch(self.global_batch_size)
                        else:
                            train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool[7][:length],self.action_pool[7][:length],self.next_state_pool[7][:length],self.reward_pool[7][:length],self.done_pool[7][:length])).shuffle(len(self.state_pool)).batch(self.global_batch_size)
                    else:
                        train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                    for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                        if self.jit_compile==True:
                            self.train_step([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                        else:
                            self.train_step_([state_batch,action_batch,next_state_batch,reward_batch,done_batch],train_loss,optimizer)
                        self.batch_counter+=1
            if self.update_steps!=None:
                if self.mp_flag==True:
                    if self.batch_counter%self.update_batches==0:
                        self.update_param()
                        if self.PPO:
                            self.state_pool=None
                            self.action_pool=None
                            self.next_state_pool=None
                            self.reward_pool=None
                            self.done_pool=None
                else:
                    if self.step_counter%self.update_steps==0:
                        self.update_param()
                        if self.PPO:
                            self.state_pool=None
                            self.action_pool=None
                            self.next_state_pool=None
                            self.reward_pool=None
                            self.done_pool=None
            else:
                self.update_param()
                if self.PPO:
                    self.state_pool=None
                    self.action_pool=None
                    self.next_state_pool=None
                    self.reward_pool=None
                    self.done_pool=None
        if self.distributed_flag==True:
            return (total_loss / num_batches).numpy()
        else:
            return train_loss.result().numpy()
    
    
    def train2(self, train_loss, optimizer):
        self.reward=0
        s=self.env_(initial=True)
        s=np.array(s)
        while True:
            s=np.expand_dims(s,axis=0)
            a=self.choose_action(s)
            next_s,r,done=self.env_(a)
            next_s=np.array(next_s)
            r=np.array(r)
            done=np.array(done)
            self.pool(s,a,next_s,r,done)
            if self.pr==True:
                self.prioritized_replay.TD=np.append(self.prioritized_replay.TD,self.initial_TD)
                if len(self.state_pool)>self.pool_size:
                    TD=np.array(0)
                    self.prioritized_replay.TD=np.append(TD,self.prioritized_replay.TD[2:])
            self.reward=r+self.reward
            loss=self.train1(train_loss,optimizer)
            self.step_counter+=1
            if done:
                self.reward_list.append(self.reward)
                if len(self.reward_list)>self.trial_count:
                    del self.reward_list[0]
                return loss
            s=next_s
    
    
    def store_in_parallel(self,p):
        self.reward[p]=0
        s=self.env_(initial=True,p=p)
        s=np.array(s)
        while True:
            s=np.expand_dims(s,axis=0)
            a=self.choose_action(s)
            next_s,r,done=self.env_(a,p=p)
            next_s=np.array(next_s)
            r=np.array(r)
            done=np.array(done)
            self.pool(s,a,next_s,r,done)
            self.reward[p]=r+self.reward[p]
            self.step_counter+=1
            if done:
                self.reward_list.append(self.reward[p])
                if len(self.reward_list)>self.trial_count:
                    del self.reward_list[0]
            s=next_s
    
    
    def fit(self, train_loss, optimizer, episodes=None, jit_compile=True, mp=None, manager=None, processes=None, shuffle=False, p=None):
        avg_reward=None
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if episodes%10!=0:
            p=episodes-episodes%self.p
            p=int(p/self.p)
        else:
            p=episodes/(self.p+1)
            p=int(p)
        if p==0:
            p=1
        self.mp_flag=False
        self.shuffle=shuffle
        if mp!=None:
            self.mp_flag=True
            self.state_pool=manager.dict({})
            self.action_pool=manager.dict({})
            self.next_state_pool=manager.dict({})
            self.reward_pool=manager.dict({})
            self.done_pool=manager.dict({})
            self.state_pool[7]=None
            self.action_pool[7]=None
            self.next_state_pool[7]=None
            self.reward_pool[7]=None
            self.done_pool[7]=None
            self.reward=np.zeros(processes,dtype='float32')
            self.reward=Array('f',self.reward)
            self.reward_list=manager.list([])
            self.step_counter=Value('i',0)
        self.distributed_flag=False
        self.optimizer_=optimizer
        self.episodes=episodes
        self.jit_compile=jit_compile
        if episodes!=None:
            for i in range(episodes):
                t1=time.time()
                train_loss.reset_states()
                if self.mp_flag==True:
                    process_list=[]
                    for p in range(processes):
                        process=mp.Process(target=self.store_in_parallel,args=(p,))
                        process.start()
                        process_list.append(process)
                    for process in process_list:
                        process.join()
                    loss=self.train1(train_loss, self.optimizer_)
                else:
                    loss=self.train2(train_loss,self.optimizer_)
                self.loss=loss
                self.loss_list.append(loss)
                self.total_episode+=1
                if self.path!=None and i%self.save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(self.path)
                    else:
                        self.save_(self.path)
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.4f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                train_loss.reset_states()
                if self.mp_flag==True:
                    process_list=[]
                    for p in range(processes):
                        process=mp.Process(target=self.store_in_parallel,args=(p,))
                        process.start()
                        process_list.append(process)
                    for process in process_list:
                        process.join()
                    loss=self.train1(train_loss, self.optimizer_)
                else:
                    loss=self.train2(train_loss,self.optimizer_)
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.total_episode+=1
                if self.path!=None and i%self.save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(self.path)
                    else:
                        self.save_(self.path)
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.4f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                t2=time.time()
                self.time+=(t2-t1)
        time_=self.time-int(self.time)
        if time_<0.5:
            self.total_time=int(self.time)
        else:
            self.total_time=int(self.time)+1
        self.total_time+=self.time
        print('last loss:{0:.4f}'.format(loss))
        print('last reward:{0}'.format(self.reward))
        print()
        print('time:{0}s'.format(self.time))
        return
    
    
    def distributed_fit(self, global_batch_size, optimizer, strategy, episodes=None, jit_compile=True, mp=None, manager=None, processes=None, shuffle=False, p=None):
        avg_reward=None
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if episodes%10!=0:
            p=episodes-episodes%self.p
            p=int(p/self.p)
        else:
            p=episodes/(self.p+1)
            p=int(p)
        if p==0:
            p=1
        self.mp_flag=False
        self.shuffle=shuffle
        if mp!=None:
            self.mp_flag=True
            self.state_pool=manager.dict({})
            self.action_pool=manager.dict({})
            self.next_state_pool=manager.dict({})
            self.reward_pool=manager.dict({})
            self.done_pool=manager.dict({})
            self.state_pool[7]=None
            self.action_pool[7]=None
            self.next_state_pool[7]=None
            self.reward_pool[7]=None
            self.done_pool[7]=None
            self.reward=np.zeros(processes,dtype='float32')
            self.reward=Array('f',self.reward)
            self.reward_list=manager.list([])
            self.step_counter=Value('i',0)
        self.distributed_flag=True
        self.global_batch_size=global_batch_size
        self.batch=global_batch_size
        self.optimizer_=optimizer
        self.strategy=strategy
        self.episodes=episodes
        self.jit_compile=jit_compile
        with strategy.scope():
            def compute_loss(self, per_example_loss):
                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        if episodes!=None:
            for i in range(episodes):
                t1=time.time()
                if self.mp_flag==True:
                    process_list=[]
                    for p in range(processes):
                        process=mp.Process(target=self.store_in_parallel,args=(p,))
                        process.start()
                        process_list.append(process)
                    for process in process_list:
                        process.join()
                    loss=self.train1(None, self.optimizer_)
                else:
                    loss=self.train2(None,self.optimizer_)
                self.loss=loss
                self.loss_list.append(loss)
                self.total_episode+=1
                if self.path!=None and i%self.save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(self.path)
                    else:
                        self.save_(self.path)
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.4f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                if self.mp_flag==True:
                    process_list=[]
                    for p in range(processes):
                        process=mp.Process(target=self.store_in_parallel,args=(p,))
                        process.start()
                        process_list.append(process)
                    for process in process_list:
                        process.join()
                    loss=self.train1(None, self.optimizer_)
                else:
                    loss=self.train2(None,self.optimizer_)
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.total_episode+=1
                if self.path!=None and i%self.save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(self.path)
                    else:
                        self.save_(self.path)
                if self.trial_count!=None:
                    if len(self.reward_list)>=self.trial_count:
                        avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                        if self.criterion!=None and avg_reward>=self.criterion:
                            t2=time.time()
                            self.total_time+=(t2-t1)
                            time_=self.total_time-int(self.total_time)
                            if time_<0.5:
                                self.total_time=int(self.total_time)
                            else:
                                self.total_time=int(self.total_time)+1
                            print('episode:{0}'.format(self.total_episode))
                            print('last loss:{0:.4f}'.format(loss))
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.4f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                t2=time.time()
                self.time+=(t2-t1)
        time_=self.time-int(self.time)
        if time_<0.5:
            self.total_time=int(self.time)
        else:
            self.total_time=int(self.time)+1
        self.total_time+=self.time
        print('last loss:{0:.4f}'.format(loss))
        print('last reward:{0}'.format(self.reward))
        print()
        print('time:{0}s'.format(self.time))
        return
    
    
    def run_agent(self, max_steps, seed=None):
        state_history = []

        steps = 0
        reward_ = 0
        if seed==None:
            state = self.env.reset()
        else:
            state = self.env.reset(seed=seed)
        for step in range(max_steps):
            if not hasattr(self, 'noise'):
                action = np.argmax(self.action(state))
            else:
                action = self.action(state).numpy()
            next_state, reward, done, _ = self.env.step(action)
            state_history.append(state)
            steps+=1
            reward_+=reward
            if done:
                break
            state = next_state
        
        return state_history,reward_,steps
    
    
    def animate_agent(self, max_steps, mode='rgb_array', save_path=None, fps=None, writer='imagemagick'):
        state_history,reward,steps = self.run_agent(max_steps)
        
        fig = plt.figure()
        ax = fig.add_subplot()
        self.env.reset()
        img = ax.imshow(self.env.render(mode=mode))

        def update(frame):
            img.set_array(self.env.render(mode=mode))
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=state_history, blit=True)
        plt.show()
        
        print('steps:{0}'.format(steps))
        print('reward:{0}'.format(reward))
        
        if save_path!=None:
            ani.save(save_path, writer=writer, fps=fps)
        return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('reward:{0:.4f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('loss:{0:.4f}'.format(self.loss_list[-1]))
        return
    
    
    def visualize_reward_loss(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list,'r-',label='reward')
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list,'b-',label='train loss')
        plt.xlabel('epoch')
        plt.ylabel('reward and loss')
        plt.xticks(np.arange(1,self.total_epoch+1))
        plt.show()
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
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save_param(path)
                        self.avg_reward=avg_reward
        return
    
    
    def save_param(self,path):
        output_file=open(path,'wb')
        pickle.dump(self.param,output_file)
        output_file.close()
        return
    
    
    def restore_param(self,path):
        input_file=open(path,'rb')
        param=pickle.load(input_file)
        nn.assign(self.param,param)
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
            optimizer_config=tf.keras.optimizers.serialize(self.optimizer_)
            self.optimizer_=None
            pickle.dump(self,output_file)
            pickle.dump(optimizer_config,output_file)
            output_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save(path)
                        self.avg_reward=avg_reward
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
