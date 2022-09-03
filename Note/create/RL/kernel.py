from tensorflow import data as tf_data
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle


class kernel:
    def __init__(self,nn=None,thread=None,thread_lock=None,state=None,state_name=None,action_name=None,save_episode=True):
        self.nn=nn
        try:
            self.nn.km=1
        except AttributeError:
            pass
        if thread!=None:
            self.state_list=np.array(0,dtype='int8')
            self.threadnum=np.arange(thread)
            self.threadnum=list(self.threadnum)
            self.reward=np.zeros(thread)
            self.loss=np.zeros(thread)
            self.step_counter=np.zeros(thread)
            self.episode_num=np.zeros(thread)
        self.state_pool=[]
        self.action_pool=[]
        self.next_state_pool=[]
        self.reward_pool=[]
        self.done_pool=[]
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.epsilon=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.suspend=False
        self.stop=None
        self.save_flag=None
        self.stop_flag=1
        self.end_loss=None
        self.thread=thread
        self.thread_counter=0
        self.thread_lock=thread_lock
        self.p=[]
        self._state_list=[]
        self.finish_list=[]
        self.PN=True
        self.save_episode=save_episode
        self.reward_list=[]
        self.loss_list=[]
        self.total_time=0
    
    
    def action_init(self,action_num=None,dtype=np.int32):
        if self.action_name!=None:
            self.action_num=len(self.action_name)
        else:
            action_num=self.action_num
            self.action_num=action_num
        if self.action_name!=None and len(self.action_name)>self.action_num:
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_num,dtype=dtype)))
        elif self.action_name!=None:
            if self.epsilon!=None:
                self.action_one=np.ones(len(self.action_name),dtype=dtype)
        if self.action_num>action_num:
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(self.action_num-action_num,dtype=dtype)))
        else:
            if self.epsilon!=None:
                self.action_one=np.ones(self.action_num,dtype=dtype)
        return
    
    
    def add_threads(self,thread):
        threadnum=np.arange(thread)+self.thread
        self.threadnum=self.threadnum.extend(threadnum)
        self.thread+=thread
        self.step_counter=np.concatenate((self.step_counter,np.zeros(thread)))
        self.reward=np.concatenate((self.reward,np.zeros(thread)))
        self.loss=np.concatenate((self.loss,np.zeros(thread)))
        self.episode_num=np.concatenate((self.episode_num,np.zeros(thread)))
        return
    
    
    def set_up(self,epsilon=None,episode_step=None,pool_size=None,batch=None,update_step=None,trial_num=None,criterion=None,end_loss=None,init=None):
        if type(epsilon)!=list and epsilon!=None:
            self.epsilon=np.ones(self.thread)*epsilon
        elif epsilon==None:
            self.epsilon=np.zeros(self.thread)
        else:
            self.epsilon=epsilon
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
        if update_step!=None:
            self.update_step=update_step
        if trial_num!=None:
            self.trial_num=trial_num
        if criterion!=None:
            self.criterion=criterion
        if end_loss!=None:
            self.end_loss=end_loss
        if init==True:
            self.p=[]
            self._state_list=[]
            self.finish_list=[]
            self.state_list=np.array(0,dtype='int8')
            self.PN=True
            self.episode=[]
            self.epsilon=[]
            self.state_pool=[]
            self.action_pool=[]
            self.next_state_pool=[]
            self.reward_pool=[]
            self.loss=np.zeros(self.thread)
            self.loss_list=[]
            self.step_counter=np.zeros(self.thread)
            self.episode_num=np.zeros(self.thread)
            self.total_episode=0
            self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one,epsilon):
        action_prob=action_one*epsilon/len(action_one)
        if self.state==None:
            best_a=np.argmax(self.nn.nn(s))
        else:
            best_a=np.argmax(self.nn.nn(self.state[self.state_name[s]]))
        action_prob[best_a.numpy()]+=1-epsilon
        return action_prob
    
    
    def pool(self,s,a,next_s,r,done,i,index):
        if self.PN==True:
            self.thread_lock[0].acquire()
            if self.state_pool[index]==None and type(self.state_pool[index])!=np.ndarray:
                if self.state==None:
                    self.state_pool[index]=s
                    if len(a.shape)==1:
                        self.action_pool[index]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[index]=a
                    self.next_state_pool[index]=np.expand_dims(next_s,axis=0)
                    self.reward_pool[index]=np.expand_dims(r,axis=0)
                    self.done_pool[index]=np.expand_dims(done,axis=0)
                else:
                    self.state_pool[index]=np.expand_dims(self.state[self.state_name[s]],axis=0)
                    if len(a.shape)==1:
                        self.action_pool[index]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[index]=a
                    self.next_state_pool[index]=np.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool[index]=np.expand_dims(r,axis=0)
                    self.done_pool[index]=np.expand_dims(done,axis=0)
            else:
                try:
                    if self.state==None:
                        self.state_pool[index]=np.concatenate((self.state_pool[index],s),0)
                        if len(a.shape)==1:
                            self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                        else:
                            self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                        self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(next_s,axis=0)),0)
                        self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                        self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                    else:
                        self.state_pool[index]=np.concatenate((self.state_pool[index],np.expand_dims(self.state[self.state_name[s]],axis=0)),0)
                        if len(a.shape)==1:
                            self.action_pool[index]=np.concatenate((self.action_pool[index],np.expand_dims(a,axis=0)),0)
                        else:
                            self.action_pool[index]=np.concatenate((self.action_pool[index],a),0)
                        self.next_state_pool[index]=np.concatenate((self.next_state_pool[index],np.expand_dims(self.state[self.state_name[next_s]],axis=0)),0)
                        self.reward_pool[index]=np.concatenate((self.reward_pool[index],np.expand_dims(r,axis=0)),0)
                        self.done_pool[index]=np.concatenate((self.done_pool[index],np.expand_dims(done,axis=0)),0)
                except:
                    pass
            self.thread_lock[0].release()
        else:
            if self.state_pool[i]==None and type(self.state_pool[i])!=np.ndarray:
                if self.state==None:
                    self.state_pool[i]=s
                    if len(a.shape)==1:
                        self.action_pool[i]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[i]=a
                    self.next_state_pool[i]=np.expand_dims(next_s,axis=0)
                    self.reward_pool[i]=np.expand_dims(r,axis=0)
                    self.done_pool[i]=np.expand_dims(done,axis=0)
                else:
                    self.state_pool[i]=np.expand_dims(self.state[self.state_name[s]],axis=0)
                    if len(a.shape)==1:
                        self.action_pool[i]=np.expand_dims(a,axis=0)
                    else:
                        self.action_pool[i]=a
                    self.next_state_pool[i]=np.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool[i]=np.expand_dims(r,axis=0)
                    self.done_pool[i]=np.expand_dims(done,axis=0)
            else:
                if self.state==None:
                    self.state_pool[i]=np.concatenate((self.state_pool[i],s),0)
                    if len(a.shape)==1:
                        self.action_pool[i]=np.concatenate((self.action_pool[i],np.expand_dims(a,axis=0)),0)
                    else:
                        self.action_pool[i]=np.concatenate((self.action_pool[i],a),0)
                    self.next_state_pool[i]=np.concatenate((self.next_state_pool[i],np.expand_dims(next_s,axis=0)),0)
                    self.reward_pool[i]=np.concatenate((self.reward_pool[i],np.expand_dims(r,axis=0)),0)
                    self.done_pool[i]=np.concatenate((self.done_pool[i],np.expand_dims(done,axis=0)),0)
                else:
                    self.state_pool[i]=np.concatenate((self.state_pool[i],np.expand_dims(self.state[self.state_name[s]],axis=0)),0)
                    if len(a.shape)==1:
                        self.action_pool[i]=np.concatenate((self.action_pool[i],np.expand_dims(a,axis=0)),0)
                    else:
                        self.action_pool[i]=np.concatenate((self.action_pool[i],a),0)
                    self.next_state_pool[i]=np.concatenate((self.next_state_pool[i],np.expand_dims(self.state[self.state_name[next_s]],axis=0)),0)
                    self.reward_pool[i]=np.concatenate((self.reward_pool[i],np.expand_dims(r,axis=0)),0)
                    self.done_pool[i]=np.concatenate((self.done_pool[i],np.expand_dims(done,axis=0)),0)
        if self.state_pool[i]!=None and len(self.state_pool[i])>self.pool_size:
            self.state_pool[i]=self.state_pool[i][1:]
            self.action_pool[i]=self.action_pool[i][1:]
            self.next_state_pool[i]=self.next_state_pool[i][1:]
            self.reward_pool[i]=self.reward_pool[i][1:]
            self.done_pool[i]=self.done_pool[i][1:]
        return
    
    
    def explore(self,s,epsilon,i):
        try:
            if self.nn.nn!=None:
                pass
            try:
                if self.nn.explore!=None:
                    pass
                s=np.expand_dims(s,axis=0)
                if self.epsilon==None:
                    self.epsilon[i]=self.nn.epsilon(self.step_counter[i],i)
                try:
                    if self.nn.action!=None:
                        pass
                    a=self.nn.action(s)
                except AttributeError:
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action_num,p=action_prob)
                if self.action_name==None:
                    next_s,r,done=self.nn.explore(a)
                else:
                    next_s,r,done=self.nn.explore(self.action_name[a])
            except AttributeError:
                if self.epsilon==None:
                    self.epsilon[i]=self.nn.epsilon(self.step_counter[i],i)
                try:
                    if self.nn.action!=None:
                        pass
                    a=self.nn.action(s)
                except AttributeError:
                    action_prob=self.epsilon_greedy_policy(s,self.action_one)
                    a=np.random.choice(self.action_num,p=action_prob)
                next_s,r,done=self.nn.transition(self.state_name[s],self.action_name[a])
        except AttributeError:
            try:
                if self.nn.explore!=None:
                    pass 
                if self.state_name==None:
                    s=np.expand_dims(s,axis=0)
                    a=(self.nn.actor(s)+self.nn.noise()).numpy()
                else:
                    a=(self.nn.actor(self.state[self.state_name[s]])+self.nn.noise()).numpy()
                next_s,r,done=self.nn.explore(a)
            except AttributeError:
                a=(self.nn.actor(self.state[self.state_name[s]])+self.nn.noise()).numpy()
                next_s,r,done=self.nn.transition(self.state_name[s],a)
        if self.PN==True:
            while len(self._state_list)<i:
                pass
            if len(self._state_list)==i:
                self.thread_lock[2].acquire()
                self._state_list.append(self.state_list[1:])
                self.thread_lock[2].release()
            else:
                if len(self._state_list[i])<self.thread_counter:
                    self._state_list[i]=self.state_list[1:]
            while len(self.p)<i:
                pass
            if len(self.p)==i:
                self.thread_lock[2].acquire()
                self.p.append(np.array(self._state_list[i],dtype=np.float16)/np.sum(self._state_list[i]))
                self.thread_lock[2].release()
            else:
                if len(self.p[i])<self.thread_counter:
                    self.p[i]=np.array(self._state_list[i],dtype=np.float16)/np.sum(self._state_list[i])
            while True:
                index=np.random.choice(len(self.p[i]),p=self.p[i])
                if index in self.finish_list:
                    continue
                else:
                    break
        else:
            index=None
        self.pool(s,a,next_s,r,done,i,index)
        if self.save_episode==True:
            if self.state_name==None and self.action_name==None:
                episode=[s,a,next_s,r]
            elif self.state_name==None:
                episode=[s,self.action_name[a],next_s,r]
            elif self.action_name==None:
                episode=[self.state_name[s],a,self.state_name[next_s],r]
            else:
                episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
        return next_s,r,done,episode,index
    
        
    def get_episode(self,s):
        next_s=None
        episode=[]
        self.end_flag=False
        while True:
            s=next_s
            try:
                if self.nn.nn!=None:
                    pass
                try:
                    if self.nn.explore!=None:
                        pass
                    s=np.expand_dims(s,axis=0)
                    a=np.argmax(self.nn.nn(s)).numpy()
                    if self.action_name==None:
                        next_s,r,done=self.nn.explore(a)
                    else:
                        next_s,r,done=self.nn.explore(self.action_name[a])
                except AttributeError:
                    a=np.argmax(self.nn.nn(s))
                    next_s,r,done=self.nn.transition(self.state_name[s],self.action_name[a])
            except AttributeError:
                try:
                    if self.nn.explore!=None:
                        pass
                    if self.state_name==None:
                        s=np.expand_dims(s,axis=0)
                        a=self.nn.actor(s).numpy()
                    else:
                        a=self.nn.actor(self.state[self.state_name[s]]).numpy()
                    next_s,r,done=self.nn.explore(a)
                except AttributeError:
                    a=self.nn.actor(self.state[self.state_name[s]]).numpy()
                    next_s,r,done=self.nn.transition(self.state_name[s],a)
            if done:
                if self.state_name==None and self.action_name==None:
                    episode.append([s,a,next_s,r])
                elif self.state_name==None:
                    episode.append([s,self.action_name[a],next_s,r])
                elif self.action_name==None:
                    episode.append([self.state_name[s],a,self.state_name[next_s],r])
                else:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
                episode.append('done')
                break
            elif self.end_flag==True:
                break
            else:
                if self.state_name==None and self.action_name==None:
                    episode.append([s,a,next_s,r])
                elif self.state_name==None:
                    episode.append([s,self.action_name[a],next_s,r])
                elif self.action_name==None:
                    episode.append([self.state_name[s],a,self.state_name[next_s],r])
                else:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
        return episode
    
    
    
    def end(self):
        if self.end_loss!=None and self.loss_list[-1]<=self.end_loss:
            return True
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        self.thread_lock[1].acquire()
        self.nn.opt(loss)
        self.thread_lock[1].release()
        return loss
    
    
    def opt_t(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        return loss
    
    
    def _train(self,i,j=None,batches=None,length=None):
        if len(self.state_pool[i])<self.batch:
            self.suspend_func()
            return
        else:
            self.suspend_func()
            if length%self.batch!=0:
                try:
                    if self.nn.data_func!=None:
                        pass
                    state_batch,action_batch,next_state_batch,reward_batch=self.nn.data_func(self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i],self.pool_size,self.batch,self.nn.rp,self.nn.alpha,self.nn.beta)
                except AttributeError:
                    batches+=1
                    index1=batches*self.batch
                    index2=self.batch-(length-batches*self.batch)
                    state_batch=np.concatenate((self.state_pool[i][index1:length],self.state_pool[i][:index2]),0)
                    action_batch=np.concatenate((self.action_pool[i][index1:length],self.action_pool[i][:index2]),0)
                    next_state_batch=np.concatenate((self.next_state_pool[i][index1:length],self.next_state_pool[i][:index2]),0)
                    reward_batch=np.concatenate((self.reward_pool[i][index1:length],self.reward_pool[i][:index2]),0)
                    done_batch=np.concatenate((self.done_pool[i][index1:length],self.done_pool[i][:index2]),0)
                loss=self.opt_t(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                self.loss[i]+=loss
                try:
                    self.nn.bc[i]+=1
                except AttributeError:
                    pass
                return
            try:
                if self.nn.data_func!=None:
                    pass
                state_batch,action_batch,next_state_batch,reward_batch=self.nn.data_func(self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i],self.pool_size,self.batch,self.nn.rp,self.nn.alpha,self.nn.beta)
            except AttributeError:
                index1=j*self.batch
                index2=(j+1)*self.batch
                state_batch=self.state_batch[i][index1:index2]
                action_batch=self.action_batch[i][index1:index2]
                next_state_batch=self.next_state_batch[i][index1:index2]
                reward_batch=self.reward_batch[i][index1:index2]
                done_batch=self.done_batch[i][index1:index2]
                loss=self.opt_t(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                self.loss[i]+=loss
            try:
                self.nn.bc[i]=j
            except AttributeError:
                pass
        return
    
    
    def train_(self,i):
        train_ds=tf_data.Dataset.from_tensor_slices((self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i])).shuffle(len(self.state_pool[i])).batch(self.batch)
        for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
            if self.stop==True:
                if self.stop_func() or self.stop_flag==0:
                    return
            self.suspend_func()
            loss=self.opt_t(state_batch,action_batch,next_state_batch,reward_batch)
            self.loss[i]+=loss
            try:
                self.nn.bc[i]+=1
            except AttributeError:
                pass
        return
            
    
    def _train_(self,i):
        if len(self.state_pool[i])<self.batch:
            self._train(i)
        else:
            self.loss[i]=0
            if self.PN==True:
                length=min(len(self.state_pool[i]),len(self.action_pool[i]),len(self.next_state_pool[i]),len(self.reward_pool[i]))
                batches=int((length-length%self.batch)/self.batch)
                if length%self.batch!=0:
                    batches+=1
                for j in range(batches):
                    if self.stop==True:
                        if self.stop_func() or self.stop_flag==0:
                            return
                    self._train(i,j,batches,length)
            else:
                try:
                    self.nn.bc[i]=0
                except AttributeError:
                    pass
                self.train_(i)
            self.thread_lock[2].acquire()
            if self.update_step!=None:
                if self.step_counter[i]%self.update_step==0:
                    self.nn.update_param()
            else:
                self.nn.update_param()
            self.thread_lock[2].release()
            self.loss[i]=self.loss[i]/batches
        self.step_counter[i]+=1
        try:
            self.nn.ec[i]+=1
        except AttributeError:
            pass
        return
    
    
    def train(self,episode_num):
        try:
            i=self.threadnum.pop(0)
        except IndexError:
            print('\nError,please add thread.')
            return
        while self.state_pool!=None and len(self.state_pool)<i:
            pass
        if self.state_pool!=None and len(self.state_pool)==i:
            self.thread_lock[3].acquire()
            self.state_pool.append(None)
            self.action_pool.append(None)
            self.next_state_pool.append(None)
            self.reward_pool.append(None)
            try:
                self.nn.ec.append(0)
            except AttributeError:
                pass
            try:
                self.nn.bc.append(0)
            except AttributeError:
                pass
            if type(self.state_list)==np.ndarray:
                self.state_list=np.append(self.state_list,np.array(1,dtype='int8'))
            self.thread_counter+=1
            self.thread_lock[3].release()
        for k in range(episode_num):
            print(self.index_matrix)
            if self.stop==True:
                if self.stop_func() or self.stop_flag==0:
                    return
            self.episode_num[i]+=1
            episode=[]
            if self.state_name==None:
                s=self.nn.explore(init=True)
            else:
                s=int(np.random.uniform(0,len(self.state_name)))
            if self.episode_step==None:
                while True:
                    if self.stop==True:
                        if self.stop_func() or self.stop_flag==0:
                            return
                    try:
                        epsilon=self.epsilon[i]
                    except:
                        pass
                    next_s,r,done,_episode,index=self.explore(s,epsilon,i)
                    self.reward[i]+=r
                    s=next_s
                    if self.state_pool[i]!=None and self.action_pool[i]!=None and self.next_state_pool[i]!=None and self.reward_pool[i]!=None:
                        self._train_(i)
                    if self.stop_flag==0:
                        return
                    if self.save_episode==True:
                        try:
                            if index not in self.finish_list:
                                episode.append(_episode)
                        except UnboundLocalError:
                            pass
                    if done:
                        self.thread_lock[3].acquire()
                        self.loss_list.append(self.loss[i])
                        self.thread_lock[3].release()
                        if self.save_episode==True:
                            episode.append('done')
                        break
            else:
                for _ in range(self.episode_step):
                    if self.stop==True:
                        if self.stop_func() or self.stop_flag==0:
                            return
                    try:
                        epsilon=self.epsilon[i]
                    except:
                        pass
                    next_s,r,done,episode,index=self.explore(s,epsilon,i)
                    self.reward[i]+=r
                    s=next_s
                    if self.state_pool[i]!=None and self.action_pool[i]!=None and self.next_state_pool[i]!=None and self.reward_pool[i]!=None:
                        self._train_(i)
                    if self.stop_flag==0:
                        return
                    if self.save_episode==True:
                        try:
                            if index not in self.finish_list:
                                episode.append(_episode)
                        except UnboundLocalError:
                            pass
                    if done:
                        self.thread_lock[3].acquire()
                        self.loss_list.append(self.loss[i])
                        self.thread_lock[3].release()
                        if self.save_episode==True:
                            episode.append('done')
                        break
            self.thread_lock[3].acquire()
            self.reward_list.append(self.reward[i])
            self.reward[i]=0
            if self.save_episode==True:
                self.episode.append(episode)
            self.thread_lock[3].release()
        self.thread_lock[3].acquire()
        if i not in self.finish_list:
            self.finish_list.append(i)
        self.thread-=1
        self.thread_lock[3].release()
        if self.PN==True:
            self.state_pool[i]=None
            self.action_pool[i]=None
            self.next_state_pool[i]=None
            self.reward_pool[i]=None
        return
    
    
    def suspend_func(self):
        if self.suspend==True:
            while True:
                if self.suspend==False:
                    break
        return
    
    
    def stop_func(self):
        if self.trial_num!=None:
            if len(self.reward_list)>=self.trial_num:
                avg_reward=statistics.mean(self.reward_list[-self.trial_num:])
                if self.criterion!=None and avg_reward>=self.criterion:
                    self.thread_lock[4].acquire()
                    self.save(self.total_episode,True)
                    self.save_flag=True
                    self.thread_lock[4].release()
                    self.stop_flag=0
                    return True
        elif self.end():
            self.thread_lock[4].acquire()
            self.save(self.total_episode,True)
            self.save_flag=True
            self.thread_lock[4].release()
            self.stop_flag=0
            return True
        elif self.stop_flag==1:
            self.stop_flag=0
            return True
        return False
    
    
    def reward_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.reward_list)),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
        return
    
    
    def train_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(len(self.loss_list)),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def save_e(self):
        episode_file=open('episode.dat','wb')
        pickle.dump(self.episode,episode_file)
        episode_file.close()
        return
    
    
    def save(self):
        if self.save_flag==True:
            return
        output_file=open('save.dat','wb')
        if self.save_episode==True:
            episode_file=open('episode.dat','wb')
            pickle.dump(self.episode,episode_file)
            episode_file.close()
        pickle.dump(self.nn,output_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.done_pool,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.step_counter,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.thread_counter,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.state_list,output_file)
        pickle.dump(self._state_list,output_file)
        pickle.dump(self.finish_list,output_file)
        pickle.dump(self.PN,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.reward_list,output_file)
        pickle.dump(self.loss_list,output_file)
        output_file.close()
        if self.save_flag==True:
            print('\nSystem have stopped,Neural network have saved.')
        return
    
    
    def restore(self,s_path,e_path=None):
        input_file=open(s_path,'rb')
        if e_path!=None:
            episode_file=open(e_path,'rb')
            self.episode=pickle.load(episode_file)
            episode_file.close()
        self.nn=pickle.load(input_file)
        try:
            self.nn.km=1
        except AttributeError:
            pass
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.done_pool=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_one=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.step_counter=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.thread_counter=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.state_list=pickle.load(input_file)
        self._state_list=pickle.load(input_file)
        self.finish_list=pickle.load(input_file)
        self.PN=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.reward_list=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        input_file.close()
        return
