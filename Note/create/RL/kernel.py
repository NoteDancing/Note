import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle


class kernel:
    def __init__(self,nn=None,state=None,state_name=None,action_name=None,thread=None,thread_lock=None,save_episode=True):
        if nn!=None:
            self.nn=nn
            self.opt=nn.opt
            try:
                self.nn.km=1
            except AttributeError:
                pass
        self.state_pool=[]
        self.action_pool=[]
        self.next_state_pool=[]
        self.reward_pool=[]
        self.episode=[]
        self.state=state
        self.state_name=state_name
        self.action_name=action_name
        self.epsilon=[]
        self.discount=None
        self.episode_step=None
        self.pool_size=None
        self.batch=None
        self.update_step=None
        self.end_loss=None
        self.thread=thread
        self.t=list(-np.arange(-self.thread,1))
        self.thread_lock=thread_lock
        self.state_list=None
        self._state_list=[]
        self.p=[]
        self.im=[]
        self.om=[]
        self.imr=[]
        self.rank_flag=False
        self.row_one=None
        self.rank_one=None
        self.row_p=[]
        self.d_index=0
        self.finish_list=[]
        if nn!=None:
            try:
                if self.nn.row!=None:
                    pass
                self.row_one=np.array(0,dtype='int8')
                self.rank_one=np.array(0,dtype='int8')
            except AttributeError:
                self.state_list=np.array(0,dtype='int8')
        self.pool_net=True
        self.PO=None
        self.save_episode=save_episode
        self.loss=np.zeros(self.thread)
        self._loss=[]
        self.loss_list=[]
        self.a=0
        self.epi_num=[]
        self.episode_num=np.zeros(self.thread)
        self.total_episode=0
        self.total_time=0
        
        
    def init(self,dtype=np.int32):
        self.action_len=len(self.action_name)
        if len(self.action_name)>self.action_len:
            self.action=np.concatenate((self.action,np.arange(len(self.action_name)-self.action_len,dtype=dtype)+self.action_len))
            if self.epsilon!=None:
                self.action_one=np.concatenate((self.action_one,np.ones(len(self.action_name)-self.action_len,dtype=dtype)))
        else:
            self.action=np.arange(len(self.action_name),dtype=dtype)
            if self.epsilon!=None:
                self.action_one=np.ones(len(self.action_name),dtype=dtype)
        return
    
    
    def add_threads(self,thread):
        t=-np.arange(-thread,1)+self.thread+1
        self.t=t.extend(self.t)
        self.thread+=thread
        if self.PO!=True:
            self.loss=np.concatenate((self.train_loss,np.zeros(thread)))
            self.loss_list.extend([[] for _ in range(thread)])
        self.episode_num=np.concatenate((self.epoch,np.zeros(thread)))
        return
    
    
    def set_up(self,param=None,discount=None,episode_num=None,episode_step=None,pool_size=None,batch=None,update_step=None,end_loss=None):
        if param!=None:
            self.nn.param=param
        if discount!=None:
            self.discount=discount
        if episode_num!=None:
            self.epi_num=episode_num
        if episode_step!=None:
            self.episode_step=episode_step
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
            self.index=np.arange(batch)
        if update_step!=None:
            self.update_step=update_step
        if end_loss!=None:
            self.end_loss=end_loss
        self.thread=0
        self._state_list=[]
        self.p=[]
        self.im=[]
        self.imr=[]
        self.rank_flag=False
        self.row_p=[]
        self.rank_one=[]
        self.d_index=0
        self.finish_list=[]
        try:
            if self.nn.row!=None:
                pass
            self.row_one=np.array(0,dtype='int8')
            self.rank_one=np.array(0,dtype='int8')
        except AttributeError:
            self.state_list=np.array(0,dtype='int8')
        self.pool_net=True
        self.episode=[]
        self.epsilon=[]
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.loss=np.zeros(self.thread)
        self._loss=[]
        self.loss_list=[]
        self.a=0
        self.epi_num=[]
        self.episode_num=np.zeros(self.thread)
        self.total_episode=0
        self.total_time=0
        return
    
    
    def epsilon_greedy_policy(self,s,action_one,epsilon):
        action_prob=action_one
        action_prob=action_prob*epsilon/len(action_one)
        if self.state==None:
            best_a=np.argmax(self.nn.nn(s))
        else:
            best_a=np.argmax(self.nn.nn(self.state[self.state_name[s]]))
        action_prob[best_a]+=1-epsilon
        return action_prob
    
    
    def _epsilon_greedy_policy(self,a,action_one):
        action_prob=action_one
        action_prob=action_prob*self.epsilon/len(action_one)
        best_a=np.argmax(a)
        action_prob[best_a]+=1-self.epsilon
        return action_prob
    
    
    def _explore(self,s,epsilon,i):
        if type(self.nn.nn)!=list:
            try:
                if self.nn.explore!=None:
                    pass
                action_prob=self.epsilon_greedy_policy(s,self.action_one)
                a=np.random.choice(self.action,p=action_prob)
                if self.action_name==None:
                    next_s,r,end=self.nn.explore(a)
                else:
                    next_s,r,end=self.nn.explore(self.action_name[a])
            except AttributeError:
                action_prob=self.epsilon_greedy_policy(s,self.action_one,epsilon)
                a=np.random.choice(self.action,p=action_prob)
                next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
        else:
            try:
                if self.nn.explore!=None:
                    pass
                if len(self.nn.param)==4:
                    if self.state_name==None:
                        a=(self.nn.nn[1](s,p=1)+tf.random.normal([1])).numpy()
                    else:
                        a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+tf.random.normal([1])).numpy()
                else:
                    if self.state_name==None:
                        a=self.nn.nn[1](s).numpy()
                    else:
                        a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                if len(a.shape)>0:
                    a=self._epsilon_greedy_policy(a,self.action_one)
                    next_s,r,end=self.nn.explore(self.action_name[a])
                else:
                    next_s,r,end=self.nn.explore(a)
            except AttributeError:
                if len(self.nn.param)==4:
                    a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+tf.random.normal([1])).numpy()
                else:
                    a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                if len(a.shape)>0:
                    a=self._epsilon_greedy_policy(a,self.action_one)
                    next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
                else:
                    next_s,r,end=self.nn.transition(self.state_name[s],a)
        if self.pool_net==True:
            if self.state_list==None:
                while True:
                    row_sum=np.sum(self.row_one)
                    row_len=len(self.row_one)
                    self.row_p[i]=self.row_one[:row_len]/row_sum
                    row_index=np.random.choice(len(self.row_one),p=self.row_p[i])-1
                    rank_sum=np.sum(self.om[row_index])
                    rank_len=len(self.om[row_index])
                    if rank_sum==0:
                        self.row_one[row_index]=0
                        break
                    rank_index=np.random.choice(len(self.om[row_index][:rank_len]),p=self.om[row_index][:rank_len]/rank_sum)
                    index=self.im[row_index][rank_index]
                    if index in self.finish_list:
                        self.om[row_index][rank_index]=0
                        continue
                    else:
                        break
            else:
                while len(self._state_list)<i:
                    pass
                if len(self._state_list)==i:
                    self.thread_lock.acquire()
                    self._state_list.append(self.state_list[1:])
                    self.thread_lock.release()
                else:
                    if len(self._state_list[i])<self.thread_sum:
                        self._state_list[i]=self.state_list[1:]
                while len(self.p)<i:
                    pass
                if len(self.p)==i:
                    self.thread_lock.acquire()
                    self.p.append(np.array(self._state_list[i],dtype=np.float16)/np.sum(self._state_list[i]))
                    self.thread_lock.release()
                else:
                    if len(self.p[i])<self.thread_sum:
                        self.p[i]=np.array(self._state_list[i],dtype=np.float16)/np.sum(self._state_list[i])
                while True:
                    index=np.random.choice(len(self.p[i]),p=self.p[i])
                    if index in self.finish_list:
                        continue
                    else:
                        break
        if self.pool_net==True:
            self.thread_lock.acquire()
            if len(self.state_pool[index])>self.pool_size:
                self.thread_lock.acquire()
                self.state_pool[index]=self.state_pool[index][1:]
                self.action_pool[index]=self.action_pool[index][1:]
                self.next_state_pool[index]=self.next_state_pool[index][1:]
                self.reward_pool[index]=self.reward_pool[index][1:]
            if self.state_pool[index]==None:
                if self.state==None:
                    self.state_pool[index]=tf.expand_dims(s,axis=0)
                    self.action_pool[index]=tf.expand_dims(a,axis=0)
                    self.next_state_pool[index]=tf.expand_dims(next_s,axis=0)
                    self.reward_pool[index]=tf.expand_dims(r,axis=0)
                else:
                    self.state_pool[index]=tf.expand_dims(self.state[self.state_name[s]],axis=0)
                    self.action_pool[index]=tf.expand_dims(a,axis=0)
                    self.next_state_pool[index]=tf.expand_dims(self.state[self.state_name[next_s]],axis=0)
                    self.reward_pool[index]=tf.expand_dims(r,axis=0)
            else:
                try:
                    if self.state==None:
                        self.state_pool[index]=tf.concat([self.state_pool[index],tf.expand_dims(s,axis=0)],0)
                        self.action_pool[index]=tf.concat([self.action_pool[index],tf.expand_dims(a,axis=0)],0)
                        self.next_state_pool[index]=tf.concat([self.next_state_pool[index],tf.expand_dims(next_s,axis=0)],0)
                        self.reward_pool[index]=tf.concat([self.reward_pool[index],tf.expand_dims(r,axis=0)],0)
                    else:
                        self.state_pool[index]=tf.concat([self.state_pool[index],tf.expand_dims(self.state[self.state_name[s]],axis=0)],0)
                        self.action_pool[index]=tf.concat([self.action_pool[index],tf.expand_dims(a,axis=0)],0)
                        self.next_state_pool[index]=tf.concat([self.next_state_pool[index],tf.expand_dims(self.state[self.state_name[next_s]],axis=0)],0)
                        self.reward_pool[index]=tf.concat([self.reward_pool[index],tf.expand_dims(r,axis=0)],0)
                except:
                    pass
            self.thread_lock.release()
        else:
            if self.state==None:
                self.state_pool[i]=tf.concat([self.state_pool[i],tf.expand_dims(s,axis=0)],0)
                self.action_pool[i]=tf.concat([self.action_pool[i],tf.expand_dims(a,axis=0)],0)
                self.next_state_pool[i]=tf.concat([self.next_state_pool[i],tf.expand_dims(next_s,axis=0)],0)
                self.reward_pool[i]=tf.concat([self.reward_pool[i],tf.expand_dims(r,axis=0)],0)
            else:
                self.state_pool[i]=tf.concat([self.state_pool[i],tf.expand_dims(self.state[self.state_name[s]],axis=0)],0)
                self.action_pool[i]=tf.concat([self.action_pool[i],tf.expand_dims(a,axis=0)],0)
                self.next_state_pool[i]=tf.concat([self.next_state_pool[i],tf.expand_dims(self.state[self.state_name[next_s]],axis=0)],0)
                self.reward_pool[i]=tf.concat([self.reward_pool[i],tf.expand_dims(r,axis=0)],0)
        if len(self.state_pool[i])>self.pool_size:
            self.state_pool[i]=self.state_pool[i][1:]
            self.action_pool[i]=self.action_pool[i][1:]
            self.next_state_pool[i]=self.next_state_pool[i][1:]
            self.reward_pool[i]=self.reward_pool[i][1:]
        if self.save_episode==True:
            if self.state_name==None and self.action_name==None:
                episode=[s,a,next_s,r]
            elif self.state_name==None:
                episode=[s,self.action_name[a],next_s,r]
            elif self.action_name==None:
                episode=[self.state_name[s],a,self.state_name[next_s],r]
            else:
                episode=[self.state_name[s],self.action_name[a],self.state_name[next_s],r]
        return next_s,end,episode,index
    
        
    def get_episode(self,s):
        next_s=None
        episode=[]
        self.end=False
        while True:
            s=next_s
            if type(self.nn.nn)!=list:
                try:
                    if self.nn.explore!=None:
                        pass
                    a=np.argmax(self.nn.nn(s))
                    if self.action_name==None:
                        next_s,r,end=self.nn.explore(a)
                    else:
                        next_s,r,end=self.nn.explore(self.action_name[a])
                except AttributeError:
                    a=np.argmax(self.nn.nn(s))
                    next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
            else:
                try:
                    if self.nn.explore!=None:
                        pass
                    if len(self.nn.param)==4:
                        if self.state_name==None:
                            a=(self.nn.nn[1](s,p=1)+tf.random.normal([1])).numpy()
                        else:
                            a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+tf.random.normal([1])).numpy()
                    else:
                        if self.state_name==None:
                            a=self.nn.nn[1](s).numpy()
                        else:
                            a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                    if len(a.shape)>0:
                        a=np.argmax(a)
                        next_s,r,end=self.nn.explore(self.action_name[a])
                    else:
                        next_s,r,end=self.nn.explore(a)
                except AttributeError:
                    if len(self.nn.param)==4:
                        a=(self.nn.nn[1](self.state[self.state_name[s]],p=1)+tf.random.normal([1])).numpy()
                    else:
                        a=self.nn.nn[1](self.state[self.state_name[s]]).numpy()
                    if len(a.shape)>0:
                        a=np.argmax(a)
                        next_s,r,end=self.nn.transition(self.state_name[s],self.action_name[a])
                    else:
                        next_s,r,end=self.nn.transition(self.state_name[s],a)
            if end:
                if self.state_name==None and self.action_name==None:
                    episode.append([s,a,next_s,r])
                elif self.state_name==None:
                    episode.append(s,self.action_name[a],next_s,r)
                elif self.action_name==None:
                    episode.append([self.state_name[s],a,self.state_name[next_s],r])
                else:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
                episode.append('end')
                break
            elif self.end==True:
                break
            else:
                if self.state_name==None and self.action_name==None:
                    episode.append([s,a,next_s,r])
                elif self.state_name==None:
                    episode.append(s,self.action_name[a],next_s,r)
                elif self.action_name==None:
                    episode.append([self.state_name[s],a,self.state_name[next_s],r])
                else:
                    episode.append([self.state_name[s],self.action_name[a],self.state_name[next_s],r])
        return episode
    
    
    def index_matrix(self,i):
        if len(self.im)==self.nn.row and self.rank_flag==False:
            self.rank_flag=True
            self.d_index=0
        if self.rank_flag==True:
            if len(self.im)==self.nn.row:
                self.d_index=0
            if len(self.im[self.d_index])!=self.nn.row:
                self.im[self.d_index].append(i)
                self.rank_one[self.d_index]=np.append(self.rank_one[self.d_index],np.array(1,dtype='int8'))
            else:
                self.d_index+=1
                if len(self.im)!=self.nn.row and self.d_index>len(self.im):
                    self.imr=[]
                    self.rank_one[self.d_index]=np.array(0,dtype='int8')
                    self.im.append(self.imr)
                    self.om.append(self.rank_one[self.d_index])
                self.im[self.d_index].append(i)
                self.rank_one[self.d_index]=np.append(self.rank_one[self.d_index],np.array(1,dtype='int8'))
        if len(self.imr)!=self.nn.rank and len(self.im)!=self.nn.row and self.rank_flag!=True:
            if len(self.imr)==0:
                self.im.append(self.imr)
                self.rank_one=np.append(self.rank_one,np.array(1,dtype='int8'))
                self.om.append(self.rank_one)
            self.imr.append(i)
            self.rank_one=np.append(self.rank_one,np.array(1,dtype='int8'))
        else:
            self.imr=[]
        return
    
    
    def learn1(self,i,j=None,batches=None,length=None,episode_num=None,k=None):
        if len(self.state_pool[i])<self.batch:
            length=min(len(self.state_pool[i]),len(self.action_pool[i]),len(self.next_state_pool[i]),len(self.reward_pool[i]))
            if self.PO==1:
                with tf.GradientTape() as tape:
                    if type(self.nn.nn)!=list:
                        self.loss[i]=self.nn.loss(self.nn.nn,self.state_pool[i][:length],self.action_pool[i][:length],self.next_state_pool[i][:length],self.reward_pool[i][:length])
                    elif len(self.nn.param)==4:
                        value=self.nn.nn[0](self.state_pool[i][:length],p=0)
                        TD=tf.reduce_mean((self.reward_pool[i][:length]+self.discount*self.nn.nn[0](self.next_state_pool[i][:length],p=2)-value)**2)
                    else:
                        value=self.nn.nn[0](self.state_pool[i][:length])
                        TD=tf.reduce_mean((self.reward_pool[i][:length]+self.discount*self.nn.nn[0](self.next_state_pool[i][:length])-value)**2)
                if type(self.nn.nn)!=list:
                    self.gradient=tape.gradient(self.loss[i],self.nn.param[0])
                    self.opt(self.gradient,self.nn.param[0])
                elif len(self.nn.param)==4:
                    self.value_gradient=tape.gradient(TD,self.nn.param[0])
                    self.actor_gradient=tape.gradient(value,self.action_pool[i][:length])*tape.gradient(self.action_pool[i][:length],self.nn.param[1])
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                else:
                    self.value_gradient=tape.gradient(TD,self.nn.param[0])
                    self.actor_gradient=TD*tape.gradient(tf.math.log(self.action_pool[i][:length]),self.nn.param[1])
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                if self.update_step!=None:
                    if self.a%self.update_step==0:
                        self.nn.update_param(self.nn.param)
                else:
                    self.nn.update_param(self.nn.param)
                self.loss[i]=0
            else:
                self.thread_lock.acquire()
                self.param=self.nn.param
                with tf.GradientTape() as tape:
                    if type(self.nn.nn)!=list:
                        self.loss[i]=self.nn.loss(self.nn.nn,self.state_pool[i][:length],self.action_pool[i][:length],self.next_state_pool[i][:length],self.reward_pool[i][:length])
                    elif len(self.nn.param)==4:
                        self.value=self.nn.nn[0](self.state_pool[i][:length],p=0)
                        self.TD=tf.reduce_mean((self.reward_pool[i][:length]+self.discount*self.nn.nn[0](self.next_state_pool[i][:length],p=2)-self.value)**2)
                    else:
                        self.value=self.nn.nn[0](self.state_pool[i][:length])
                        self.TD=tf.reduce_mean((self.reward_pool[i][:length]+self.discount*self.nn.nn[0](self.next_state_pool[i][:length])-self.value)**2)
                if type(self.nn.nn)!=list:
                    self.gradient=tape.gradient(self.loss[i],self.param[0])
                elif len(self.nn.param)==4:
                    self.value_gradient=tape.gradient(self.TD,self.param[0])
                    self.actor_gradient=tape.gradient(self.value,self.action_pool[i][:length])*tape.gradient(self.action_pool[i][:length],self.nn.param[1])
                else:
                    self.value_gradient=tape.gradient(self.TD,self.param[0])
                    self.actor_gradient=self.TD*tape.gradient(tf.math.log(self.action_pool[i][:length]),self.param[1])
                self.thread_lock.release()
                self.thread_lock.acquire()
                if type(self.nn.nn)!=list:
                    self.opt(self.gradient,self.nn.param[0])
                else:
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                if self.update_step!=None:
                    if self.a%self.update_step==0:
                        self.nn.update_param(self.nn.param)
                else:
                    self.nn.update_param(self.nn.param)
                self.thread_lock.release()
                self.loss[i]=0
        else:
            try:
                if self.nn.pr!=None:
                    pass
                state_batch,action_batch,next_state_batch,reward_batch=self.nn.pr(self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i],self.pool_size,self.batch,self.nn.rp,self.nn.alpha,self.nn.beta)
            except AttributeError:
                index1=j*self.batch
                index2=(j+1)*self.batch
                state_batch=self.state_pool[i][index1:index2]
                action_batch=self.action_pool[i][index1:index2]
                next_state_batch=self.next_state_pool[i][index1:index2]
                reward_batch=self.reward_pool[i][index1:index2]
                if self.PO==1:
                    with tf.GradientTape() as tape:
                        if type(self.nn.nn)!=list:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        elif len(self.nn.param)==4:
                            value=self.nn.nn[0](state_batch,p=0)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                        else:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                    if type(self.nn.nn)!=list:
                        self.gradient=tape.gradient(batch_loss,self.param[0])
                        self.opt(self.gradient,self.nn.param[0],self.lr)
                        if j>=1:
                            self.loss[i]+=batch_loss
                        if k==episode_num-1:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                            self._loss[i]+=batch_loss
                    elif len(self.nn.param)==4:
                        self.value_gradient=tape.gradient(TD,self.nn.param[0])
                        self.actor_gradient=tape.gradient(value,action_batch)*tape.gradient(action_batch,self.nn.param[1])
                        self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                        if j>=1:
                            self.loss[i]+=TD
                        if k==episode_num-1:
                            value=self.nn.nn[0](state_batch,p=0)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                            self._loss[i]+=TD 
                    else:
                        self.value_gradient=tape.gradient(TD,self.nn.param[0])
                        self.actor_gradient=TD*tape.gradient(tf.math.log(action_batch),self.nn.param[1])
                        self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                        if j>=1:
                            self.loss[i]+=TD
                        if k==episode_num-1:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                            self._loss[i]+=TD
                else:
                    self.thread_lock.acquire()
                    self.param=self.nn.param
                    with tf.GradientTape() as tape:
                        if type(self.nn.nn)!=list:
                            self.batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        elif len(self.nn.param)==4:
                            self.value=self.nn.nn[0](state_batch,p=0)
                            self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-self.value)**2)
                        else:
                            self.value=self.nn.nn[0](state_batch)
                            self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-self.value)**2)
                    if type(self.nn.nn)!=list:
                        self.gradient=tape.gradient(self.batch_loss,self.param[0])
                    elif len(self.nn.param)==4:
                        self.value_gradient=tape.gradient(self.TD,self.param[0])
                        self.actor_gradient=tape.gradient(self.value,action_batch)*tape.gradient(action_batch,self.nn.param[1])
                    else:
                        self.value_gradient=tape.gradient(self.TD,self.param[0])
                        self.actor_gradient=self.TD*tape.gradient(tf.math.log(action_batch),self.param[1])
                    self.thread_lock.release()
                    self.thread_lock.acquire()
                    if type(self.nn.nn)!=list:
                        self.opt(self.gradient,self.nn.param[0],self.lr)
                        if j>=1:
                            self.loss[i]+=self.batch_loss
                        if k==episode_num-1:
                            self.batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                            self._loss[i]+=self.batch_loss
                    elif len(self.nn.param)==4:
                        self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                        if j>=1:
                            self.loss[i]+=self.TD
                        if k==episode_num-1:
                            self.value=self.nn.nn[0](state_batch,p=0)
                            self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-self.value)**2)
                            self._loss[i]+=self.TD
                    else:
                        self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                        if j>=1:
                            self.loss[i]+=self.TD
                        if k==episode_num-1:
                            self.value=self.nn.nn[0](state_batch)
                            self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-self.value)**2)
                            self._loss[i]+=self.TD
                    self.thread_lock.release()
            try:
                self.nn.bc[i]=j
            except AttributeError:
                pass
            if length%self.batch!=0:
                try:
                    if self.pr!=None:
                        pass
                    state_batch,action_batch,next_state_batch,reward_batch=self.nn.pr(self.state_pool[i],self.action_pool[i],self.next_state_pool[i],self.reward_pool[i],self.pool_size,self.batch,self.nn.rp,self.nn.alpha,self.nn.beta)
                except AttributeError:
                    batches+=1
                    index1=batches*self.batch
                    index2=self.batch-(self.shape0-batches*self.batch)
                    state_batch=tf.concat([self.state_pool[i][index1:length],self.state_pool[i][:index2]],0)
                    action_batch=tf.concat([self.action_pool[i][index1:length],self.action_pool[i][:index2]],0)
                    next_state_batch=tf.concat([self.next_state_pool[i][index1:length],self.next_state_pool[i][:index2]],0)
                    reward_batch=tf.concat([self.reward_pool[i][index1:length],self.reward_pool[i][:index2]],0)
                if self.PO==1:
                    with tf.GradientTape() as tape:
                        if type(self.nn.nn)!=list:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        elif len(self.nn.param)==4:
                            value=self.nn.nn[0](state_batch,p=0)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                        else:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                    if type(self.nn.nn)!=list:
                        self.gradient=tape.gradient(batch_loss,self.nn.param[0])
                        self.opt(self.gradient,self.nn.param[0],self.lr)
                        self.loss[i]+=batch_loss
                        if k==episode_num-1:
                            batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                            self._loss[i]+=batch_loss
                    elif len(self.nn.param)==4:
                        self.value_gradient=tape.gradient(TD,self.nn.param[0])
                        self.actor_gradient=tape.gradient(value,action_batch)*tape.gradient(action_batch,self.nn.param[1])
                        self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                        self.loss[i]+=TD
                        if k==episode_num-1:
                            value=self.nn.nn[0](state_batch,p=0)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                            self._loss[i]+=TD
                    else:
                        self.value_gradient=tape.gradient(TD,self.nn.param[0])
                        self.actor_gradient=TD*tape.gradient(tf.math.log(action_batch),self.nn.param[1])
                        self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                        self.loss[i]+=TD
                        if k==episode_num-1:
                            value=self.nn.nn[0](state_batch)
                            TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,)-value)**2)
                            self._loss[i]+=TD
                else:
                    self.thread_lock.acquire()
                    self.param=self.nn.param
                    with tf.GradientTape() as tape:
                        if type(self.nn.nn)!=list:
                            self.batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        elif len(self.nn.param)==4:
                            self.value=self.nn.nn[0](state_batch,p=0)
                            self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-self.value)**2)
                        else:
                            self.value=self.nn.nn[0](state_batch)
                            self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-self.value)**2)
                    if type(self.nn.nn)!=list:
                        self.gradient=tape.gradient(self.batch_loss,self.param[0])
                    elif len(self.nn.param)==4:
                        self.value_gradient=tape.gradient(self.TD,self.param[0])
                        self.actor_gradient=tape.gradient(self.value,action_batch)*tape.gradient(action_batch,self.nn.param[1])
                    else:
                        self.value_gradient=tape.gradient(self.TD,self.param[0])
                        self.actor_gradient=self.TD*tape.gradient(tf.math.log(action_batch),self.param[1])
                    self.thread_lock.release()
                    self.thread_lock.acquire()
                    if type(self.nn.nn)!=list:
                        self.opt(self.gradient,self.nn.param[0],self.lr)
                        self.loss[i]+=self.batch_loss
                        if k==episode_num-1:
                            self.batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                            self._loss[i]+=self.batch_loss
                    elif len(self.nn.param)==4:
                        self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                        self.loss[i]+=self.TD
                        if k==episode_num-1:
                            self.value=self.nn.nn[0](state_batch,p=0)
                            self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-self.value)**2)
                            self._loss[i]+=self.TD
                    else:
                        self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                        self.loss[i]+=self.TD
                        if k==episode_num-1:
                            self.value=self.nn.nn[0](state_batch)
                            self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-self.value)**2)
                            self._loss[i]+=self.TD
                    self.thread_lock.release()
                try:
                    self.nn.bc[i]+=1
                except AttributeError:
                    pass
        return
    
    
    def learn2(self,i,episode_num=None,k=None):
        length=min(len(self.state_pool[i]),len(self.action_pool[i]),len(self.next_state_pool[i]),len(self.reward_pool[i]))
        train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool[i][:length],self.action_pool[i][:length],self.next_state_pool[i][:length],self.reward_pool[i][:length])).shuffle(length).batch(self.batch)
        for state_batch,action_batch,next_state_batch,reward_batch in train_ds:
            if self.PO==1:
                with tf.GradientTape() as tape:
                    if type(self.nn.nn)!=list:
                        batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                    elif len(self.nn.param)==4:
                        value=self.nn.nn[0](state_batch,p=0)
                        TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                    else:
                        value=self.nn.nn[0](state_batch)
                        TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-value)**2)
                if type(self.nn.nn)!=list:
                    self.gradient=tape.gradient(batch_loss,self.nn.param[0])
                    self.opt(self.gradient,self.nn.param[0],self.lr)
                    self.loss[i]+=batch_loss
                    if k==episode_num-1:
                        batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        self._loss[i]+=batch_loss
                elif len(self.nn.param)==4:
                    self.value_gradient=tape.gradient(TD,self.nn.param[0])
                    self.actor_gradient=tape.gradient(value,action_batch)*tape.gradient(action_batch,self.nn.param[1])
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                    self.loss[i]+=TD
                    if k==episode_num-1:
                        value=self.nn.nn[0](state_batch,p=0)
                        TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-value)**2)
                        self._loss[i]+=TD
                else:
                    self.value_gradient=tape.gradient(TD,self.nn.param[0])
                    self.actor_gradient=TD*tape.gradient(tf.math.log(action_batch),self.nn.param[1])
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                    self.loss[i]+=TD
                    if k==episode_num-1:
                        value=self.nn.nn[0](state_batch,param=0)
                        TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,param=1)-value)**2)
                        self._loss[i]+=TD
            else:
                self.thread_lock.acquire()
                self.param=self.nn.param
                with tf.GradientTape() as tape:
                    if type(self.nn.nn)!=list:
                        self.batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                    elif len(self.nn.param)==4:
                        self.value=self.nn.nn[0](state_batch,p=0)
                        self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-self.value)**2)
                    else:
                        self.value=self.nn.nn[0](state_batch)
                        self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-self.value)**2)
                if type(self.nn.nn)!=list:
                    self.gradient=tape.gradient(self.batch_loss,self.param[0])
                elif len(self.nn.param)==4:
                    self.value_gradient=tape.gradient(self.TD,self.param[0])
                    self.actor_gradient=tape.gradient(self.value,action_batch)*tape.gradient(action_batch,self.nn.param[1])
                else:
                    self.value_gradient=tape.gradient(self.TD,self.param[0])
                    self.actor_gradient=self.TD*tape.gradient(tf.math.log(action_batch),self.param[1])
                self.thread_lock.release()
                self.thread_lock.acquire()
                if type(self.nn.nn)!=list:
                    self.opt(self.gradient,self.nn.param[0],self.lr)
                    self.loss[i]+=self.batch_loss
                    if k==episode_num-1:
                        self.batch_loss=self.nn.loss(self.nn.nn,state_batch,action_batch,next_state_batch,reward_batch)
                        self._loss[i]+=self.batch_loss
                elif len(self.nn.param)==4:
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                    self.loss[i]+=self.TD
                    if k==episode_num-1:
                        self.value=self.nn.nn[0](state_batch,p=0)
                        self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch,p=2)-self.value)**2)
                        self._loss[i]+=self.TD
                else:
                    self.opt(self.value_gradient,self.actor_gradient,self.nn.param)
                    self.loss[i]+=self.TD
                    if k==episode_num-1:
                        self.value=self.nn.nn[0](state_batch)
                        self.TD=tf.reduce_mean((reward_batch+self.discount*self.nn.nn[0](next_state_batch)-self.value)**2)
                        self._loss[i]+=self.TD
                self.thread_lock.release()
            try:
                self.nn.bc[i]+=1
            except AttributeError:
                pass
        return
            
    
    def learn3(self,i,episode_num,k):
        self.a+=1
        if len(self.state_pool[i])<self.batch:
            self.learn1(i,episode_num=episode_num,k=k)
        else:
            self.loss[i]=0
            if self.pool_net==True:
                length=min(len(self.state_pool[i]),len(self.action_pool[i]),len(self.next_state_pool[i]),len(self.reward_pool[i]))
                batches=int((length-length%self.batch)/self.batch)
                if length%self.batch!=0:
                    batches+=1
                for j in range(batches):
                    self.learn1(i,j,batches,length,episode_num,k)
            else:
                try:
                    self.nn.bc[i]=0
                except AttributeError:
                    pass
                self.learn2(i,episode_num,k)
            if self.PO==1:
                if self.update_step!=None:
                    if self.a%self.update_step==0:
                        self.nn.update_param(self.nn.param)
                else:
                    self.nn.update_param(self.nn.param)
            else:
                self.thread_lock.acquire()
                if self.update_step!=None:
                    if self.a%self.update_step==0:
                        self.nn.update_param(self.nn.param)
                else:
                    self.nn.update_param(self.nn.param)
                self.thread_lock.release()
            if len(self.state_pool[i])<self.batch:
                self.loss[i]=self.loss[i].numpy()
            else:
                self.loss[i]=self.loss[i].numpy()/batches
                if k==episode_num-1:
                    self.loss[i]=self._loss[i].numpy()/batches
        try:
            self.nn.ec[i]+=1
        except AttributeError:
            pass
        return
    
    
    def learn(self,epsilon,episode_num):
        i=self.t.pop()
        self.thread_lock.acquire()
        self.thread+=1
        self._loss.append(0)
        self.thread_lock.release()
        while len(self.state_pool)<i:
            pass
        if len(self.state_pool)==i:
            self.thread_lock.acquire()
            self.index_matrix(i)
            self.row_p.append(None)
            self.rank_one.append(None)
            self.state_pool.append(None)
            self.action_pool.append(None)
            self.next_state_pool.append(None)
            self.reward_pool.append(None)
            self.epsilon.append(epsilon)
            self.epi_num.append(episode_num)
            try:
                self.nn.ec.append(0)
            except AttributeError:
                pass
            try:
                self.nn.bc.append(0)
            except AttributeError:
                pass
            if self.state_list!=None:
                self.state_list=np.append(self.state_list,np.array(1,dtype='int8'))
            self.thread_sum+=1
            self.thread_lock.release()
        elif i not in self.finish_lis and self.state_list!=None:
            self.state_list[i+1]=1
        for k in range(episode_num):
            if self.episode_num[i]==self.epi_num[i]:
                break
            self.episode_num[i]+=1
            episode=[]
            if self.state_name==None:
                s=self.nn.explore(init=True)
            else:
                s=int(np.random.uniform(0,len(self.state_name)))
            if self.episode_step==None:
                while True:
                    next_s,end,_episode,index=self._explore(s,self.epsilon[i],i)
                    s=next_s
                    if self.state_pool[i]!=None and self.action_pool[i]!=None and self.next_state_pool[i]!=None and self.reward_pool[i]!=None:
                        if self.PO==1:
                            self.thread_lock.acquire()
                            self.learn3(i,episode_num,k)
                            self.thread_lock.release()
                        else:
                            self.learn3(i,episode_num,k)
                    if self.save_episode==True:
                        if index not in self.finish_list:
                            episode.append(_episode)
                    if end:
                        self.thread_lock.acquire()
                        self.loss_list.append(self.loss[i])
                        if self.save_episode==True:
                            episode.append('end')
                            self.episode.append(episode)
                        break
                        self.thread_lock.release()
            else:
                for _ in range(self.episode_step):
                    next_s,end,episode,index=self._explore(s,self.epsilon[i],i)
                    s=next_s
                    if self.state_pool[i]!=None and self.action_pool[i]!=None and self.next_state_pool[i]!=None and self.reward_pool[i]!=None:
                        if self.PO==1:
                            self.thread_lock.acquire()
                            self.learn3(i,episode_num,k)
                            self.thread_lock.release()
                        else:
                            self.learn3(i,episode_num,k)
                    if self.save_episode==True:
                        if index not in self.finish_list:
                            episode.append(_episode)
                    if end:
                        self.thread_lock.acquire()
                        self.loss_list.append(self.loss[i])
                        if self.save_episode==True:
                            episode.append('end')
                            self.episode.append(episode)
                        break
                        self.thread_lock.release()
                if self.save_episode==True:
                    self.thread_lock.acquire()
                    self.episode.append(episode)
                    self.thread_lock.release()
        if i not in self.finish_list:
            self.thread_lock.acquire()
            self.finish_list.append(i)
            self.thread_lock.release()
        self.thread_lock.acquire()
        self.thread-=1
        self.thread_lock.release()
        if self.pool_net==True:
            self.state_pool[i]=None
            self.action_pool[i]=None
            self.next_state_pool[i]=None
            self.reward_pool[i]=None
        return
    
    
    def train_visual(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(self.total_episode),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
        return
    
    
    def save_p(self,path):
        parameter_file=open(path+'.dat','wb')
        pickle.dump(self.value_p,parameter_file)
        parameter_file.close()
        return
    
    
    def save_e(self,path):
        episode_file=open(path+'.dat','wb')
        pickle.dump(self.episode,episode_file)
        episode_file.close()
        return
    
    
    def save(self,path):
        output_file=open(path+'\save.dat','wb')
        path=path+'\save.dat'
        index=path.rfind('\\')
        parameter_file=open(path.replace(path[index+1:],'parameter.dat'),'wb')
        if self.save_episode==True:
            episode_file=open(path.replace(path[index+1:],'episode.dat'),'wb')
            pickle.dump(self.episode,episode_file)
            episode_file.close()
        self.one_list=self.one_list*0
        pickle.dump(self.nn.param,parameter_file)
        self.nn.param=None
        pickle.dump(self.nn,output_file)
        pickle.dump(self.state_pool,output_file)
        pickle.dump(self.action_pool,output_file)
        pickle.dump(self.next_state_pool,output_file)
        pickle.dump(self.reward_pool,output_file)
        pickle.dump(self.action_len,output_file)
        pickle.dump(self.action,output_file)
        pickle.dump(self.action_one,output_file)
        pickle.dump(self.index,output_file)
        pickle.dump(self.epsilon,output_file)
        pickle.dump(self.discount,output_file)
        pickle.dump(self.episode_step,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_step,output_file)
        pickle.dump(self.end_loss,output_file)
        pickle.dump(self.state_list,output_file)
        pickle.dump(self._state_list,output_file)
        pickle.dump(self.p,output_file)
        pickle.dump(self.im,output_file)
        pickle.dump(self.om,output_file)
        pickle.dump(self.rank_flag,output_file)
        pickle.dump(self.row_one,output_file)
        pickle.dump(self.row_p,output_file)
        pickle.dump(self.finish_list,output_file)
        pickle.dump(self.pool_net,output_file)
        pickle.dump(self.save_episode,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.a,output_file)
        pickle.dump(self.epi_num,output_file)
        pickle.dump(self.episode_num,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path,p_path,e_path=None):
        input_file=open(s_path,'rb')
        parameter_file=open(p_path,'rb')
        if e_path!=None:
            episode_file=open(e_path,'rb')
            self.episode=pickle.load(episode_file)
            episode_file.close()
        param=pickle.load(parameter_file)
        self.nn=pickle.load(input_file)
        self.nn.param=param
        param=None
        try:
            self.nn.km=1
        except AttributeError:
            pass
        self.opt=self.nn.opt
        self.state_pool=pickle.load(input_file)
        self.action_pool=pickle.load(input_file)
        self.next_state_pool=pickle.load(input_file)
        self.reward_pool=pickle.load(input_file)
        self.action_len=pickle.load(input_file)
        self.action=pickle.load(input_file)
        self.action_one=pickle.load(input_file)
        self.index=pickle.load(input_file)
        self.epsilon=pickle.load(input_file)
        self.discount=pickle.load(input_file)
        self.episode_step=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_step=pickle.load(input_file)
        self.end_loss=pickle.load(input_file)
        self.state_list=pickle.load(input_file)
        self._state_list=pickle.load(input_file)
        self.p=pickle.load(input_file)
        self.im=pickle.load(input_file)
        self.om=pickle.load(input_file)
        self.rank_flag=pickle.load(input_file)
        self.row_one=pickle.load(input_file)
        self.row_p=pickle.load(input_file)
        self.finish_list=pickle.load(input_file)
        self.pool_net=pickle.load(input_file)
        self.save_episode=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.a=pickle.load(input_file)
        self.epi_num=pickle.load(input_file)
        self.episode_num=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
