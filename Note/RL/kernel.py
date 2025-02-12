import tensorflow as tf
from Note.RL import rl
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest
import numpy as np
import matplotlib.pyplot as plt
import statistics
import pickle
import os
import time


class kernel:
    def __init__(self,nn=None):
        self.nn=nn
        if hasattr(self.nn,'km'):
            self.nn.km=1
        self.platform=None
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        self.pool_size=None
        self.batch=None
        self.update_steps=None
        self.trial_count=None
        self.criterion=None
        self.reward_list=[]
        self.suspend=False
        self.save_epi=None
        self.max_episode_count=None
        self.path=None
        self.avg_reward=None
        self.save_best_only=False
        self.save_param_only=False
        self.path_list=[]
        self.loss=None
        self.loss_list=[]
        self.step_counter=0
        self.total_episode=0
        self.time=0
        self.total_time=0
    
    
    def init(self):
        self.suspend=False
        self.save_epi=None
        self.episode_set=[]
        self.state_pool=None
        self.action_pool=None
        self.next_state_pool=None
        self.reward_pool=None
        self.done_pool=None
        self.reward_list=[]
        self.loss=0
        self.loss_list=[]
        self.step_counter=0
        self.total_episode=0
        self.time=0
        self.total_time=0
        return
    
    
    def set(self,policy=None,noise=None,pool_size=None,batch=None,update_steps=None,trial_count=None,criterion=None,PPO=None,HER=None,MARL=None,PR=None,IRL=None):
        if policy!=None:
            self.policy=policy
        if noise!=None:
            self.noise=noise
            self.nn.noise=True
        if pool_size!=None:
            self.pool_size=pool_size
        if batch!=None:
            self.batch=batch
        if update_steps!=None:
            self.update_steps=update_steps
        if trial_count!=None:
            self.trial_count=trial_count
        if criterion!=None:
            self.criterion=criterion
        if self.PPO!=None:
            self.PPO=PPO
        if self.HER!=None:
            self.HER=HER
        if self.MARL!=None:
            self.MARL=MARL
        if self.PR!=None:
            self.PR=PR
            self.nn.pr.pool_network=False
        if self.IRL!=None:
            self.IRL=IRL
        return
    
    
    def run_agent(self, max_steps, seed=None):
        state_history = []

        steps = 0
        reward_ = 0
        if seed==None:
            state = self.nn.genv.reset()
        else:
            state = self.nn.genv.reset(seed=seed)
        for step in range(max_steps):
            if hasattr(self.platform,'DType'):
                if not hasattr(self, 'noise'):
                    action = np.argmax(self.nn.nn.fp(state))
                else:
                    action = self.nn.actor.fp(state).numpy()
            else:
                if not hasattr(self, 'noise'):
                    action = np.argmax(self.nn.nn.fp(state))
                else:
                    action = self.nn.actor.fp(state).detach().numpy()
            next_state, reward, done, _ = self.nn.genv.step(action)
            state_history.append(state)
            steps+=1
            reward_+=reward
            if done:
                break
            state = next_state
        
        return state_history,reward_,steps
    
    
    @tf.function(jit_compile=True)
    def tf_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        with self.platform.GradientTape(persistent=True) as tape:
            loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        if hasattr(self.nn,'gradient'):
            gradient=self.nn.gradient(tape,loss)
            if hasattr(self.nn.opt,'apply_gradients'):
                self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
            else:
                self.nn.opt(gradient)
        else:
            if hasattr(self.nn,'nn'):
                gradient=tape.gradient(loss,self.nn.param)
                self.nn.opt.apply_gradients(zip(gradient,self.nn.param))
            else:
                actor_gradient=tape.gradient(loss,self.nn.param[0])
                critic_gradient=tape.gradient(loss,self.nn.param[1])
                self.nn.opt.apply_gradients(zip(actor_gradient,self.nn.param[0]))
                self.nn.opt.apply_gradients(zip(critic_gradient,self.nn.param[1]))
        return loss
    
    
    def pytorch_opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        loss=self.nn.loss(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        self.nn.backward(loss)
        self.nn.opt()
        return loss
    
    
    def opt(self,state_batch,action_batch,next_state_batch,reward_batch,done_batch):
        if hasattr(self.platform,'DType'):
            loss=self.tf_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        else:
            loss=self.pytorch_opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
        return loss
    
    
    def opt_ol(self,state,action,next_state,reward,done):
        if hasattr(self.platform,'DType'):
            loss=self.tf_opt(state,action,next_state,reward,done)
        else:
            loss=self.pytorch_opt(state,action,next_state,reward,done)
        return loss
    
    
    def pool(self,s,a,next_s,r,done):
        if type(self.state_pool)!=np.ndarray and self.state_pool==None:
            self.state_pool=s
            self.action_pool=np.expand_dims(a,axis=0)
            self.next_state_pool=np.expand_dims(next_s,axis=0)
            self.reward_pool=np.expand_dims(r,axis=0)
            self.done_pool=np.expand_dims(done,axis=0)
        else:
            self.state_pool=np.concatenate((self.state_pool,s),0)
            self.action_pool=np.concatenate((self.action_pool,np.expand_dims(a,axis=0)),0)
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
    
    
    @tf.function(jit_compile=True)
    def forward(self,s,i=None):
        if self.MARL!=True:
            if hasattr(self.nn,'nn'):
                output=self.nn.nn.fp(s)
            else:
                output=self.nn.actor.fp(s)
        else:
            if hasattr(self.nn,'nn'):
                output=self.nn.nn.fp(s,i)
            else:
                output=self.nn.actor.fp(s,i)
        return output
    
    
    def select_action_(self,output):
        if hasattr(self.nn,'nn'):
            if self.IRL!=True:
                output=output.numpy()
            else:
                output=output[1].numpy()
            output=np.squeeze(output, axis=0)
            if isinstance(self.policy, rl.SoftmaxPolicy):
                a=self.policy.select_action(len(output), output)
            elif isinstance(self.policy, rl.EpsGreedyQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.AdaptiveEpsGreedyPolicy):
                a=self.policy.select_action(output, self.step_counter)
            elif isinstance(self.policy, rl.GreedyQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.BoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.MaxBoltzmannQPolicy):
                a=self.policy.select_action(output)
            elif isinstance(self.policy, rl.BoltzmannGumbelQPolicy):
                a=self.policy.select_action(output, self.step_counter)
        else:
            if hasattr(self.platform,'DType'):
                if self.IRL!=True:
                    a=(output+self.noise.sample()).numpy()
                else:
                    a=(output[1]+self.noise.sample()).numpy()
            else:
                if self.IRL!=True:
                    a=(output+self.noise.sample()).detach().numpy()
                else:
                    a=(output[1]+self.noise.sample()).detach().numpy()
        if self.IRL!=True:
            return a
        else:
            return [output[0],a]
    
    
    def select_action(self,s):
        if hasattr(self.nn,'nn'):
            if hasattr(self.platform,'DType'):
                if self.MARL!=True:
                    output=self.forward(s)
                    a=self.select_action_(output)
                else:
                    a=[]
                    for i in len(s[0]):
                        s=np.expand_dims(s[0][i],axis=0)
                        output=self.forward(s,i)
                        a.append([self.select_action_(output)])
                    a=np.array(a)
            else:
                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                if self.MARL!=True:
                    output=self.nn.nn(s)
                    a=self.select_action_(output)
                else:
                    a=[]
                    for i in len(s[0]):
                        s=np.expand_dims(s[0][i],axis=0)
                        output=self.nn.nn(s,i)
                        a.append([self.select_action_(output)])
                    a=np.array(a)
        else:
            if hasattr(self.platform,'DType'):
                if self.MARL!=True:
                    output=self.forward(s)
                    a=self.select_action_(output)
                else:
                    a=[]
                    for i in len(s[0]):
                        s=np.expand_dims(s[0][i],axis=0)
                        output=self.forward(s,i)
                        a.append([self.select_action_(output)])
                    a=np.array(a)
            else:
                s=self.platform.tensor(s,dtype=self.platform.float).to(self.nn.device)
                if self.MARL!=True:
                    output=self.nn.actor(s)
                    a=self.select_action_(output)
                else:
                    a=[]
                    for i in len(s[0]):
                        s=np.expand_dims(s[0][i],axis=0)
                        output=self.nn.actor(s,i)
                        a.append([self.select_action_(output)])
                    a=np.array(a)
        return a
    
    
    def data_func(self):
        if self.PR:
            s,a,next_s,r,d=self.nn.data_func(self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool,self.batch)
        elif self.HER:
            s = []
            a = []
            next_s = []
            r = []
            d = []
            for _ in range(self.batch):
                step_state = np.random.randint(0, len(self.state_pool)-1)
                step_goal = np.random.randint(step_state+1, step_state+np.argmax(self.done_pool[step_state+1:])+2)
                state = self.state_pool[step_state]
                next_state = self.next_state_pool[step_state]
                action = self.action_pool[step_state]
                goal = self.state_pool[step_goal]
                reward, done = self.nn.reward_done_func(next_state, goal)
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
        
    
    def _train(self):
        if len(self.state_pool)<self.batch:
            if self.loss!=None:
                return self.loss
            else:
                return np.array(0.)
        else:
            loss=0
            self.step_counter+=1
            batches=int((len(self.state_pool)-len(self.state_pool)%self.batch)/self.batch)
            if len(self.state_pool)%self.batch!=0:
                batches+=1
            if self.PR or self.HER==True:
                for j in range(batches):
                    self.suspend_func()
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                    loss+=batch_loss
                    if hasattr(self.nn,'bc'):
                        try:
                            self.nn.bc.assign_add(1)
                        except Exception:
                            self.nn.bc+=1
                if len(self.state_pool)%self.batch!=0:
                    self.suspend_func()
                    state_batch,action_batch,next_state_batch,reward_batch,done_batch=self.data_func()
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                    loss+=batch_loss
                    if hasattr(self.nn,'bc'):
                        try:
                            self.nn.bc.assign_add(1)
                        except Exception:
                            self.nn.bc+=1
            else:
                train_ds=tf.data.Dataset.from_tensor_slices((self.state_pool,self.action_pool,self.next_state_pool,self.reward_pool,self.done_pool)).shuffle(len(self.state_pool)).batch(self.batch)
                for state_batch,action_batch,next_state_batch,reward_batch,done_batch in train_ds:
                    self.suspend_func()
                    if hasattr(self.platform,'DType'):
                        pass
                    else:
                        state_batch=state_batch.numpy()
                        action_batch=action_batch.numpy()
                        next_state_batch=next_state_batch.numpy()
                        reward_batch=reward_batch.numpy()
                        done_batch=done_batch.numpy()
                    batch_loss=self.opt(state_batch,action_batch,next_state_batch,reward_batch,done_batch)
                    loss+=batch_loss
                    if hasattr(self.nn,'bc'):
                        try:
                            self.nn.bc.assign_add(1)
                        except Exception:
                            self.nn.bc+=1
            if self.update_steps!=None:
                if self.step_counter%self.update_steps==0:
                    self.nn.update_param()
                    if self.PPO:
                        self.state_pool=None
                        self.action_pool=None
                        self.next_state_pool=None
                        self.reward_pool=None
                        self.done_pool=None
            else:
                self.nn.update_param()
        if hasattr(self.platform,'DType'):
            loss=loss.numpy()/batches
        else:
            loss=loss.detach().numpy()/batches
        return loss
    
    
    def train_(self):
        episode=[]
        self.reward=0
        s=self.nn.env(initial=True)
        while True:
            s=np.expand_dims(s,axis=0)
            a=self.select_action(s)
            next_s,r,done=self.nn.env(a)
            if hasattr(self.platform,'DType'):
                next_s=np.array(next_s)
                r=np.array(r)
                done=np.array(done)
            self.pool(s,a,next_s,r,done)
            if self.PR:
                self.nn.pr.TD=np.append(self.nn.pr.TD,self.nn.initial_TD)
                if len(self.state_pool)>self.pool_size:
                    self.nn.pr.TD=self.nn.pr.TD[1:]
            self.reward=r+self.reward
            if hasattr(self.platform,'DType'):
                self.nn.pr.TD=tf.Variable(self.nn.pr.TD)
            loss=self._train()
            if done:
                self.reward_list.append(self.reward)
                if len(self.reward_list)>self.trial_count:
                    del self.reward_list[0]
                return loss,episode,done
            s=next_s
    
    
    def train(self,episode_count,path=None,save_freq=1,max_save_files=None,p=None):
        avg_reward=None
        if p==None:
            self.p=9
        else:
            self.p=p-1
        if episode_count%10!=0:
            p=episode_count-episode_count%self.p
            p=int(p/self.p)
        else:
            p=episode_count/(self.p+1)
            p=int(p)
        if p==0:
            p=1
        self.max_save_files=max_save_files
        if episode_count!=None:
            for i in range(episode_count):
                t1=time.time()
                loss,episode,done=self.train_()
                self.loss=loss
                self.loss_list.append(loss)
                self.total_episode+=1
                if path!=None and i%save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(path)
                    else:
                        self.save_(path)
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
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                try:
                    try:
                        self.nn.ec.assign_add(1)
                    except Exception:
                        self.nn.ec+=1
                except Exception:
                    pass
                t2=time.time()
                self.time+=(t2-t1)
        else:
            i=0
            while True:
                t1=time.time()
                loss,episode,done=self.train_()
                self.loss=loss
                self.loss_list.append(loss)
                i+=1
                self.total_episode+=1
                if path!=None and i%save_freq==0:
                    if self.save_param_only==False:
                        self.save_param_(path)
                    else:
                        self.save_(path)
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
                            print('average reward:{0}'.format(avg_reward))
                            print()
                            print('time:{0}s'.format(self.total_time))
                            return
                if i%p==0:
                    if len(self.state_pool)>=self.batch:
                        print('episode:{0}   loss:{1:.6f}'.format(i+1,loss))
                    if avg_reward!=None:
                        print('episode:{0}   average reward:{1}'.format(i+1,avg_reward))
                    else:
                        print('episode:{0}   reward:{1}'.format(i+1,self.reward))
                    print()
                if hasattr(self.nn,'ec'):
                    try:
                        self.nn.ec.assign_add(1)
                    except Exception:
                        self.nn.ec+=1
                t2=time.time()
                self.time+=(t2-t1)
        time_=self.time-int(self.time)
        if time_<0.5:
            self.total_time=int(self.time)
        else:
            self.total_time=int(self.time)+1
        self.total_time+=self.time
        print('time:{0}s'.format(self.time))
        return
    
    
    def train_online(self):
        while True:
            if hasattr(self.nn,'save'):
                self.nn.save(self.save_)
            if hasattr(self.nn,'stop_flag'):
                if self.nn.stop_flag==True:
                    return
            if hasattr(self.nn,'stop_func'):
                if self.nn.stop_func():
                    return
            if hasattr(self.nn,'suspend_func'):
                self.nn.suspend_func()
            data=self.nn.online()
            if data=='stop':
                return
            elif data=='suspend':
                self.nn.suspend_func()
            loss=self.opt_ol(data[0],data[1],data[2],data[3],data[4])
            loss=loss.numpy()
            self.nn.train_loss_list.append(loss)
            if len(self.nn.train_acc_list)==self.nn.max_length:
                del self.nn.train_acc_list[0]
            if hasattr(self.nn,'counter'):
                self.nn.counter+=1
        return
        
    
    def suspend_func(self):
        if self.suspend==True:
            if self.save_epoch==None:
                print('Training have suspended.')
            else:
                self._save()
            while True:
                if self.suspend==False:
                    print('Training have continued.')
                    break
        return
    
    
    def _save(self):
        if self.save_epi==self.total_episode:
            self.save_()
            self.save_epi=None
            print('\nNeural network have saved and training have suspended.')
            return
        elif self.save_epi!=None and self.save_epi>self.total_episode:
            print('\nsave_epoch>total_epoch')
        return
    
    
    def visualize_reward(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.reward_list)
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('reward:{0:.6f}'.format(self.reward_list[-1]))
        return
    
    
    def visualize_train(self):
        print()
        plt.figure(1)
        plt.plot(np.arange(1,self.total_episode+1),self.loss_list)
        plt.title('train loss')
        plt.xlabel('episode')
        plt.ylabel('loss')
        plt.xticks(np.arange(1,self.total_episode+1))
        plt.show()
        print('loss:{0:.6f}'.format(self.loss_list[-1]))
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
            if self.max_save_files==None:
                parameter_file=open(path,'wb')
            else:
                path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_episode))
                parameter_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self.nn.param,parameter_file)
            parameter_file.close()
        else:
            if self.trial_count!=None:
                if len(self.reward_list)>=self.trial_count:
                    avg_reward=statistics.mean(self.reward_list[-self.trial_count:])
                    if self.avg_reward==None or avg_reward>self.avg_reward:
                        self.save_param(path)
                        self.avg_reward=avg_reward
        return
    
    
    def save_param(self,path):
        parameter_file=open(path,'wb')
        pickle.dump(self.nn.param,parameter_file)
        parameter_file.close()
        return
    
    
    def restore_param(self,path):
        parameter_file=open(path,'rb')
        param=pickle.load(parameter_file)
        param_flat=nest.flatten(param)
        param_flat_=nest.flatten(self.nn.param)
        for i in range(len(param_flat)):
            state_ops.assign(param_flat_[i],param_flat[i])
        self.nn.param=nest.pack_sequence_as(self.nn.param,param_flat_)
        parameter_file.close()
        return
    
    
    def save_(self,path):
        if self.save_best_only==False:
            if self.max_save_files==None:
                output_file=open(path,'wb')
            else:
                path=path.replace(path[path.find('.'):],'-{0}.dat'.format(self.total_episode))
                output_file=open(path,'wb')
                self.path_list.append(path)
                if len(self.path_list)>self.max_save_files:
                    os.remove(self.path_list[0])
                    del self.path_list[0]
            pickle.dump(self.nn,output_file)
            pickle.dump(self.policy,output_file)
            pickle.dump(self.noise,output_file)
            pickle.dump(self.pool_size,output_file)
            pickle.dump(self.batch,output_file)
            pickle.dump(self.update_steps,output_file)
            pickle.dump(self.trial_count,output_file)
            pickle.dump(self.criterion,output_file)
            pickle.dump(self.PPO,output_file)
            pickle.dump(self.HER,output_file)
            pickle.dump(self.MARL,output_file)
            pickle.dump(self.PR,output_file)
            pickle.dump(self.IRL,output_file)
            pickle.dump(self.reward_list,output_file)
            pickle.dump(self.loss,output_file)
            pickle.dump(self.loss_list,output_file)
            pickle.dump(self.step_counter,output_file)
            pickle.dump(self.total_episode,output_file)
            pickle.dump(self.total_time,output_file)
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
        pickle.dump(self.nn,output_file)
        pickle.dump(self.policy,output_file)
        pickle.dump(self.noise,output_file)
        pickle.dump(self.pool_size,output_file)
        pickle.dump(self.batch,output_file)
        pickle.dump(self.update_steps,output_file)
        pickle.dump(self.trial_count,output_file)
        pickle.dump(self.criterion,output_file)
        pickle.dump(self.PPO,output_file)
        pickle.dump(self.HER,output_file)
        pickle.dump(self.MARL,output_file)
        pickle.dump(self.PR,output_file)
        pickle.dump(self.IRL,output_file)
        pickle.dump(self.reward_list,output_file)
        pickle.dump(self.loss,output_file)
        pickle.dump(self.loss_list,output_file)
        pickle.dump(self.step_counter,output_file)
        pickle.dump(self.total_episode,output_file)
        pickle.dump(self.total_time,output_file)
        output_file.close()
        return
    
    
    def restore(self,s_path):
        input_file=open(s_path,'rb')
        self.nn=pickle.load(input_file)
        if hasattr(self.nn,'km'):
            self.nn.km=1
        self.policy=pickle.load(input_file)
        self.noise=pickle.load(input_file)
        self.pool_size=pickle.load(input_file)
        self.batch=pickle.load(input_file)
        self.update_steps=pickle.load(input_file)
        self.trial_count=pickle.load(input_file)
        self.criterion=pickle.load(input_file)
        self.PPO=pickle.load(input_file)
        self.HER=pickle.load(input_file)
        self.MARL=pickle.load(input_file)
        self.PR=pickle.load(input_file)
        self.IRL=pickle.load(input_file)
        self.reward_list=pickle.load(input_file)
        self.loss=pickle.load(input_file)
        self.loss_list=pickle.load(input_file)
        self.step_counter=pickle.load(input_file)
        self.total_episode=pickle.load(input_file)
        self.total_time=pickle.load(input_file)
        input_file.close()
        return
