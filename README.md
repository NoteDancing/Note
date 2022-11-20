# Note


# documentation:
https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv


# Introduction:
Note is an AI system for deep learning and reinforcement learning.


# Deep Learning:
If you done your neural network,you can use kernel to train.

neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv/DL/neural%20network

example:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)                 #start kernel
kernel.platform=tf                       #use platform
kernel.data(x_train,y_train)   #input you data,if you have test data can transfer to kernel API data()
                               #data can be a list,[data1,data2,...,datan]
kernel.train(32,5)         #train neural network
                           #batch: batch size
                           #epoch:epoch
kernel.save()              #save neural network
```
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
kernel=k.kernel()                 #start kernel
kernel.platform=tf                    #use platform
kernel.data(x_train,y_train)   #input you data,if you have test data can transfer to kernel API data()
                               #data can be a list,[data1,data2,...,datan]
kernel.restore('save.dat')     #restore neural network
kernel.nn.opt=tf.keras.optimizers.Adam().from_config(kernel.config)
kernel.train(32,1)             #train again
```

pytorch:
```python
import Note.create.DL.kernel as k   #import kernel
import torch                         #import platform
import nn as n                          #import neural network
from torchvision import datasets
training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
train_data,train_labels=training_data
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)                 #start kernel
kernel.platform=torch                   #use platform
kernel.data(train_data,train_labels)   #input you data,if you have test data can transfer to kernel API data()
                                       #data can be a list,[data1,data2,...,datan]
kernel.train(64,5)         #train neural network
                           #batch: batch size
                           #epoch:epoch
```
## Parallel optimization:
You can use parallel optimization to speed up neural network training,parallel optimization speed up training by multithreading.

Note have three types of parallel optimization:
1. not parallel computing gradient and optimizing.(kernel.PO=1)
2. parallel computing gradient and optimizing.(kernel.PO=2)

parallel optimization may cause unstable training(the estimate of the gradient is biased) but it can make the loss function jump out of the local minimum.

neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv/DL/neural%20network

**Use second parallel optimization to train on MNIST,speed was increased by more than 2 times!**

**Tensorflow version:2.9.1**

**batch size:32**

**epoch:6**

**thread count:2**

**PO:2**

**CPU:i5-8400**

**GPU:GTX 1050 Ti**

**Not use parallel optimization to train spending 15s,use parallel optimization to train spending 6.8s.**

![1](https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv/picture/time.png)
![2](https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv/picture/time(PO).png)

## Multithreading:
**Note can speed up training by multithreading and has stop mechanism to resolve unstable training.**

**Note uses multithreading parallel forward propagation and optimizes neural network.**

neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv/DL/neural%20network

example:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=2                        #thread count
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
kernel.save()              #save neural network
```
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
kernel=k.kernel()                 #start kernel
kernel.platform=tf                    #use platform
kernel.restore('save.dat')     #restore neural network
kernel.thread=2                        #thread count
kernel.PO=2
kernel.data(x_train,y_train)   #input you data,if you have test data can transfer to kernel API data()
                               #data can be a list,[data1,data2,...,datan]
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,1)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
```

pytorch:
```python
import Note.create.DL.kernel as k   #import kernel
import torch                         #import platform
import nn as n                        #import neural network
import threading
from torchvision import datasets
training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
train_data,train_labels=training_data
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=torch                            #use platform
kernel.thread=2                        #thread count
kernel.PO=1
kernel.data(train_data,train_labels)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
```

PO3:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=7                        #thread count
kernel.PO=3
kernel.threading=threading
kernel.max_lock=7
kernel.data(x_train,y_train)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,1)
for _ in range(7):
	_thread=thread()
	_thread.start()
for _ in range(7):
	_thread.join()
```
PO3(matrix):
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.thread=6                        #thread count
kernel.PO=3
kernel.threading=threading
kernel.row=2
kernel.rank=3
kernel.data(x_train,y_train)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,1)
for _ in range(6):
	_thread=thread()
	_thread.start()
for _ in range(6):
	_thread.join()
```

Stop multithreading training and saving when condition is met.

example:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()                                #create neural network object
kernel=k.kernel(cnn)   #start kernel
kernel.platform=tf                            #use platform
kernel.stop=True
kernel.end_loss=0.7
kernel.thread=2                        #thread count
kernel.PO=2
kernel.data(x_train,y_train)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
```

pytorch:
```python
import Note.create.DL.kernel as k   #import kernel
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
import threading
from torchvision import datasets
training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
train_data,train_labels=training_data
nn=n.nn()                                #create neural network object
kernel=k.kernel(nn)   #start kernel
kernel.platform=torch                            #use platform
kernel.stop=True
kernel.end_loss=0.7
kernel.thread=2                        #thread count
kernel.PO=1
kernel.data(train_data,train_labels)   #input you data
kernel.thread_lock=[threading.Lock(),threading.Lock()]
class thread(threading.Thread):
	def run(self):
		kernel.train(32,3)
for _ in range(2):
	_thread=thread()
	_thread.start()
for _ in range(2):
	_thread.join()
kernel.train_loss_list or kernel.train_loss       #view training loss
kernel.visualize_train()
```

## Test neural network：
```python
import Note.create.DL.dl.test_nn
import tensorflow as tf              #import platform
import cnn as c                          #import neural network
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test =x_train/255.0,x_test/255.0
y_train=tf.one_hot(y_train,10).numpy()
cnn=c.cnn()
test_nn.test(cnn,tf,x_train[:32],y_train[:32])
```


# Reinforcement Learning:
neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv/RL/neural%20network

example:

DQN:
```python
import Note.create.RL.nspn.kernel as k   #import kernel
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel=k.kernel(dqn)   #start kernel
kernel.action_num=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_num=10)
kernel.train(500)
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```
```python
import Note.create.RL.rl.visual as v
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
display=Display(visible=0, size=(400, 300))
display.start()
visual=v.visual(dqn,dqn.genv,1000,10)
images=visual.render_episode()
visual.visualize_episode(images,'cartpole-v0.gif',append_images=images[1:]) #visualize episode
```
```python
import Note.create.RL.rl.reward as r
r=r.reward(dqn,dqn.genv)
reward=r.reward(1000) #test neural network
```

DDPG:
```python
import Note.create.RL.nspn.kernel as k   #import kernel
import DDPG as d
import gym
env=gym.make('Pendulum-v0')
state_dim=env.observation_spave.shape[0]
action_dim=env.action_spave.shape[0]
action_bound=env.action_spave.high[0]
ddpg=d.DDPG(state_dim,64,action_dim,action_bound,0.01,0.98,0.005,5e-4,5e-3)         #create neural network object
ddpg.genv=env
kernel=k.kernel(ddpg)   #start kernel
kernel.set_up(pool_size=10000,batch=64)
kernel.train(200)
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

## Pool Net:
![3](https://github.com/NoteDancing/Note-documentation/blob/main/Note%207.0%20pv/picture/Pool%20Net.png)

neural network example:https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv/RL/neural%20network
### list:
example:
```python
import Note.create.RL.kernel as k   #import kernel
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel.threading=threading
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock(),threading.Lock()]
kernel=k.kernel(dqn,5,thread_lock)   #start kernel
kernel.action_num=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=2
class thread(threading.Thread):
	def run(self):
		kernel.train(100)
for _ in range(5):
	_thread=thread()
	_thread.start()
for _ in range(5):
	_thread.join()
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

Stop multithreading training and saving when condition is met.

example:
```python
import Note.create.RL.kernel as k   #import kernel
import DQN as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel.threading=threading
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock(),threading.Lock()]
kernel=k.kernel(dqn,5,thread_lock)   #start kernel
kernel.stop=True
kernel.action_num=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10,trial_num=10,criterion=200)
kernel.PO=2
class thread(threading.Thread):
	def run(self):
		kernel.train(100)
for _ in range(5):
	_thread=thread()
	_thread.start()
for _ in range(5):
	_thread.join()
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

### matrix:
example:
```python
import Note.create.RL.kernel as k   #import kernel
import DQNm as d
dqn=d.DQN(4,128,2)                               #create neural network object
kernel.threading=threading
kernel.thread_lock=[threading.Lock(),threading.Lock(),threading.Lock(),threading.Lock()]
kernel=k.kernel(dqn,6,thread_lock)   #start kernel
kernel.action_num=2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10)
kernel.PO=2
class thread(threading.Thread):
	def run(self):
		kernel.train(100)
for _ in range(6):
	_thread=thread()
	_thread.start()
for _ in range(6):
	_thread.join()
kernel.loss_list or kernel.loss       #view training loss
kernel.visualize_train()
kernel.reward                         #view reward
kernel.visualize_reward()
```

## Test neural network：
```python
import DQN as d
import Note.create.RL.rl.test_nn
dqn=d.DQN(4,128,2)                               #create neural network object
test_nn.test(dqn,2)
```


# Note Compiler:
documentation:https://github.com/NoteDancing/Note-documentation/tree/main/Note%207.0%20pv/compiler
```python
import Note.create.nc as nc
c=nc.compiler('nn.n')
c.Compile()
```


# Patreon:
You can support this project on Patreon.

https://www.patreon.com/NoteDancing
