# Introduction:
- **Note is a system(library) for deep learning and reinforcement learning, supporting TensorFlow and PyTorch platforms, supporting non-parallel training and parallel training. Note makes the building and training of neural networks easy and flexible. To train a neural network on Note, you first need to write a neural network class, pass the neural network object to the kernel, and then use the methods provided by the kernel to train a neural network. Note is based on the multiprocessing module of Python to implement parallel training. Because Note is based on the multiprocessing module of Python to implement parallel training, the number of parallel processes is related to the CPU. Note allows you to easily implement parallel training. Currently, the performance of Note’s parallel training has not been sufficiently tested, so don’t know how the performance of parallel training using the multiprocessing module with tf.device is. If you have done the test, you can share your test with me.**
- **Note.nn.layer package contains many layer modules, you can use them to build neural networks. Note’s layer modules are implemented based on TensorFlow, which means they are compatible with TensorFlow’s API. The layer modules allow you to build neural networks in the style of PyTorch or Keras. You can not only use the layer modules to build neural networks trained on Note but also use them to build neural networks trained with TensorFlow.**


# Installation:
**To use Note, you need to download it from https://github.com/NoteDance/Note and then unzip it to the site-packages folder of your Python environment.**

**To import the neural network example conveniently, you can download it from https://github.com/NoteDance/Note-documentation/tree/neural-network-example and unzip the neuralnetwork package to the site-packages folder of your Python environment.**


# Layer modules:
**To use the layer module, you first need to create a layer object, then input data to get the output, like using pytorch, or you can use the layer module like using keras. The args of the layer classes in Note are similar to those of the layer classes in tf.keras.layers, so you can refer to the API documentation of tf.keras.layers to use the layer classes in Note. Neural networks created with the layer module are compatible with TensorFlow, which means you can train these neural networks with TensorFlow.**

https://github.com/NoteDance/Note/tree/Note-7.0/Note/nn/layer

**Documentation:** https://github.com/NoteDance/Note-documentation/tree/layer-7.0

**Using Note’s Layer module, you can determine the shape of the training parameters when you input data like Keras, or you can give the shape of the training parameters in advance like PyTorch.**

**Pytorch:**
```python
from Note.nn.layer.dense import dense
from Note.nn.Module import Module

class nn(Module):
    def __init__(self):
	super().__init__()
        self.layer1=dense(128,784,activation='relu')
        self.layer2=dense(10,128)
    
    def __call__(self,data):
        x=self.layer1(data)
        x=self.layer2(x)
        return x
```
**Keras:**
```python
from Note.nn.layer.dense import dense
from Note.nn.Module import Module

class nn(Module):
    def __init__(self):
	super().__init__()
        self.layer1=dense(128,activation='relu')
        self.layer2=dense(10)
    
    def __call__(self,data):
        x=self.layer1(data)
        x=self.layer2(x)
        return x
```
```python
from Note.nn.layer.dense import dense
from Note.nn.Module import Module

Module.init()
def nn(data):
    x=dense(128,activation='relu')(data)
    x=dense(10)(x)
    return x
```
**Note.neuralnetwork.tf package contains neural networks implemented with Note’s layer module that can be trained with TensorFlow.**
https://github.com/NoteDance/Note/tree/Note-7.0/Note/neuralnetwork/tf

**Documentation:** https://github.com/NoteDance/Note-documentation/tree/tf-7.0


# Note.nn.Module.Module:
**Module class manages the parameters of the neural network. You can initialize the param list by calling the init function of the Module class, which can clear the neural network parameters stored in the param list.**
```python
Module.init()
```

https://github.com/NoteDance/Note/blob/Note-7.0/Note/nn/Module.py


# Note.nn.initializer.initializer:
**This function is used to initialize the parameters of the neural network, and it returns a TensorFlow variable.**

https://github.com/NoteDance/Note/blob/Note-7.0/Note/nn/initializer.py


# Note.nn.initializer.initializer_:
**This function is used to initialize the parameters of the neural network, it returns a TensorFlow variable and stores the variable in Module.param.**

https://github.com/NoteDance/Note/blob/Note-7.0/Note/nn/initializer.py


# Note.nn.Layers.Layers:
**This class is used similarly to the tf.keras.Sequential class.**

https://github.com/NoteDance/Note/blob/Note-7.0/Note/nn/Layers.py


# The models that can be trained with TensorFlow:
**This package include Llama2, CLIP, ViT, ConvNeXt, SwiftFormer, etc. These models built with Note are compatible with TensorFlow and can be trained with TensorFlow.**

https://github.com/NoteDance/Note/tree/Note-7.0/Note/neuralnetwork/tf

**Documentation:** https://github.com/NoteDance/Note-documentation/tree/tf-7.0


# Build neural network trained with Note:
- **Every neural network is regarded as an object, and the neural network object is passed into the kernel and trained by the kernel. To build a neural network that can be trained on Note, you need to follow some rules, otherwise you will get errors during training. You can see the examples of neural networks in the documentation. You can first learn the rules from the simple neural network examples named nn.py, nn_acc.py, and nn_device.py. Then, you can write a Python module for your neural network class and import it. Next, pass the neural network object to the kernel and train it.**

- **Neural network class should define a forward propagation function fp(data), and if using parallel kernel, it should define fp(data,p) where p is the process number. fp passes in the data and returns output, a loss function loss(output,labels), and if using parallel kernel, it should define loss(output,labels,p) where p is the process number. loss passes in the output and labels and returns loss value. If using parallel kernel, it should also define an optimization function opt(gradient,p) and GradientTape(data,labels,p) where p is the process number. opt passes in the gradient and returns parameter, GradientTape passes in the data and labels and returns tape, output and loss. It should also have a self.param object, which is a list for storing the trainable parameters. The kernel needs this list to calculate the gradients and update the parameters.**

- **The Note.neuralnetwork package contains more complex neural network implementations. For some unknown reason, neural networks built with Keras may not be able to train in parallel on Note. The neural networks in the Note.neuralnetwork package use the layer module provided by Note, so they can train in parallel on Note. I hope you can learn how to build more complex networks on Note from these neural network codes.**

**Examples of training neural networks with kernel are shown below.**


# Deep Learning:

## Non-parallel training:
```python
import Note.DL.kernel as k   #import kernel module
import tensorflow as tf      #import tensorflow library
import neuralnetwork.DL.tensorflow.non_parallel.nn as n   #import neural network module
mnist=tf.keras.datasets.mnist #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                    #create neural network object
kernel=k.kernel(nn)          #create kernel object with the network
kernel.platform=tf           #set the platform to tensorflow
kernel.data(x_train,y_train) #input train data to the kernel
kernel.train(32,5)           #train the network with batch size 32 and epoch 5
kernel.test(x_test,y_test,32)#test the network performance on the test set with batch size 32
```


## Parallel training:

**Parallel optimization:**

**You can use parallel optimization to speed up neural network training, and parallel optimization is implemented through multiprocessing.**

**Note have three types of parallel optimization:**

**1. Perform forward propagation and optimization in parallel. (PO1)**

**2. Perform forward propagation, one gradient calculation or multiple gradient computations and optimization in parallel. (PO2)**

**3. Perform forward propagation, gradient calculation and optimization in parallel without locks. (PO3)**

**Neural networks built with Keras may not be able to train in parallel on Note’s parallel kernel. You can use the layer modules in the Note.nn.layer package and the low-level API of Tensorflow to build neural networks that can train in parallel on Note’s parallel kernel. Do not use Keras’s optimizers because they cannot be serialized by the multiprocessing module. You can use the optimizers in the Note.nn.parallel package, or implement your own optimizers with the low-level API of Tensorflow.**
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```

**Multidevice:**

**If you have multiple devices that you want to allocate, you can use the process index to freely assign devices to your operations. For example, if you are using TensorFlow as the platform(backend), you can use tf.device and assign_device to assign devices to the computation. Here is a simple example of a neural network: https://github.com/NoteDance/Note-documentation/blob/neural-network-example/7.0/neuralnetwork/DL/tensorflow/parallel/nn_device.py. The neuralnetwork package contains more complex examples of neural networks. Basically, the multiprocessing module is responsible for launching parallel processes and sharing some data, and the TensorFlow platform(backend) is responsible for computing and allocating computation to different devices.**
```python
import Note.DL.parallel.kernel as k   #import kernel module
import tensorflow as tf              #import tensorflow library
import neuralnetwork.DL.tensorflow.parallel.nn_device as n   #import neural network module
from multiprocessing import Process,Manager #import multiprocessing tools
mnist=tf.keras.datasets.mnist        #load mnist dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data() #split data into train and test sets
x_train,x_test =x_train/255.0,x_test/255.0 #normalize data
nn=n.nn()                            #create neural network object
nn.build()                           #build the network structure
kernel=k.kernel(nn)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(x_train,y_train)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
kernel.test(x_train,y_train,32)      #test the network performance on the train set with batch size 32
```


# Reinforcement Learning:

**The version of gym used in the example is less than 0.26.0.**

## Non-parallel training:
```python
import Note.RL.kernel as k   #import kernel module
import tensorflow as tf           #import tensorflow library
import neuralnetwork.RL.tensorflow.non_parallrl.DQN as d   #import deep Q-network module
dqn=d.DQN(4,128,2)                #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn)              #create kernel object with the network
kernel.platform=tf                #set the platform to tensorflow
kernel.action_count=2             #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.train(100)                 #train the network for 100 episodes
kernel.visualize_train()
kernel.visualize_reward()
```


## Parallel training:

**Pool Network:**

![3](https://github.com/NoteDance/Note-documentation/blob/Note-7.0/picture/Pool%20Network.png)

**Pool net use multiprocessing parallel and random add episode in pool,which would make data being uncorrelated in pool,
then pools would be used parallel training agent.**
```python
import Note.RL.parallel.kernel as k   #import kernel module
import neuralnetwork.RL.tensorflow.parallrl.DQN as d   #import deep Q-network module
from multiprocessing import Process,Lock,Manager #import multiprocessing tools
dqn=d.DQN(4,128,2)           #create neural network object with 4 inputs, 128 hidden units and 2 outputs
kernel=k.kernel(dqn,5)       #create kernel object with the network and 5 processes to train
kernel.episode=100           #set the number of episodes to 100
manager=Manager()            #create manager object to share data among processes
kernel.init(manager)         #initialize shared data with the manager
kernel.action_count=2        #set the number of actions to 2
kernel.set_up(epsilon=0.01,pool_size=10000,batch=64,update_step=10) #set up the hyperparameters for training
kernel.PO=3                  #use PO3 algorithm for parallel optimization
pool_lock=[Lock(),Lock(),Lock(),Lock(),Lock()] #create a list of locks for each process's replay pool
lock=[Lock(),Lock()]         #create two locks for synchronization
for p in range(5):           #loop over the processes
    Process(target=kernel.train,args=(p,lock,pool_lock)).start() #start each process with the train function and pass the process id, the number of episodes, the locks and the pool locks as arguments
```


# Neural network:
**The neuralnetwork package in Note has models that can be trained in parallel on Note, such as ConvNeXt, EfficientNetV2, EfficientNet, etc. You only need to provide the training data and operate simply, then you can train these neural networks in parallel on Note.**

https://github.com/NoteDance/Note/tree/Note-7.0/Note/neuralnetwork

**Documentation:** https://github.com/NoteDance/Note-documentation/tree/neural-network-7.0

**The following is an example of training on the CIFAR10 dataset.**

**ConvNeXtV2:**

**Train:**
```python
import Note.DL.parallel.kernel as k   #import kernel module
from Note.neuralnetwork.ConvNeXtV2 import ConvNeXtV2 #import neural network class
from tensorflow.keras import datasets
from multiprocessing import Process,Manager #import multiprocessing tools
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
train_images,test_images=train_images/255.0,test_images/255.0
convnext_atto=ConvNeXtV2(model_type='atto',classes=10)  #create neural network object
convnext_atto.build()                           #build the network structure
kernel=k.kernel(convnext_atto)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(train_images,train_labels)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
```
**Use the trained model:**
```python
convnext_atto.km=0
output=convnext_atto.fp(data)
```
**ConvNeXtV2:**

**Train:**
```python
import Note.DL.parallel.kernel as k   #import kernel module
from Note.neuralnetwork.ConvNeXtV2 import ConvNeXtV2 #import neural network class
convnext_atto=ConvNeXtV2(model_type='atto',classes=1000)  #create neural network object
convnext_atto.build()                           #build the network structure
kernel=k.kernel(convnext_atto)                  #create kernel object with the network
kernel.process=3                     #set the number of processes to train
kernel.epoch=5                       #set the number of epochs to train
kernel.batch=32                      #set the batch size
kernel.PO=3                          #use PO3 algorithm for parallel optimization
kernel.data(train_data,train_labels)         #input train data to the kernel
manager=Manager()                    #create manager object to share data among processes
kernel.init(manager)                 #initialize shared data with the manager
for p in range(3):                   #loop over the processes
	Process(target=kernel.train,args=(p,)).start() #start each process with the train function and pass the process id as argument
kernel.update_nn_param()             #update the network parameters after training
```
**Fine tuning:**
```python
convnext_atto.fine_tuning(10,0.0001)
kernel.process=3
kernel.epoch=1
kernel.batch=32                      #set the batch size
kernel.data(fine_tuning_data,fine_tuning_labels)
manager=Manager()
kernel.init(manager)
for p in range(3):
	Process(target=kernel.train,args=(p,)).start()
kernel.update_nn_param()
```
**Use the trained model:**
```python
convnext_atto.fine_tuning(flag=1)
convnext_atto.km=0
output=convnext_atto.fp(data)
```


# Study kernel:
**If you want to study kernel, you can see the kernel with comments at the link below.**

**Actually, the kernel code comments are generated by GPT-4. If you want to study the kernel, you can input the kernel code to GPT-4, and GPT-4 should be able to give a good explanation. This can help you quickly understand how the kernel works.**

**DL:** https://github.com/NoteDance/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/DL/kernel

**RL:** https://github.com/NoteDance/Note-documentation/tree/Note-7.0/Note%207.0%20documentation/RL/kernel


# Documentation:
**The document has kernel code and other code with comments that can help you understand.**

**Documentation readme has other examples.**

https://github.com/NoteDance/Note-documentation


# Patreon:
**You can support this project on Patreon.**

https://www.patreon.com/NoteDance


# Contact:
**If you have any issues with the use, or you have any suggestions, you can contact me.**

**E-mail:** notedance@outlook.com
