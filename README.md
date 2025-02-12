# Introduction:
Note is a machine learning library. Note makes the building and training of neural networks easy and flexible. Note can be used not only for deep learning but also for reinforcement learning, it allows you to easily train agents built with Note, Keras, or PyTorch using reinforcement learning.


# Installation:
To use Note, you need to download it from https://github.com/NoteDance/Note and then unzip it to the site-packages folder of your Python environment.

**dependent packages**:

tensorflow>=2.16.1

pytorch>=2.3.1

gym<=0.25.2

matplotlib>=3.8.4

einops>=0.8.0

**python requirement**:

python>=3.10


# Deep learning:
Note.nn.layer package contains many layer modules, you can use them to build neural networks. Note’s layer modules are implemented based on TensorFlow, which means they are compatible with TensorFlow’s API. The layer modules allow you to build neural networks in the style of PyTorch or Keras. You can use the layer modules to build neural networks trained with TensorFlow.

https://github.com/NoteDance/Note/tree/Note-7.0/Note/nn/layer

**Documentation**: https://github.com/NoteDance/Note-documentation/tree/layer-7.0

Using Note’s Layer module, you can determine the shape of the training parameters when you input data like Keras, or you can give the shape of the training parameters in advance like PyTorch.

**Pytorch:**
```python
from Note import nn

class model(nn.Model):
    def __init__(self):
	super().__init__()
        self.layer1=nn.dense(128,784,activation='relu')
        self.layer2=nn.dense(10,128)
    
    def __call__(self,data):
        x=self.layer1(data)
        x=self.layer2(x)
        return x
```
**Keras:**
```python
from Note import nn

class model(nn.Model):
    def __init__(self):
	super().__init__()
        self.layer1=nn.dense(128,activation='relu')
        self.layer2=nn.dense(10)
    
    def __call__(self,data):
        x=self.layer1(data)
        x=self.layer2(x)
        return x
```
Note.models.tf package contains neural networks implemented with Note’s layer module that can be trained with TensorFlow. You can also consider these models as examples using the Note. The documentation shows how to train, test, save, and restore models built with Note.

https://github.com/NoteDance/Note/tree/Note-7.0/Note/models/tf

**Documentation**: https://github.com/NoteDance/Note-documentation/tree/Model-7.0


# Reinforcement learning：
You just need to have your agent class inherit from the RL or RL_pytorch class, and you can easily train your agent built with Note, Keras or PyTorch. You can learn how to build an agent from the examples [here](https://github.com/NoteDance/Note/tree/Note-7.0/Note/models/docs_example/RL). The documentation shows how to train, save, and restore agent built with Note, Keras or PyTorch.

**Documentation**: https://github.com/NoteDance/Note-documentation/tree/RL-7.0


# Function modules:
**Documentation**: https://github.com/NoteDance/Note-documentation/tree/function-7.0


# Note.nn.Model.Model:
Model class manages the parameters and layers of neural network.

**Model's function documentation**: https://github.com/NoteDance/Note-documentation/tree/function-7.0


# Note.nn.initializer.initializer:
This function is used to initialize the parameters of the neural network, it returns a TensorFlow variable and stores the variable in trainable parameters list(Model.param).


# Note.nn.initializer.initializer_:
This function is used to initialize the parameters of the neural network, and it returns a TensorFlow variable.


# Note.nn.parameter.Parameter:
Its function is similar to torch.nn.parameter.Parameter.


# Note.nn.Sequential.Sequential:
This class is used similarly to tf.keras.Sequential and torch.nn.Sequential.


# Patreon:
You can support this project on Patreon.

https://www.patreon.com/NoteDance


# Contact:
If you have any issues with the use, or you have any suggestions, you can contact me.

**E-mail:** notedance@outlook.com
