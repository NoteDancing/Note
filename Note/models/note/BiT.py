import tensorflow as tf
from Note.nn.layer.conv2d import conv2d
from Note.nn.layer.group_norm import group_norm
from Note.nn.layer.zeropadding2d import zeropadding2d
from Note.nn.layer.max_pool2d import max_pool2d
from Note.nn.layer.adaptive_avg_pooling2d import adaptive_avg_pooling2d
from Note.nn.Sequential import Sequential
from Note.nn.parallel.optimizer import Adam
from Note.nn.parallel.assign_device import assign_device
from Note.nn.Model import Model


class StdConv2d:
  def __init__(self,filters,kernel_size,input_size,stride=[1,1],padding=None,bias=True):
      self.zeropadding2d = zeropadding2d(filters, padding)
      self.conv2d = conv2d(filters=filters,kernel_size=kernel_size,input_size=input_size,
                           strides=stride,use_bias=bias)
      w = self.conv2d.weight
      v, m = tf.nn.moments(w, axes=[1, 2, 3], keepdims=True)
      w = (w - m) / tf.math.sqrt(v + 1e-10)
      self.conv2d.weight.assign(w)


  def __call__(self, x):
    out = self.zeropadding2d(x)
    out = self.conv2d(out)
    return out


def conv3x3(cin, cout, stride=1, bias=False):
  return StdConv2d(cout, input_size=cin, kernel_size=3, strides=stride, padding=1,
                   bias=bias)


def conv1x1(cin, cout, stride=1, bias=False):
  return StdConv2d(cout, input_size=cin, kernel_size=1, strides=stride, padding=0,
                   bias=bias)


class PreActBottleneck:
  """Pre-activation (v2) bottleneck block.

  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, cin, cout=None, cmid=None, stride=1):
    cout = cout or cin
    cmid = cmid or cout//4

    self.gn1 = group_norm(32, cin)
    self.conv1 = conv1x1(cin, cmid)
    self.gn2 = group_norm(32, cmid)
    self.conv2 = conv3x3(cmid, cmid, stride)  # Original code has it on conv1!!
    self.gn3 = group_norm(32, cmid)
    self.conv3 = conv1x1(cmid, cout)
    self.relu = tf.nn.relu

    if (stride != 1 or cin != cout):
      # Projection also with pre-activation according to paper.
      self.downsample = conv1x1(cin, cout, stride)

  def __call__(self, x):
    out = self.relu(self.gn1(x))

    # Residual branch
    residual = x
    if hasattr(self, 'downsample'):
      residual = self.downsample(out)

    # Unit's branch
    out = self.conv1(out)
    out = self.conv2(self.relu(self.gn2(out)))
    out = self.conv3(self.relu(self.gn3(out)))

    return out + residual


class BiT(Model):
  """Implementation of Pre-activation (v2) ResNet mode."""

  def __init__(self, model_type, head_size=21843, zero_head=False, device='GPU'):
    super().__init__()
    
    block_units = model_type['block_units']
    wf = model_type['width_factor']  # shortcut 'cause we'll use it a lot.
    
    self.wf = wf

    # The following will be unreadable if we split lines.
    # pylint: disable=line-too-long
    self.root = Sequential()
    self.root.add(StdConv2d(64*wf, input_size=3, kernel_size=7, stride=2, padding=3, bias=False))
    self.root.add(zeropadding2d(64*wf,1))
    self.root.add(max_pool2d(3, 2, 'VALID'))

    self.body = Sequential()
    self.body.add(PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))
    for i in range(2, block_units[0] + 1):
        self.body.add(PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf))
    self.body.add(PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))
    for i in range(2, block_units[1] + 1):
        self.body.add(PreActBottleneck(cin=512*wf, cout=512*wf, cmid=128*wf)) 
    self.body.add(PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))
    for i in range(2, block_units[2] + 1):
        self.body.add(PreActBottleneck(cin=1024*wf, cout=1024*wf, cmid=256*wf)) 
    self.body.add(PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))
    for i in range(2, block_units[3] + 1):
        self.body.add(PreActBottleneck(cin=2048*wf, cout=2048*wf, cmid=512*wf)) 
    # pylint: enable=line-too-long

    self.zero_head = zero_head
    self.head = Sequential()
    self.head.add(group_norm(32, 2048*wf))
    self.head.add(tf.nn.relu)
    self.head.add(adaptive_avg_pooling2d(1))
    self.head.add(conv2d(head_size, 1, 2048*wf))
    
    self.loss_object=tf.keras.losses.SparseCategoricalCrossentropy()
    self.optimizer=Adam()
    self.device=device
    self.km=0
    
    
  def fine_tuning(self,classes=None,flag=0):
      self.flag=flag
      if flag==0:
          self.conv2d=self.head.layer[-1]
          if self.zero_head:
              self.head.layer[-1]=conv2d(classes, 1, 2048*self.wf, weight_initializer='zeros')
          else:
              self.head.layer[-1]=conv2d(classes, 1, 2048*self.wf)
          self.param[-len(self.head.layer[-1].param):]=self.head.layer[-1].param
          for param in self.param[:-len(self.head.layer[-1].param)]:
              param._trainable=False
      elif flag==1:
            for param in self.param[:-len(self.head.layer[-1].param)]:
                param._trainable=True
      else:
          self.head.layer[-1],self.nn.conv2d=self.nn.conv2d,self.head.layer[-1]
          self.param[-len(self.head.layer[-1].param):]=self.head.layer[-1].param
          for param in self.param[:-len(self.head.layer[-1].param)]:
              param._trainable=True
      return


  def fp(self, data, p=None):
    if self.km==1:
        with tf.device(assign_device(p,self.device)):
            output = self.head(self.body(self.root(data)))
            assert output.shape[-2:] == (1, 1)  # We should have no spatial shape left.
            return tf.nn.softmax(tf.transpose(output,[0,3,1,2])[...,0,0])
    else:
        output = self.head(self.body(self.root(data)))
        assert output.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return tf.nn.softmax(tf.transpose(output,[0,3,1,2])[...,0,0])
    
    
  def loss(self,output,labels,p):
      with tf.device(assign_device(p,self.device)):
          loss=self.loss_object(labels,output)
          return loss
    
    
  def GradientTape(self,data,labels,p):
      with tf.device(assign_device(p,self.device)):
          with tf.GradientTape(persistent=True) as tape:
              output=self.fp(data,p)
              loss=self.loss(output,labels,p)
          return tape,output,loss
    
    
  def opt(self,gradient,p):
      with tf.device(assign_device(p,self.device)):
          param=self.optimizer(gradient,self.param,self.bc[0])
          return param


MODEL_CONFIGS = {
    'BiT-M-R50x1': {'block_units': [3, 4, 6, 3], 'width_factor': 1},
    'BiT-M-R50x3': {'block_units': [3, 4, 6, 3], 'width_factor': 3},
    'BiT-M-R101x1': {'block_units': [3, 4, 23, 3], 'width_factor': 1},
    'BiT-M-R101x3': {'block_units': [3, 4, 23, 3], 'width_factor': 3},
    'BiT-M-R152x2': {'block_units': [3, 8, 36, 3], 'width_factor': 2},
    'BiT-M-R152x4': {'block_units': [3, 8, 36, 3], 'width_factor': 4},
    'BiT-S-R50x1': {'block_units': [3, 4, 6, 3], 'width_factor': 1},
    'BiT-S-R50x3': {'block_units': [3, 4, 6, 3], 'width_factor': 3},
    'BiT-S-R101x1': {'block_units': [3, 4, 23, 3], 'width_factor': 1},
    'BiT-S-R101x3': {'block_units': [3, 4, 23, 3], 'width_factor': 3},
    'BiT-S-R152x2': {'block_units': [3, 8, 36, 3], 'width_factor': 2},
    'BiT-S-R152x4': {'block_units': [3, 8, 36, 3], 'width_factor': 4},
}