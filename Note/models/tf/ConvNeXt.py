import tensorflow as tf
from Note import nn
import numpy as np


class ConvNeXtBlock:
    def __init__(self, input_channels, projection_dim, drop_path_rate=0.0, layer_scale_init_value=1e-6, dtype='float32'):
        self.conv2d=nn.conv2d(projection_dim,[7,7],input_channels//projection_dim,padding='SAME',dtype=dtype)
        self.layer_norm=nn.layer_norm(self.conv2d.output_size,dtype=dtype)
        self.dense1=nn.dense(4*projection_dim,projection_dim,activation='gelu',dtype=dtype)
        self.dense2=nn.dense(projection_dim,4*projection_dim,dtype=dtype)
        self.gamma=tf.Variable(tf.ones([projection_dim],dtype=dtype)*layer_scale_init_value)
        self.drop_path_rate=drop_path_rate
        self.layer_scale_init_value=layer_scale_init_value
        self.output_size=projection_dim
        nn.Model.layer_list.append(self)
        self.training=True
        
    
    def LayerScale(self,x):
        return x * self.gamma
    
    
    def StochasticDepth(self,x):
        if self.training:
            keep_prob = 1 - self.drop_path_rate
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1, dtype=x.dtype)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    
    
    def __call__(self,data):
        x=self.conv2d(data)
        x=self.layer_norm(x)
        x=self.dense1(x)
        x=self.dense2(x)
        if self.layer_scale_init_value is not None:
            x=self.LayerScale(x)
        if self.drop_path_rate:
            x=self.StochasticDepth(x)
        else:
            x=x
        return data + x


class ConvNeXt(nn.Model):
    def __init__(self,model_type='base',drop_path_rate=0.0,layer_scale_init_value=1e-6,classes=1000,include_top=True,pooling=None):
        super().__init__()
        self.model_type=model_type
        self.classes=classes
        self.depths=MODEL_CONFIGS[model_type]['depths']
        self.projection_dims=MODEL_CONFIGS[model_type]['projection_dims']
        self.drop_path_rate=drop_path_rate
        self.layer_scale_init_value=layer_scale_init_value
        self.include_top=include_top
        self.pooling=pooling
        
        # Stem block.
        layers=nn.Sequential()
        layers.add(nn.conv2d(self.projection_dims[0],[4,4],3))
        layers.add(nn.layer_norm())
        
        # Downsampling blocks.
        self.downsample_layers = []
        self.downsample_layers.append(layers)
        
        num_downsample_layers = 3
        for i in range(num_downsample_layers):
            layers=nn.Sequential()
            layers.add(nn.layer_norm(self.projection_dims[i]))
            layers.add(nn.conv2d(self.projection_dims[i+1],[2,2],self.projection_dims[i]))
            self.downsample_layers.append(layers)
        
        # Stochastic depth schedule.
        # This is referred from the original ConvNeXt codebase:
        # https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py#L86
        depth_drop_rates = [
            float(x) for x in np.linspace(0.0, self.drop_path_rate, sum(self.depths))
        ]
        
        # First apply downsampling blocks and then apply ConvNeXt stages.
        cur = 0
        
        self.blocks=[]
        self.num_convnext_blocks = 4
        for i in range(self.num_convnext_blocks):
            input_channels=self.downsample_layers[i].output_size
            for j in range(self.depths[i]):
                block = nn.Sequential()
                block.add(ConvNeXtBlock(
                    input_channels=input_channels,
                    projection_dim=self.projection_dims[i],
                    drop_path_rate=depth_drop_rates[cur + j],
                    layer_scale_init_value=self.layer_scale_init_value,
                    ))
                input_channels=block.output_size
            self.blocks.append(block)
            cur += self.depths[i]
        self.layer_norm=nn.layer_norm(self.blocks[-1].output_size)
        self.head=self.dense(self.classes,self.blocks[-1].output_size)
    
    
    def __call__(self,data):
        for i in range(self.num_convnext_blocks):
            data = self.downsample_layers[i](data)
            for j in range(self.depths[i]):
                data=self.blocks[i](data)
        if self.include_top:
            data = tf.math.reduce_mean(data, axis=[1, 2])
            data = self.layer_norm(data)
            data = self.head(data)
        else:
            if self.pooling=="avg":
                data = tf.math.reduce_mean(data, axis=[1, 2])
            elif self.pooling=="max":
                data = tf.math.reduce_max(data, axis=[1, 2])
        return data


MODEL_CONFIGS = {
    "tiny": {
        "depths": [3, 3, 9, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "small": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [96, 192, 384, 768],
    },
    "base": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [128, 256, 512, 1024],
    },
    "large": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [192, 384, 768, 1536],
    },
    "xlarge": {
        "depths": [3, 3, 27, 3],
        "projection_dims": [256, 512, 1024, 2048],
    },
}