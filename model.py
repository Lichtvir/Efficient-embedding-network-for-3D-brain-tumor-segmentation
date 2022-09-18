import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow import keras
import numpy as np 
import os
from tensorflow.keras.optimizers import *


def Conv_block(inp, filters, data_format='channels_last', name=None):
  
    inp_res = Conv3D(
        filters=filters,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format=data_format,
        name=f'Res_{name}' if name else None)(inp)

    x = tfa.layers.GroupNormalization(
        groups=8,
        name=f'GroupNorm_1_{name}' if name else None)(inp)

    x = Activation('relu', name=f'Relu_1_{name}' if name else None)(x)
    
    x = Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name=f'Conv3D_1_{name}' if name else None)(x)

    x = tfa.layers.GroupNormalization(
        groups=8,
        name=f'GroupNorm_2_{name}' if name else None)(x)
    
    x = Activation('relu', name=f'Relu_2_{name}' if name else None)(x)
    
    x = Conv3D(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format=data_format,
        name=f'Conv3D_2_{name}' if name else None)(x)

    out = Add(name=f'Out_{name}' if name else None)([x, inp_res])
    
    return out
  
  
  
  
  def model_func(input_shape=(192, 160, 144, 4), output_channels=4, trainable_eff=True):
    """
    Parameters
    ----------
    `input_shape`: A 4-tuple, optional.
    `output_channels`: An integer, optional.
    Returns
    -------
    `model`: A keras.models.Model instance
    """
    
    H, W, D, c = input_shape
    assert len(input_shape) == 4, "Input shape must be a 4-tuple"

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------

    ## Input Layer
    Inputs = Input(input_shape)
    ## The Initial Block
    print(Inputs)
    
    x = Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        strides=(1,1,1),
        padding='same',
        data_format='channels_last',
        name='Input_L1')(Inputs)
   
    x = SpatialDropout3D(0.2, data_format='channels_last')(x)

    # -------------------------------
    
    x1 = Conv_block(x, 16, name='L1')
    
    x = Conv3D(
        filters=32,
        kernel_size=(3, 3, 3),
        strides=(1,1,3),
        padding='same',
        data_format='channels_last',
        name='Post_L1')(x1)
    
    # -------------------------------
    x2 = Conv_block(x, 32, name='L2')
    
    x = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        strides=(1,1,4),
        padding='same',
        data_format='channels_last',
        name='Post_L2')(x2)
    
    # -------------------------------

    x3 = Conv_block(x, 64, name='L3')
    
    x = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        strides=(1,1,4),
        padding='same',
        data_format='channels_last',
        name='Post_L3')(x3) 

    ## --------------------------------
    
    x = Conv_block(x, 64, name='L4')
    


    F = Conv3D(1, (1,1,1), 
                               activation='relu',
                               data_format='channels_last',
                               name='Enc_DownSample_128')(x)
    # 3D => 2D :
    print('befor reshape ', F.shape)
    CHANNELS = 3
    outputs1 = Reshape((input_shape[0],input_shape[1],CHANNELS))(F)
    print('after reshape', outputs1.shape)

    # -------------------------------------------------------------------------
                               # EffecientNet Ecoder
    # -------------------------------------------------------------------------
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=[input_shape[0],input_shape[1],CHANNELS], weights='imagenet', classifier_activation= None,
        include_top=False)

    # Activated layers for skip connection : 
    layer_names = [
        'block2a_expand_activation',   
        'block3a_expand_activation',  
        'block4a_expand_activation',   
        'block6a_expand_activation',
        'top_activation',     
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = trainable_eff

    # -------------------------------------------------------------------------
                           # Skip Connections
    # -------------------------------------------------------------------------

    def upsample(filters, size, apply_dropout=False):
      initializer = tf.random_normal_initializer(0., 0.02)
      result = tf.keras.Sequential()
      result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        use_bias=False))
      result.add(tf.keras.layers.BatchNormalization())
      if apply_dropout:
          result.add(tf.keras.layers.SpatialDropout2D(0.5))
          
      result.add(tf.keras.layers.ReLU())
      return result
  
    up_stack = [
        upsample(512, 3, apply_dropout=True),  
        upsample(256, 3, apply_dropout=True),  
        upsample(128, 3, apply_dropout=True),  
        upsample(64, 3, apply_dropout=True),   
    ]

    # -------------------------------------------------------------------------
                            # 2D Decoder
    # -------------------------------------------------------------------------


    x = outputs1
    # Downsampling through the model
    skips = down_stack(x)
    x_cod = skips[-1]

    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])
      
     # This is the last layer of the model
    Int_size = 3
    last = tf.keras.layers.Conv2DTranspose(
        Int_size, 3, strides=2,
        padding='same')  
    Outputs2 = last(x)
    
    x = tf.keras.layers.Reshape((input_shape[0],input_shape[1],3,1))(Outputs2)

    # -------------------------------------------------------------------------
                           # 3D Decoder
    # -------------------------------------------------------------------------
    
    x = Conv3D(
        filters=64,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='L1_Dec')(x) 
    
    x = UpSampling3D(
        size=(1,1,4),
        data_format='channels_last',
        name='Up_L1_Dec')(x)
    
    x = Add(name='Add1')([x, x3])
    
    x = Conv_block(x, 64, name='Post_L1_Dec')

    # ------------------------------------------------------------------------
    x = Conv3D(
        filters=32,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='L2_Dec')(x)
    
    x = UpSampling3D(
        size=(1,1,4),
        data_format='channels_last',
        name='Up_L2_Dec')(x)
    
    x = Add(name='Add2')([x, x2])
    
    x = Conv_block(x, 32, name='Post_L2_Dec')

    # -----------------------------------------------------------------------
    
    x = Conv3D(
        filters=16,
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        name='L3_Dec')(x)
    
    x = UpSampling3D(
        size=(1,1,3),
        data_format='channels_last',
        name='Up_L3_Dec')(x)
    
    x = Add(name='Add3')([x, x1])
    
    x = Conv_block(x, 16, name='Post_L3_Dec')

    # ----------------------------------------------------------------------
    
    x = Conv3D(
        filters=16,
        kernel_size=(3, 3, 3),
        strides=1,
        padding='same',
        data_format='channels_last',
        name='L4_Dec')(x)


    Outputs = Conv3D(
        filters=4,  # No. of tumor classes is 3
        kernel_size=(1, 1, 1),
        strides=1,
        data_format='channels_last',
        activation='softmax',
        name='Model_Outputs')(x)

   
    model = Model(inputs = Inputs, outputs=Outputs)  # Create the model
    model.compile(optimizer=Nadam(1e-4),
        loss= my_loss,
        metrics=my_metric
    )

    return model
  
  
  model = model_func(trainable_eff=True)
