# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:12:41 2024

@author: Andrei
"""
import numpy as np
from utilities import split_in_blocks

import tensorflow as tf
import tensorflow.keras.layers as layers

model = layers.Sequential()
model.add(layers.InputLayer(shape=(1, 34)))
model.add(layers.Dense(1024))
model.add(layers.Dense(512))
model.add(layers.Dense(20))

model.summary()