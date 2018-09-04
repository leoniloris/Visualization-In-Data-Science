import keras
from keras.layers import *
from keras.layers.merge import concatenate
from keras.models import load_model
from keras import regularizers
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K

import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from pathlib import Path
import imageio
from IPython.display import Image
import plot
import importlib

importlib.reload(plot)

import numpy as np

import ipywidgets as widgets
from ipywidgets import interact, FloatSlider