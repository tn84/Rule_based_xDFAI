from tqdm.notebook import tqdm
from PIL import Image
import copy
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import Dataset, TensorDataset
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pickle
import json
from functools import reduce
from collections import Counter
import sys
import math


class_name = "List of the classes of the dataset"  #e.g. ['1','2','3']
