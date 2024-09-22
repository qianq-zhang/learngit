#生成一个NMS代码
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import os
import time
import random
import math
import copy
from PIL import Image
from torchvision import transforms
from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchvision import datasets
