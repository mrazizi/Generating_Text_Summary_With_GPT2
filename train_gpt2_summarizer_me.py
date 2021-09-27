import argparse
from datetime import datetime
import os
import time

import numpy as np
from transformers import GPT2LMHeadModel,AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tnrange, tqdm_notebook

from dataset import GPT21024Dataset
from utils import add_special_tokens, generate_sample, set_seed
