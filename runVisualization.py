import re
import hickle as hkl
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from visualize import make_dot


params = hkl.load('resnet-18-export.hkl')

# convert numpy arrays to torch Variables
for k in sorted(params.keys()):
    v = params[k]
    print k, v.shape
    params[k] = Variable(torch.from_numpy(v), requires_grad=True)