import numpy as np
import argparse
import sys

# custom modules
from utils.options import Options
from utils.factory import EnvDict, CircuitDict, AgentDict

# 0. Take in arguments
description = 'An implementation of DeepMind\'s DNC and NTM using PyTorch'
parser = argparse.ArgumentParser(description=description)
help = 'Mode: Train (1) or Test (2). Default mode is test'
parser.add_argument('--mode', action='store', dest='mode', type=int, help=help)

# Task & Network argument
help = 'Config: Choose which configuration (task & network) to use. Available:'
help += '\n(repeat-copy, vanilla) : 0\n(copy, NTM) : 1\n (repeat-copy, NTM) : '
help += '2 \n(repeat-copy, DNC) : 3 '
help += '\nDefault : 1'
parser.add_argument('--config', action='store', dest='config', type=int,
                    help=help)

# Visualization enable switch
help = 'Use this switch to enable online visualizations'
parser.add_argument('--visualize', action='store_true',
                    dest='visualize', help=help)

# GPU enable switch
help = 'Use this switch to *try* to use CUDA. Note, if your machine does not'
help += ' have CUDA + a card, CPU only will be used'
parser.add_argument('--gpu', action='store_true', dest='gpu', help=help)

# Model name argument
help = 'The name of the model to use (not the path). By default, uses the '
help += 'value in utils/options.py\n'
help += 'If a different model is used, it must be in the models directory.'
parser.add_argument('--model', action='store', dest='model', type=str,
                    help=help)

# Read Heads
help = 'The number of read heads to use.'
parser.add_argument('--read_heads', action='store', dest='num_read_heads',
                    type=int, help=help)

# Write Heads
help = 'The number of write heads to use.'
parser.add_argument('--write_heads', action='store', dest='num_write_heads',
                    type=int, help=help)

# Number of memory slots for soft addressable memory
help = 'The number of memory slots to use.'
parser.add_argument('--mem_slots', action='store', dest='num_mem_slots',
                    type=int, help=help)

# Width of a memory slot for soft addressable memory
help = 'The width of a memory slot.'
parser.add_argument('--mem_width', action='store', dest='mem_width',
                    type=int, help=help)

# The Controller size (in hidden units)
help = 'The number of hidden units in the controller network'
parser.add_argument('--controller_size', action='store',
                    dest='controller_size', type=int, help=help)

# The Learning Rate
help = 'The learning rate to use (in the form xe-y)'
parser.add_argument('--lr', action='store', dest='lr', type=float, help=help)

args = parser.parse_args()
# 1. setting up
opt = Options(**vars(args)) #unpack the arguments
np.random.seed(opt.seed)

# 2. env     (prototype)
env_prototype     = EnvDict[opt.env_type]
# 3. circuit (prototype)
circuit_prototype = CircuitDict[opt.circuit_type]
# 4. agent
agent = AgentDict[opt.agent_type](opt.agent_params,
                                  env_prototype     = env_prototype,
                                  circuit_prototype = circuit_prototype)
if args.gpu == True:
    circuit_prototype = torch.nn.DataParallel(circuit_prototype).cuda()

# 6. fit model
if opt.mode == 1:   # train
    agent.fit_model()
elif opt.mode == 2: # test opt.model_file
    agent.test_model()
