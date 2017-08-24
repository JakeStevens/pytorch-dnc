import numpy as np
import argparse
import sys

# custom modules
from utils.options import Options
from utils.factory import EnvDict, CircuitDict, AgentDict

# 0. Take in arguments
description = 'An implementation of DeepMind\'s DNC and NTM using PyTorch'
parser = argparse.ArgumentParser(description=description)
help = 'Mode: Train (1) or Test (2). Default mode is train.'
parser.add_argument('--mode', action='store', dest='mode', type=int, help=help)

help = 'Config: Choose which configuration (task & network) to use. Available:'
help += '\n(repeat-copy, vanilla) : 0\n(copy, NTM) : 1\n (repeat-copy, NTM) : '
help += '2 \n(repeat-copy, DNC) : 3 '
help += '\nDefault : 1'
parser.add_argument('--config', action='store', dest='config', type=int,
                    help=help)

help = 'Use this switch to enable online visualizations'
parser.add_argument('--visualize', action='store_true',
                    dest='visualize', help=help)

help = 'The name of the model to use (not the path). By default, uses the '
help += 'value in utils/options.py\n'
help += 'If a different model is used, it must be in the models directory.'
parser.add_argument('--model', action='store', dest='model', type=str,
                    help=help)

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
# 6. fit model
if opt.mode == 1:   # train
    agent.fit_model()
elif opt.mode == 2: # test opt.model_file
    agent.test_model()
