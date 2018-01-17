from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable

class Head(nn.Module):
    def __init__(self, args):
        super(Head, self).__init__()
        # logging
        self.logger = args.logger
        # params
        self.visualize = args.visualize
        if self.visualize:
            self.vis      = args.vis
            self.refs     = args.refs
            self.win_head = None
        self.use_cuda = args.use_cuda
        self.dtype = args.dtype
        # params
        self.num_heads = args.num_heads
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.mem_hei = args.mem_hei
        self.mem_wid = args.mem_wid
        self.num_allowed_shifts = args.num_allowed_shifts

        # Initialize timer variables
        self.key_similarity_time = 0
        self.content_weighting_time = 0
        self.location_interpolation_time = 0
        self.shift_weighting_time = 0
        self.sharpen_weight_time = 0
        self.read_time = 0
        self.write_time = 0
        self.addressing_nn_time = 0

        # Initialize access counters
        self.num_read_operations = 0
        self.num_write_operations = 0

    def __del__(self):
        print("Heads NN time: " + str(self.addressing_nn_time))
        print("Key Similarity Time: " + str(self.key_similarity_time))
        print("Content Weighting Time: " + str(self.content_weighting_time))
        print("Location Interpolation Time: " + str(self.location_interpolation_time))
        print("Shift Weighting Time: " + str(self.shift_weighting_time))
        print("Sharpen Weight Time: " + str(self.sharpen_weight_time))
        print("Read Time: " + str(self.read_time))
        print("Write Time: " + str(self.write_time))
        print("Read Operations: " + str(self.num_read_operations))
        print("Write Operations: " + str(self.num_write_operations))

    def _reset_states(self):
        self.wl_prev_vb = Variable(self.wl_prev_ts).type(self.dtype) # batch_size x num_heads x mem_hei

    def _reset(self):           # NOTE: should be called at each child's __init__
        # self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        # TODO: see how this one is reset
        # reset internal states
        self.wl_prev_ts = torch.eye(1, self.mem_hei).unsqueeze(0).expand(self.batch_size, self.num_heads, self.mem_hei)
        self._reset_states()

    def _visual(self):
        raise NotImplementedError("not implemented in base calss")

    def _access(self, memory_vb):
        # NOTE: called at the end of forward, to use the weight to read/write from/to memory
        raise NotImplementedError("not implemented in base calss")
