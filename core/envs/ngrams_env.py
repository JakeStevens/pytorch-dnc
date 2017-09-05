from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools
from random import randint
import torch
import os
import pickle

from core.env import Env

class NgramsEnv(Env):
    def __init__(self, args, env_ind=0):
        super(NgramsEnv, self).__init__(args, env_ind)

        # state space setup
        self.batch_size = args.batch_size
        self.len_word = args.len_word
        self.min_num_words = args.min_num_words
        self.max_num_words = args.max_num_words
        self.table = None
        self.logger.warning("Word     {length}:   {%s}", self.len_word)
        self.logger.warning("Words #  {min, max}: {%s, %s}", self.min_num_words, self.max_num_words)

    def _preprocessState(self, state):
        # NOTE: state input in size: batch_size x ((num_words+1)*height_word) x len_word
        # NOTE: we return as:        ((num_words+1)*height_word) x batch_size x len_word
        # NOTE: to ease feeding in one row from all batches per forward pass
        for i in range(len(state)):
            state[i] = np.transpose(state[i], (1, 0, 2))
        return state

    # Note: The following ngram functions are taken from the open source
    # "diffmem" project by DoctorTeeth. The code is available online at
    # https://github.com/DoctorTeeth/diffmem
    def _ngram_table(self, bits):
      """
      Generate a table that contains the probability of seeing a 1
      after having seen a context of length bits-1.
      """
      # If there is a serialized object for this model, then
      # it is the ngram table and it should be used
      if self.table:
        return self.table
      if os.path.exists(self.model_name):
          with open(self.model_name, 'r') as f:
              self.table = pickle.load(f)
              return self.table
      with open(self.model_name, 'w') as f:
          assert(bits >= 2)
          prod = itertools.product([0,1], repeat=bits-1)
          table = {}
          for p in prod:
            table[p] = np.random.beta(0.5,0.5)
          pickle.dump(table, f)
          self.table = table
          return table
    
    def _sample_ngram(self, table, context, n):
      """
      Given an n-gram table and the current context, generate the next bit.
      """
      p = table[tuple(context.reshape(1,n-1).tolist()[0])]
      r = np.random.uniform(low=0, high=1)
      if r > p:
        return 1
      else:
        return 0
    
    def _ngrams(self, seq_len, n):
      """
      Implements the dynamic n-grams task - section 4.4 from the paper.
    
      For every new training model, we generate a transition table
      using ngram_table for n bits. Otherwise, we use the transition table
      for the current model
    
      We then sample from that to generate a sequence of length seq_len.
    
      The task is simply to predict the next bit.
      """
      beta_table = self._ngram_table(n)
      inputs  = np.zeros((seq_len,1))
      outputs = np.zeros((seq_len,1))
    
      for i in range(seq_len):
        if i < n - 1:
          inputs[i] = np.random.binomial(1, 0.5)
        else:
          inputs[i] = self._sample_ngram(beta_table, inputs[i - n + 1:i],n)
      # the task is simply to predict the next bit
      outputs[:-1] = inputs[1:]
      return inputs, outputs

    @property
    def state_shape(self):
        # NOTE: we use this as the input_dim to be consistent with the sl & rl tasks
        return self.len_word + 2

    @property
    def action_dim(self):
        # NOTE: we use this as the output_dim to be consistent with the sl & rl tasks
        return self.len_word

    def render(self):
        pass

    def _readable(self, datum):
        return '+' + ' '.join(['-' if x == 0 else '%d' % x for x in datum]) + '+'

    def visual(self, input_ts, target_ts, mask_ts, output_ts=None):
        """
        input_ts:  [(num_words+2) x batch_size x (len_word+2)]
        target_ts: [(num_words+2) x batch_size x (len_word)]
        mask_ts:   [(num_words+2) x batch_size x 1]
        output_ts: [(num_words+2) x batch_size x (len_word)]
        """
        output_ts = torch.round(output_ts * mask_ts) if output_ts is not None else None
        input_strings  = [self._readable(input_ts[:, 0, i])  for i in range(input_ts.size(2))]
        target_strings = [self._readable(target_ts[:, 0, i]) for i in range(target_ts.size(2))]
        mask_strings   = [self._readable(mask_ts[:, 0, 0])]
        output_strings = [self._readable(output_ts[:, 0, i]) for i in range(output_ts.size(2))] if output_ts is not None else None
        input_strings  = 'Input:\n'  + '\n'.join(input_strings)
        target_strings = 'Target:\n' + '\n'.join(target_strings)
        mask_strings   = 'Mask:\n'   + '\n'.join(mask_strings)
        output_strings = 'Output:\n' + '\n'.join(output_strings) if output_ts is not None else None
        # strings = [input_strings, target_strings, mask_strings, output_strings]
        # self.logger.warning(input_strings)
        # self.logger.warning(target_strings)
        # self.logger.warning(mask_strings)
        # self.logger.warning(output_strings)
        print(input_strings)
        print(target_strings)
        print(mask_strings)
        print(output_strings) if output_ts is not None else None

    def sample_random_action(self):
        pass

    def _generate_sequence(self):
        """
        generates [batch_size x num_words x len_word] data and
        prepare input & target & mask

        Returns:
        exp_state1[0] (input) : starts w/ a start bit, then the seq to be copied
                              : then an repeat flag, then 0's
            [0 ... 0, 1, 0;   # start bit
             data   , 0, 0;   # data with padded 0's
             0 ... 0, 0, 1;   # end bit
             0 ......... 0]   # height_word rows of 0's
        exp_state1[1] (target): 0's until after inputs has the repeat flag, then
                              : the seq to be copied, then an end bit
            [0 ... 0, 0;      # num_words+2 rows of 0's
             sorted , 0;      # data
        exp_state1[2] (mask)  : 1's for all row corresponding to the target
                              : 0's otherwise}
            [0;               # num_words+2 rows of 0's
             1];              # num_words rows of 1's
        NOTE: we pad extra rows of 0's to the end of those batches with smaller
        NOTE: length to make sure each sample in one batch has the same length
        """
        self.exp_state1 = []
        # we prepare input, target, mask for each batch
        batch_num_words     = np.random.randint(self.min_num_words, self.max_num_words+1, size=(self.batch_size))
        max_batch_num_words = np.max(batch_num_words)
        self.exp_state1.append(np.zeros((self.batch_size, max_batch_num_words+2, self.len_word + 2))) # input
        self.exp_state1.append(np.zeros((self.batch_size, max_batch_num_words+2, self.len_word)))     # target
        self.exp_state1.append(np.zeros((self.batch_size, max_batch_num_words+2, 1)))                 # mask
        for batch_ind in range(self.batch_size):
            num_words = batch_num_words[batch_ind]
            data, target = self._ngrams(num_words, 6) #TODO: Change 6 to self.ngram_n
            data = np.array(data)
            target = np.array(target)
            # prepare input  for this sample
            self.exp_state1[0][batch_ind][0][-2] = 1                        # set start bit
            self.exp_state1[0][batch_ind][1:num_words + 1][:] = data
            self.exp_state1[0][batch_ind][num_words + 1][-1] = 1  # set end bit
            # prepare target for this sample
            self.exp_state1[1][batch_ind][1:num_words+1, :] = target 
            # prepare mask   for this sample
            self.exp_state1[2][batch_ind][1:num_words+1, :] = 1

    def reset(self):
        self._reset_experience()
        self._generate_sequence()
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        self._generate_sequence()
        return self._get_experience()
