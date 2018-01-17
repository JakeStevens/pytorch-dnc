import numpy as np
import itertools

def ngram_table(bits):
  """
  Generate a table that contains the probability of seeing a 1
  after having seen a context of length bits-1.
  """
  assert(bits >= 2)
  prod = itertools.product([0,1], repeat=bits-1)
  table = {}
  for p in prod:
    table[p] = np.random.beta(0.5,0.5)
  return table

def sample_ngram(table, context, n):
  """
  Given an n-gram table and the current context, generate the next bit.
  """
  p = table[tuple(context.reshape(1,n-1).tolist()[0])]
  r = np.random.uniform(low=0, high=1)
  if r > p:
    return 1
  else:
    return 0

def ngrams(seq_len, n):
  """
  Implements the dynamic n-grams task - section 4.4 from the paper.

  For every new training sequence, we generate a transition table
  using ngram_table for n bits.

  We then sample from that to generate a sequence of length seq_len.

  The task is simply to predict the next bit.
  """
  beta_table = ngram_table(n)
  inputs  = np.zeros((seq_len+1,1))
  outputs = np.zeros((seq_len+1,1))

  for i in range(seq_len):
    if i < n - 1:
      inputs[i] = np.random.randint(2)
    else:
      inputs[i] = sample_ngram(beta_table, inputs[i - n + 1:i],n)
  # the task is simply to predict the next bit
  outputs[:-1] = inputs[:-1]
  outputs[-1] = sample_ngram(beta_table, inputs[i - n + 1 : i], n)
  return inputs, outputs

inputs, outputs = ngrams(22, 5)
print inputs
print outputs

