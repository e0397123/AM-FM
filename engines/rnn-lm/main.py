# # Deep LSTMs with Word2vec using RNN API

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import nltk
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.impute import KNNImputer
import csv
import word2vec
from data import *

imputer = KNNImputer(n_neighbors=2, weights="uniform")

def maybe_download(filename):
  """Download a file if not present"""
  print('Downloading file: ', dir_name+ os.sep+filename)
    
  if not os.path.exists(dir_name+os.sep+filename):
    filename, _ = urlretrieve(url + filename, dir_name+os.sep+filename)
  else:
    print('Not downloading. File already exists.')
#  statinfo = os.stat(dir_name+os.sep+filename)
  
  return filename

# Learning rate decay logic
def decay_learning_rate(session, v_perplexity):
  global decay_threshold, decay_count, min_perplexity  
  # Decay learning rate
  if v_perplexity < min_perplexity:
    decay_count = 0
    min_perplexity= v_perplexity
  else:
    decay_count += 1

  if decay_count >= decay_threshold:
    print('\t Reducing learning rate')
    decay_count = 0
    session.run(inc_gstep)

# ### Defining the Beam Prediction Logic
# Here we define function that takes in the session as an argument and output a beam of predictions

test_word = None

def get_beam_prediction(session):
    
    global test_word
    global sample_beam_predictions, update_sample_beam_state
    
    # Generating words within a segment with Beam Search
    # To make some calculations clearer, we use the example as follows
    # We have three classes with beam_neighbors=2 (best candidate denoted by *, second best candidate denoted by `)
    # For simplicity we assume best candidate always have probability of 0.5 in output prediction
    # second best has 0.2 output prediction
    #           a`                   b*                   c                <--- root level
    #    /     |     \         /     |     \        /     |     \   
    #   a      b      c       a*     b`     c      a      b      c         <--- depth 1
    # / | \  / | \  / | \   / | \  / | \  / | \  / | \  / | \  / | \
    # a b c  a b c  a b c   a*b c  a`b c  a b c  a b c  a b c  a b c       <--- depth 2
    # So the best beams at depth 2 would be
    # b-a-a and b-b-a

    

    # Calculate the candidates at the root level
    feed_dict = {}
    for b_n_i in range(beam_neighbors):
        feed_dict.update({sample_beam_inputs[b_n_i]: [test_word]})

    # We calculate sample predictions for all neighbors with the same starting word/character
    # This is important to update the state for all instances of beam search
    sample_preds_root = session.run(sample_beam_predictions, feed_dict = feed_dict)  
    sample_preds_root = sample_preds_root[0]

    # indices of top-k candidates
    # b and a in our example (root level)
    this_level_candidates_sorted =  (np.argsort(sample_preds_root,axis=1).ravel()[::-1]).tolist() # indices of top-k candidates
    this_level_candidates = []
    for c in this_level_candidates_sorted:
        if len(this_level_candidates)==beam_neighbors:
            break
        if c!=0:
            this_level_candidates.append(c)

    this_level_candidates = np.array(this_level_candidates)

    # probabilities of top-k candidates
    # 0.5 and 0.2
    this_level_probs = sample_preds_root[0,this_level_candidates] #probabilities of top-k candidates

    # Update test sequence produced by each beam from the root level calculation
    # Test sequence looks like for our example (at root)
    # [b,a]
    test_sequences = ['' for _ in range(beam_neighbors)]
    for b_n_i in range(beam_neighbors):
        test_sequences[b_n_i] += reverse_dictionary[this_level_candidates[b_n_i]] + ' '

    # Make the calculations for the rest of the depth of the beam search tree
    for b_i in range(beam_length-1):
        test_words = [] # candidate words for each beam
        pred_words = [] # Predicted words of each beam

        # computing feed_dict for the beam search (except root)
        # feed dict should contain the best words/chars/bigrams found by the previous level of search

        # For level 1 in our example this would be
        # sample_beam_inputs[0]: b, sample_beam_inputs[1]:a
        feed_dict = {}
        for p_idx, pred_i in enumerate(this_level_candidates):                    
            # Updating the feed_dict for getting next predictions
            test_words.append(this_level_candidates[p_idx])

            feed_dict.update({sample_beam_inputs[p_idx]:[test_words[p_idx]]})

        # Calculating predictions for all neighbors in beams
        # This is a list of vectors where each vector is the prediction vector for a certain beam
        # For level 1 in our example, the prediction values for 
        #      b             a  (previous beam search results)
        # [a,  b,  c],  [a,  b,  c] (current level predictions) would be
        # [0.1,0.1,0.1],[0.5,0.2,0]
        sample_preds_all_neighbors = session.run(sample_beam_predictions, feed_dict=feed_dict)

        # Create a single vector with 
        # Making our example [0.1,0.1,0.1,0.5,0.2,0] 
        sample_preds_all_neighbors_concat = np.concatenate(sample_preds_all_neighbors,axis=1)

        # Update this_level_candidates to be used for the next iteration
        # And update the probabilities for each beam
        # In our example these would be [3,4] (indices with maximum value from above vector)
        # We also use a simple trick to avoid UNK (word id 0) being predicted 
        this_level_candidates_sorted = np.argsort(sample_preds_all_neighbors_concat.ravel())[::-1]
        this_level_candidates = []
        for c in this_level_candidates_sorted:
            if len(this_level_candidates)==beam_neighbors:
                break
            if c!=0 and c%vocabulary_size != 0 :
                this_level_candidates.append(c)

        this_level_candidates = np.array(this_level_candidates)

        # In the example this would be [1,1]
        parent_beam_indices = this_level_candidates//vocabulary_size

        # normalize this_level_candidates to fall between [0,vocabulary_size]
        # In this example this would be [0,1]
        this_level_candidates = (this_level_candidates%vocabulary_size).tolist()

        # Here we update the final state of each beam to be
        # the state that was at the index 1. Because for both the candidates at this level the parent is 
        # at index 1 (that is b from root level)
        session.run(update_sample_beam_state, feed_dict={best_neighbor_beam_indices: parent_beam_indices})

        # Here we update the joint probabilities of each beam and add the newly found candidates to the sequence
        tmp_this_level_probs = np.asarray(this_level_probs)
        tmp_test_sequences = list(test_sequences)

        for b_n_i in range(beam_neighbors):
            # We make the b_n_i element of this_level_probs to be the probability of parents
            # In the example the parent indices are [1,1]
            # So this_level_probs become [0.5,0.5]
            this_level_probs[b_n_i] = tmp_this_level_probs[parent_beam_indices[b_n_i]]

            # Next we multipyle these by the probabilities of the best candidates from current level 
            # [0.5*0.5, 0.5*0.2] = [0.25,0.1]
            this_level_probs[b_n_i] *= sample_preds_all_neighbors[parent_beam_indices[b_n_i]][0,this_level_candidates[b_n_i]]

            # Make the b_n_i element of test_sequences to be the correct parent of the current best candidates
            # In the example this becomes [b, b]
            test_sequences[b_n_i] = tmp_test_sequences[parent_beam_indices[b_n_i]]

            # Now we append the current best candidates
            # In this example this becomes [ba,bb]
            test_sequences[b_n_i] += reverse_dictionary[this_level_candidates[b_n_i]] + ' '

            # Create one-hot-encoded representation for each candidate
            pred_words.append(this_level_candidates[b_n_i])

    # Calculate best beam id based on the highest beam probability
    # Using the highest beam probability always lead to very monotonic text
    # Let us sample one randomly where one being sampled is decided by the likelihood of that beam
    rand_cand_ids = np.argsort(this_level_probs)[-3:]
    if True in np.isnan(this_level_probs):
        this_level_probs = imputer.fit_transform(np.expand_dims(this_level_probs, axis=0))
        this_level_probs = this_level_probs.squeeze()
    rand_cand_probs = this_level_probs[rand_cand_ids]/np.sum(this_level_probs[rand_cand_ids])
    random_id = np.random.choice(rand_cand_ids, p=rand_cand_probs)

    best_beam_id = parent_beam_indices[random_id]
    # Update state and output variables for test prediction
    session.run(update_sample_beam_state,feed_dict={best_neighbor_beam_indices:[best_beam_id for _ in range(beam_neighbors)]})

    # Make the last word/character/bigram from the best beam
    test_word = pred_words[best_beam_id]
        
    return test_sequences[best_beam_id]


num_files = 2

dir_name = '../../data/twitter'

filenames = ['train_clean_10k.txt', 'valid_clean.txt']

# -------------------------------------commented section ---------------------------------------------

# # ## Downloading Stories
# # Stories are automatically downloaded from https://www.cs.cmu.edu/~spok/grimmtmp/, if not detected in the disk. The total size of stories is around ~500KB. The dataset consists of 100 stories.

# url = 'https://www.cs.cmu.edu/~spok/grimmtmp/'

# # Create a directory if needed
# dir_name = 'stories'
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)
    
# num_files = 100
# filenames = [format(i, '03d')+'.txt' for i in range(1,num_files+1)]

# for fn in filenames:
#     maybe_download(fn)


# for i in range(len(filenames)):
#     file_exists = os.path.isfile(os.path.join(dir_name,filenames[i]))
#     assert file_exists
# print('%d files found.'%len(filenames))


# -------------------------------------commented section ---------------------------------------------


# ## Reading data
# Data will be stored in a list of lists where the each list represents a document and document is a list of words. We will then break the text into words.

global documents

documents = []

for i in range(num_files):    
    print('\nProcessing file %s'%os.path.join(dir_name,filenames[i]))
    
    words = read_data(os.path.join(dir_name,filenames[i]))
    
    documents.append(words)
    print('Data size (Characters) (Document %d) %d' %(i,len(words)))
    print('Sample string (Document %d) %s'%(i,words[:50]))


# ## Building the Dictionaries (Bigrams)
# Builds the following. To understand each of these elements, let us also assume the text "I like to go to school"
# 
# * `dictionary`: maps a string word to an ID (e.g. {I:0, like:1, to:2, go:3, school:4})
# * `reverse_dictionary`: maps an ID to a string word (e.g. {0:I, 1:like, 2:to, 3:go, 4:school}
# * `count`: List of list of (word, frequency) elements (e.g. [(I,1),(like,1),(to,2),(go,1),(school,1)]
# * `data` : Contain the string of text we read, where string words are replaced with word IDs (e.g. [0, 1, 2, 3, 2, 4])
# 
# It also introduces an additional special token `UNK` to denote rare words to are too rare to make use of.

# In[7]:


global data_list, count, dictionary, reverse_dictionary,vocabulary_size

# Print some statistics about data
data_list, count, dictionary, reverse_dictionary = build_dataset(documents)
print('Most common words (+UNK)', count[:5])
print('Least common words (+UNK)', count[-15:])
print('Sample data', data_list[0][:10])
print('Sample data', data_list[1][:10])
print('Vocabulary: ',len(dictionary))
vocabulary_size = len(dictionary)
del documents  # To reduce memory.


embedding_size = 300 # Dimension of the embedding vector.
embedding_name = 'embedings.npy'

learn_w2v(num_files, data_list, reverse_dictionary, embedding_size, vocabulary_size, embedding_name)


# Running a tiny set to see if things are correct
dg = DataGeneratorSeq(data_list[0][25:50],5,5)
u_data, u_labels = dg.unroll_batches()

# Iterate through each data batch in the unrolled set of batches
for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):   
    print('\n\nUnrolled index %d'%ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs:')
    for sing_dat in dat_ind:
        print('\t%s (%d)'%(reverse_dictionary[sing_dat],sing_dat),end=", ")
    print('\n\tOutput:')
    for sing_lbl in lbl_ind:        
        print('\t%s (%d)'%(reverse_dictionary[sing_lbl],sing_lbl),end=", ")


# Number of neurons in the hidden state variables
num_nodes = [64, 48, 32]

# Number of data points in a batch we process
batch_size = 32

# Number of time steps we unroll for during optimization
num_unrollings = 50

dropout = 0.2 # We use dropout

# Use this in the CSV filename when saving
# when using dropout
filename_extension = ''
if dropout>0.0:
    filename_extension = '_dropout'
    
filename_to_save = 'lstm_word2vec'+filename_extension+'.csv' # use to save perplexity values


# ## Defining Inputs and Outputs
# 
# In the code we define two different types of inputs. 
# * Training inputs (The stories we downloaded) (batch_size > 1 with unrolling)
# * Validation inputs (An unseen validation dataset) (bach_size =1, no unrolling)
# * Test inputs (New story we are going to generate) (batch_size=1, no unrolling)

tf.reset_default_graph()

# Training Input data.
train_inputs, train_labels = [],[]
train_labels_ohe = []
# Defining unrolled training inputs
for ui in range(num_unrollings):
    train_inputs.append(tf.placeholder(tf.int32, shape=[batch_size],name='train_inputs_%d'%ui))
    train_labels.append(tf.placeholder(tf.int32, shape=[batch_size], name = 'train_labels_%d'%ui))
    train_labels_ohe.append(tf.one_hot(train_labels[ui], vocabulary_size))
    
# Validation data placeholders
valid_inputs = tf.placeholder(tf.int32, shape=[1],name='valid_inputs')
valid_labels = tf.placeholder(tf.int32, shape=[1], name = 'valid_labels')
valid_labels_ohe = tf.one_hot(valid_labels, vocabulary_size)

# Text generation: batch 1, no unrolling.
test_input = tf.placeholder(tf.int32, shape=[1],name='test_input')


# ## Loading Word Embeddings to TensorFlow
# We load the previously learned and stored embeddings to TensorFlow and define tensors to hold embeddings

# In[26]:


## If you want to change the embedding matrix to something you newly generated,
## Simply change embeddings.npy to embeddings-tmp.npy
embed_mat = np.load(embedding_name)
embeddings_size = embed_mat.shape[1]

embed_init = tf.constant(embed_mat)
embeddings = tf.Variable(embed_init, name='embeddings')

# Defining embedding lookup operations for all the unrolled
# trianing inputs
train_inputs_embeds = []
for ui in range(num_unrollings):
    # We use expand_dims to add an additional axis
    # As this is needed later for LSTM cell computation
    train_inputs_embeds.append(tf.expand_dims(tf.nn.embedding_lookup(embeddings,train_inputs[ui]),0))

# Defining embedding lookup for operations for all the validation data
valid_inputs_embeds = tf.nn.embedding_lookup(embeddings,valid_inputs)

# Defining embedding lookup for operations for all the testing data
test_input_embeds = tf.nn.embedding_lookup(embeddings, test_input)

# ## Defining Model Parameters
# 
# Now we define model parameters. Compared to RNNs, LSTMs have a large number of parameters. Each gate (input, forget, memory and output) has three different sets of parameters.

# In[27]:


print('Defining softmax weights and biases')
# Softmax Classifier weights and biases.
w = tf.Variable(tf.truncated_normal([num_nodes[-1], vocabulary_size], stddev=0.01))
b = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))

print('Defining the LSTM cell')
# Defining a deep LSTM from Tensorflow RNN API

# First we define a list of LSTM cells
# num_nodes here is a sequence of hidden layer sizes
cells = [tf.nn.rnn_cell.LSTMCell(n) for n in num_nodes]

# We now define a dropout wrapper for each LSTM cell
dropout_cells = [
    rnn.DropoutWrapper(
        cell=lstm, input_keep_prob=1.0,
        output_keep_prob=1.0-dropout, state_keep_prob=1.0,
        variational_recurrent=True, 
        input_size=tf.TensorShape([embeddings_size]),
        dtype=tf.float32
    ) for lstm in cells
]

# We first define a MultiRNNCell Object that uses the 
# Dropout wrapper (for training)
stacked_dropout_cell = tf.nn.rnn_cell.MultiRNNCell(dropout_cells)
# Here we define a MultiRNNCell that does not use dropout
# Validation and Testing
stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)


# Note: There exists the EmbeddingWrapper in RNN API to automate the embedding_lookup but,
# in many cases it may be more efficient to not use this wrapper, but instead concatenate the whole sequence of 
# your inputs in time, do the embedding on this batch-concatenated sequence, then split it and feed into your RNN.

# ## Defining LSTM Computations
# Here first we define the LSTM cell computations as a consice function. Then we use this function to define training and test-time inference logic.

# In[28]:


print('LSTM calculations for unrolled inputs and outputs')
# =========================================================
# Training inference logic

# Initial state of the LSTM memory.
initial_state = stacked_dropout_cell.zero_state(batch_size, dtype=tf.float32)

# Defining the LSTM cell computations (training)
train_outputs, initial_state = tf.nn.dynamic_rnn(
    stacked_dropout_cell, tf.concat(train_inputs_embeds,axis=0), 
    time_major=True, initial_state=initial_state
)

# Reshape the final outputs to [num_unrollings*batch_size, num_nodes]
final_output = tf.reshape(train_outputs,[-1,num_nodes[-1]])

# Computing logits
logits = tf.matmul(final_output, w) + b
# Computing predictions
train_prediction = tf.nn.softmax(logits)

# Reshape logits to time-major fashion [num_unrollings, batch_size, vocabulary_size]
time_major_train_logits = tf.reshape(logits,[num_unrollings,batch_size,-1])

# We create train labels in a time major fashion [num_unrollings, batch_size, vocabulary_size]
# so that this could be used with the loss function
time_major_train_labels = tf.reshape(tf.concat(train_labels,axis=0),[num_unrollings,batch_size])

# Perplexity related operation
train_perplexity_without_exp = tf.reduce_sum(tf.concat(train_labels_ohe,0)*-tf.log(train_prediction+1e-10))/(num_unrollings*batch_size)

# =========================================================
# Validation inference logic

# Separate state for validation data
initial_valid_state = stacked_cell.zero_state(1, dtype=tf.float32)

# Validation input related LSTM computation
valid_outputs, initial_valid_state = tf.nn.dynamic_rnn(
    stacked_cell, tf.expand_dims(valid_inputs_embeds,0), 
    time_major=True, initial_state=initial_valid_state
)

# Reshape the final outputs to [1, num_nodes]
final_valid_output = tf.reshape(valid_outputs,[-1,num_nodes[-1]])

# Computing logits
valid_logits = tf.matmul(final_valid_output, w) + b
# Computing predictions
valid_prediction = tf.nn.softmax(valid_logits)

# Perplexity related operation
valid_perplexity_without_exp = tf.reduce_sum(valid_labels_ohe*-tf.log(valid_prediction+1e-10))


# ## Calculating LSTM Loss
# We calculate the training loss of the LSTM here. It's a typical cross entropy loss calculated over all the scores we obtained for training data (`loss`) and averaged and summed in a specific way.

# In[29]:


# We use the sequence-to-sequence loss function to define the loss
# We calculate the average across the batches
# But get the sum across the sequence length
loss = tf.contrib.seq2seq.sequence_loss(
    logits = tf.transpose(time_major_train_logits,[1,0,2]),
    targets = tf.transpose(time_major_train_labels),
    weights= tf.ones([batch_size, num_unrollings], dtype=tf.float32),
    average_across_timesteps=False,
    average_across_batch=True
)

loss = tf.reduce_sum(loss)


# ## Defining Learning Rate and the Optimizer with Gradient Clipping
# Here we define the learning rate and the optimizer we're going to use. We will be using the Adam optimizer as it is one of the best optimizers out there. Furthermore we use gradient clipping to prevent any gradient explosions.

# In[30]:


# Used for decaying learning rate
gstep = tf.Variable(0, trainable=False)

# Running this operation will cause the value of gstep
# to increase, while in turn reducing the learning rate
inc_gstep = tf.assign(gstep, gstep+1)

# Adam Optimizer. And gradient clipping.
tf_learning_rate = tf.train.exponential_decay(0.001,gstep,decay_steps=1, decay_rate=0.5)

print('Defining optimizer')
optimizer = tf.train.AdamOptimizer(tf_learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(
    zip(gradients, v))

inc_gstep = tf.assign(gstep, gstep+1)


# ## LSTM with Beam-Search
# 
# Here we alter the previously defined prediction related TensorFlow operations to employ beam-search. Beam search is a way of predicting several time steps ahead. Concretely instead of predicting the best prediction we have at a given time step, we get predictions for several time steps and get the sequence of highest joint probability.

# In[31]:


beam_length = 5
beam_neighbors = 5

# We redefine the sample generation with beam search
sample_beam_inputs = [tf.placeholder(tf.int32, shape=[1]) for _ in range(beam_neighbors)]
# Embedding lookups for each beam
sampel_beam_input_embeds = [tf.nn.embedding_lookup(embeddings,b) for b in sample_beam_inputs]

best_beam_index = tf.placeholder(shape=None, dtype=tf.int32)
best_neighbor_beam_indices = tf.placeholder(shape=[beam_neighbors], dtype=tf.int32)

# We have [num_layers, beam_neighbors] shape state variable set
# Maintains output of each beam
saved_sample_beam_output = [[tf.Variable(tf.zeros([1, n])) for _ in range(beam_neighbors)] for n in num_nodes]
# Maintains the state of each beam
saved_sample_beam_state = [[tf.Variable(tf.zeros([1, n])) for _ in range(beam_neighbors)] for n in num_nodes] 

# Resetting the sample beam states (should be done at the beginning of each text snippet generation)
reset_sample_beam_state = tf.group(
    *[[saved_sample_beam_output[ni][vi].assign(tf.zeros([1, n]))  for vi in range(beam_neighbors)] for ni,n in enumerate(num_nodes)],
    *[[saved_sample_beam_state[ni][vi].assign(tf.zeros([1, n])) for vi in range(beam_neighbors)] for ni,n in enumerate(num_nodes)] 
)

# We stack them to perform gather operation below
# These should be of size [beam_neighbors, 1, num_nodes]
stacked_beam_outputs = [tf.stack(saved_sample_beam_output[n]) for n in range(len(num_nodes))]
stacked_beam_states = [tf.stack(saved_sample_beam_state[n]) for n in range(len(num_nodes))]

# The beam states for each beam (there are beam_neighbor-many beams) needs to be updated at every depth of tree
# Consider an example where you have 3 classes where we get the best two neighbors (marked with star)
#     a`      b*       c  
#   / | \   / | \    / | \
#  a  b c  a* b` c  a  b  c
# Since both the candidates from level 2 comes from the parent b
# We need to update both states/outputs from saved_sample_beam_state/output to have index 1 (corresponding to parent b)

# Our update_sample_beam_state gets very complicated
# Because we have to do this for every beam neighbor 
# as well as every layer
update_sample_beam_state = tf.group(
    *[
        [saved_sample_beam_output[n][vi].assign(
            tf.gather_nd(stacked_beam_outputs[n],[best_neighbor_beam_indices[vi]])) for vi in range(beam_neighbors)
          for n in range(len(num_nodes))]
    ],
    *[
        [saved_sample_beam_state[n][vi].assign(
            tf.gather_nd(stacked_beam_states[n],[best_neighbor_beam_indices[vi]])) for vi in range(beam_neighbors)
          for n in range(len(num_nodes))]
    ]
)

# This needs to be of shap [beam_neighbors, num_layers]
sample_beam_outputs, sample_beam_states = [],[] 

# This needs to be of shape [beam_neighbors, num_layers]
# and each item is a LSTMStateTuple
# We calculate lstm_cell state and output for each beam
tmp_state_tuple = []
for vi in range(beam_neighbors):
    single_beam_state_tuple = []
    for ni in range(len(num_nodes)):
        single_beam_state_tuple.append(
            tf.nn.rnn_cell.LSTMStateTuple(saved_sample_beam_output[ni][vi], saved_sample_beam_state[ni][vi])
        )
    tmp_state_tuple.append(single_beam_state_tuple)


for vi in range(beam_neighbors):
    # We cannot use tf.nn.dynamic_rnn as we need to manipulate
    # LSTM state a lot. So even though it is lot of work
    # It is easier to do state manipulation externelly 
    # when using beam search
    final_output, tmp_state_tuple[vi] = stacked_cell.call(
        sampel_beam_input_embeds[vi], tmp_state_tuple[vi]
    )
    
    # We need to be care how we populate sample_beam_outputs
    # and sample_beam_state
    # They both need to be of size [beam_neighbors, num_layers]
    sample_beam_outputs.append([])
    sample_beam_states.append([])
    for ni in range(len(num_nodes)):
        sample_beam_outputs[-1].append(tmp_state_tuple[vi][ni][0])
        sample_beam_states[-1].append(tmp_state_tuple[vi][ni][1])

    
# This store predictions made for each beam neighbor position
sample_beam_predictions = []

# Used to update the LSTM cell for each neighbor 
# Just as normally we do during generation
beam_update_ops = tf.group(
    [[saved_sample_beam_output[ni][vi].assign(sample_beam_outputs[vi][ni]) for vi in range(beam_neighbors)]
                            for ni in range(len(num_nodes))],
    [[saved_sample_beam_state[ni][vi].assign(sample_beam_states[vi][ni]) for vi in range(beam_neighbors)]
                            for ni in range(len(num_nodes))]
)

# Get the predictions out
# For a given set of beams, outputs a list of prediction vectors of size beam_neighbors
# each beam having the predictions for full vocabulary
for vi in range(beam_neighbors):
    with tf.control_dependencies([beam_update_ops]):
        sample_beam_predictions.append(tf.nn.softmax(tf.nn.xw_plus_b(sample_beam_outputs[vi][-1], w, b)))
        


# ## LSTM + Word2vec with Beam-Search
# 
# Here we alter the previously defined prediction related TensorFlow operations to employ beam-search. Beam search is a way of predicting several time steps ahead. Concretely instead of predicting the best prediction we have at a given time step, we get predictions for several time steps and get the sequence of highest joint probability.

# ### Learning rate Decay Logic
# 
# Here we define the logic to decrease learning rate whenever the validation perplexity does not decrease

# In[32]:


# Learning rate decay related
# If valid perpelxity does not decrease
# continuously for this many epochs
# decrease the learning rate
decay_threshold = 5
# Keep counting perplexity increases
decay_count = 0

min_perplexity = 1e10

# ### Running Training, Validation and Generation
# 
# We traing the LSTM on existing training data, check the validaiton perplexity on an unseen chunk of text and generate a fresh segment of text

num_steps = 251
steps_per_document = 100
docs_per_step = 10
valid_summary = 1
train_doc_count = 100


# In[41]:


beam_nodes = []

train_perplexity_ot = []
valid_perplexity_ot = []
session = tf.InteractiveSession()

tf.global_variables_initializer().run()

print('Initialized')
average_loss = 0

# We use the first 10 documents that has 
# more than (num_steps+1)*steps_per_document bigrams for creating the validation dataset

# Identify the first 10 documents following the above condition
long_doc_ids = []
for di in range(num_files):
  if len(data_list[di])>1000:
    long_doc_ids.append(di)
  if len(long_doc_ids)==10:
    break

# Generating data
data_gens = []
valid_gens = []
for fi in range(num_files):
  # Get all the bigrams if the document id is not in the validation document ids
  if fi not in long_doc_ids:
    data_gens.append(DataGeneratorSeq(data_list[fi],batch_size,num_unrollings))
  # if the document is in the validation doc ids, only get up to the 
  # last steps_per_document bigrams and use the last steps_per_document bigrams as validation data
  else:
    data_gens.append(DataGeneratorSeq(data_list[fi][:-steps_per_document],batch_size,num_unrollings))
    # Defining the validation data generator
    valid_gens.append(DataGeneratorSeq(data_list[fi][-steps_per_document:],1,1))

valid_docs = len(valid_gens)
assert valid_docs>0

feed_dict = {}
for step in range(num_steps):
    print('Training (Step: %d)'%step)
    for di in np.random.permutation(train_doc_count)[:docs_per_step]:            

        for doc_step_id in range(steps_per_document):
            
            u_data, u_labels = data_gens[di].unroll_batches()
            for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
                feed_dict[train_inputs[ui]] = dat
                feed_dict[train_labels[ui]] = lbl
                #print(['( %s; %s ) '%(reverse_dictionary[tid],reverse_dictionary[til]) for tid,til in zip(np.argmax(dat,axis=1),np.argmax(lbl,axis=1))])
            
            feed_dict.update({tf_learning_rate:0.0005})
            _, l, step_perplexity = session.run([optimizer, loss, train_perplexity_without_exp], 
                                                       feed_dict=feed_dict)
            
            average_loss += step_perplexity
        
        print('(%d).'%di,end='')
    print('')    
    
    if (step+1) % valid_summary == 0:
      
      average_loss = average_loss / (docs_per_step*steps_per_document*valid_summary)
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step+1, average_loss))
      print('\tPerplexity at step %d: %f' %(step+1, np.exp(average_loss)))
      train_perplexity_ot.append(np.exp(average_loss))
      average_loss = 0 # reset loss
      
      valid_loss = 0 # reset loss
        
      # calculate valid perplexity
      for v_doc_id in range(valid_docs):
          # Remember we process things as bigrams
          # So need to divide by 2
          for v_step in range(steps_per_document//2):
            uvalid_data,uvalid_labels = valid_gens[v_doc_id].unroll_batches()        

            # Run validation phase related TensorFlow operations       
            v_perp = session.run(
                valid_perplexity_without_exp,
                feed_dict = {valid_inputs:uvalid_data[0],valid_labels: uvalid_labels[0]}
            )

            valid_loss += v_perp
            
          # Reset validation data generator cursor
          valid_gens[v_doc_id].reset_indices() 
      print()      
      v_perplexity = np.exp(valid_loss/(steps_per_document*valid_docs//2))
      print("Valid Perplexity: %.2f\n"%v_perplexity)
      valid_perplexity_ot.append(v_perplexity)
          
      decay_learning_rate(session, v_perplexity)
    
      # Generating new text ...
      # We will be generating one segment having 500 bigrams
      # Feel free to generate several segments by changing
      # the value of segments_to_generate
      print('Generated Text after epoch %d ... '%step)  
      segments_to_generate = 1
      chars_in_segment = 500//beam_length
    
      for _ in range(segments_to_generate):
        print('======================== New text Segment ==========================')
        # first word randomly generated
        test_word = data_list[np.random.randint(0,num_files)][np.random.randint(0,100)]
        print("",reverse_dictionary[test_word],end=' ')
        
        # Generating words within a segment with Beam Search
        for _ in range(chars_in_segment):
            test_sequence = get_beam_prediction(session)
            print(test_sequence,end=' ')    
        print(" ")
        session.run([reset_sample_beam_state])
        
        print('====================================================================')
        
      print("")

session.close()

with open('lstm_beam_search_word2vec_rnn_api.csv', 'wt') as f:
    writer = csv.writer(f,delimiter=',')
    writer.writerow(train_perplexity_ot)
    writer.writerow(valid_perplexity_ot)
