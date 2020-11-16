# -*- coding: utf-8 -*-

import time
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
from shemaFuncs import *
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import io
import numpy as np
import tqdm

from tensorflow.keras import Model
from tensorflow.keras.layers import  Dot, Embedding, Flatten
AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 0

w2vName = './data/w2v.pkl'
# do not run this; it takes a long time. used to get book version for each verse
# getAllVersions()

_, testVersions = loadBookVersions()


test_version, test_book, test_chapter, test_verse = testVersions
test_version = np.array(test_version)
test_book = np.array(test_book)
test_chapter = np.array(test_chapter)
test_verse = np.array(test_verse)


_, y_test = loadTest()
y_test = np.array(y_test)

unique_versions = pd.unique(test_version)

len(test_chapter)

#####################

    

#####################
num_ns = 6
window_size = 2

embedding_dim = 128
trunc_type = 'post'
pad_type = 'post'
oov_tok = "<OOV>"

BATCH_SIZE = 512
BUFFER_SIZE = 10000

# allw2v = pd.DataFrame(columns=['version', 'word', 'vector', 'book', 'chapter'])
allw2v = pd.DataFrame(columns=['version', 'word', 'vector'])

total_lens = 0
for curVersion in unique_versions:
    isCurVersion = test_version == curVersion
    curVerEng = y_test[isCurVersion].tolist()
    curVerBook = test_book[isCurVersion]
    curVerChaps = test_chapter[isCurVersion]
    curVerVerse = test_verse[isCurVersion]
    total_lens += len(curVerBook)
    
    tokenizer = Tokenizer(oov_token=oov_tok)
    tokenizer.fit_on_texts(curVerEng)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    sequences = tokenizer.texts_to_sequences(curVerEng)
    padded = pad_sequences(sequences, padding=pad_type, truncating=trunc_type)
    
    # Generates skip-gram pairs with negative sampling for a list of sequences
    # (int-encoded sentences) based on window size, number of negative samples
    # and vocabulary size.
    def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
        # Elements of each training example are appended to these lists.
        targets, contexts, labels = [], [], []
        
        # Build the sampling table for vocab_size tokens.
        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
        
        # Iterate over all sequences (sentences) in dataset.
        for sequence in tqdm.tqdm(sequences):
        
            # Generate positive skip-gram pairs for a sequence (sentence).
            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                    sequence, 
                    vocabulary_size=vocab_size,
                    sampling_table=sampling_table,
                    window_size=window_size,
                    negative_samples=0)
          
            # Iterate over each positive skip-gram pair to produce training examples 
            # with positive context word and negative samples.
            for target_word, context_word in positive_skip_grams:
                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1)
                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1, 
                    num_sampled=num_ns, 
                    unique=True, 
                    range_max=vocab_size, 
                    seed=SEED, 
                    name="negative_sampling")
          
              # Build context and label vectors (for one target word)
                negative_sampling_candidates = tf.expand_dims(
                      negative_sampling_candidates, 1)
            
                context = tf.concat([context_class, negative_sampling_candidates], 0)
                label = tf.constant([1] + [0]*num_ns, dtype="int64")
            
                # Append each element from the training example to global lists.
                targets.append(target_word)
                contexts.append(context)
                labels.append(label)
        
        return targets, contexts, labels
    
    targets, contexts, labels = generate_training_data(
        sequences=padded, 
        window_size=window_size, 
        num_ns=num_ns, 
        vocab_size=vocab_size, 
        seed=SEED)
    # print(len(targets), len(contexts), len(labels))
    
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    # print(dataset)
    
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # print(dataset)
    
    
    class Word2Vec(Model):
        def __init__(self, vocab_size, embedding_dim):
            super(Word2Vec, self).__init__()
            self.target_embedding = Embedding(vocab_size, 
                                              embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
            self.context_embedding = Embedding(vocab_size, 
                                                  embedding_dim, 
                                                  input_length=num_ns+1)
            self.dots = Dot(axes=(3,2))
            self.flatten = Flatten()
           
        def call(self, pair):
            target, context = pair
            we = self.target_embedding(target)
            ce = self.context_embedding(context)
            dots = self.dots([ce, we])
            return self.flatten(dots)

    
    def custom_loss(x_logit, y_true):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
    
    
    word2vec = Word2Vec(vocab_size, embedding_dim)
    word2vec.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    
    word2vec.fit(dataset, epochs=20)
    
    weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    vocab = word_index
    
    tempDF = pd.DataFrame()
    
    tempVersionList = []
    tempWordList = []
    tempVecList = []
    
    for index, word in enumerate(vocab):
       if  index == 0: continue # skip 0, it's padding.
       tempVersionList.append(curVersion) 
       tempWordList.append(word)
       vec = weights[index]
       tempVecList.append(vec)
       
    tempDF['version'] = tempVersionList
    tempDF['word'] = tempWordList
    tempDF['vector'] = tempVecList
    # tempDF['book'] = curVerBook
    # tempDF['chapter'] = curVerChaps
    
    allw2v = pd.concat([allw2v, tempDF], ignore_index=True)


print('total lengths: {}'.format(total_lens))

allw2v.to_pickle(w2vName)




# out_v = io.open('./embeddings/vectors.tsv', 'w', encoding='utf-8')
# out_m = io.open('./embeddings/metadata.tsv', 'w', encoding='utf-8')

    
# out_v.close()
# out_m.close()
