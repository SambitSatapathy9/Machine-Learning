#WORD EMBEDDING LAYER 
"""
1. Sentences
2. One hot Encoding
3. Padding --> Post and Pre-Padding --> OHE
4. OHE --> Vectors

Vocalbulary -  unique words
"""
import tensorflow as tf
from tensorflow.preprocess.text import one_hot

#Sentences
sent = [  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good']

#Initialize Vocabulary Size
voc_size = 10000

#One hot representation
onehot_encode = [one_hot(words, n=voc_size) for words in sent]
onehot_encode

"""
Output: [[6930, 8628, 3284, 2721],
 [6930, 8628, 3284, 5355],
 [6930, 5339, 3284, 7304],
 [7013, 5081, 365, 4966, 6752],
 [7013, 5081, 365, 4966, 2033],
 [3324, 6930, 5695, 3284, 3415],
 [3847, 8603, 1923, 4966]]
 
Basically what this means is:
the ---> 6930
Out of the 10000 size of vectors, the index number 6930 is 1 and rest are zero, and this represents the word "the". 
the = [0,0,0,0,0,......,1,0,0,0,......]
index position of 1: 6930

Similar is the case for all the other words.
"""
## Word Embedding Representation
"""
Padding - To convert all the sentences in the corpus to same length, given by the user.
How does it do that? 
Answer is either you add zeros to the right(post) or the left(pre).
If your given length is shorter than the orignal sentences then there might be chances of information loss. 
"""
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

import numpy as np

sent_length = 5

#Pre-padding
embedded_docs = pad_sequences(onehot_encod, maxlen=sent_length, padding='pre')
print(embedded_docs)

#Post Padding
post_embedded_docs = pad_sequences(onehot_encod, maxlen=sent_length, padding='post')
print(embedded_docs)

dim = 10 #specifying the output dimension

#Creating a model with embedding layer
model = Sequential()
model.add(Embedding(input_dim=voc_size,output_dim=10, input_length= sent_length))
model.compile('adam','mse')
model.summary()

#Predicting the model
print(model.predict(embedded_docs))

print("Original Sentence: ",sent[0])
print("OHE with Pre-Padding: ",embedded_docs[0])
pred = model.predict(embedded_docs[0])
print("Feature Representation (word --> Vector):\n",pred)

printt = sent[0].split(" ")
print([word for word in printt])
for i,j in enumerate(printt):
  print(f"{j} - {pred[i+1]}")














