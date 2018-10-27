
# coding: utf-8

# In[2]:


# preprocessing imports
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence


# In[3]:


# functions we implemented
from custom_functions import init_embeddings_map, get_embed_and_pad_func


# In[ ]:


raw_data = pd.read_csv("../data/unembedded_grouped_cleaned_data.csv")


# In[ ]:


# Train/test split for our model is unique, we need to hold out a
# set of users and movies so that our network never learns those 
test_size = 0.005

# get test_size percentage of users
unique_users = raw_data.loc[:, "reviewerID"].unique()
users_size = len(unique_users)
test_idx = np.random.choice(users_size,
                              size=int(users_size * test_size),
                              replace=False)

# get test users
test_users = unique_users[test_idx]

# everyone else is a training user
train_users = np.delete(unique_users, test_idx)

test = raw_data[raw_data["reviewerID"].isin(test_users)]
train = raw_data[raw_data["reviewerID"].isin(train_users)]

unique_test_movies = test["asin"].unique()

# drop the movies that also appear in our test set. In order to be
# a true train/test split, we are forced to discard some data entirely
train = train.where(np.logical_not(train["asin"].isin(unique_test_movies))).dropna()


# In[ ]:

#
user_seq_sizes = raw_data.loc[:, "userReviews"].apply(lambda x: x.split()).apply(len)
item_seq_sizes = raw_data.loc[:, "movieReviews"].apply(lambda x: x.split()).apply(len)


# In[ ]:


u_ptile = 40
i_ptile = 15
u_seq_len = int(np.percentile(user_seq_sizes, u_ptile))
i_seq_len = int(np.percentile(item_seq_sizes, i_ptile))


# In[ ]:



emb_size = 50
embedding_map = init_embeddings_map("glove.6B." + str(emb_size) + "d.txt")


embedding_fn = get_embed_and_pad_func(i_seq_len, u_seq_len, np.array([0.0] * emb_size), embedding_map)
    
train_embedded = train.apply(embedding_fn, axis=1)
test_embedded = test.apply(embedding_fn, axis=1)


# # DeepCoNN Recommendation Model

# In[ ]:


# modeling imports
import tensorflow as tf
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Input, Dense
from keras.layers.merge import Add, Dot, Concatenate


# In[ ]:


class DeepCoNN():
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 u_seq_len,
                 m_seq_len,
                 filters=2,
                 kernel_size=10,
                 strides=6):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.filters = filters
        self.kernel_size = kernel_size
        self.inputU, self.towerU = self.create_deepconn_tower(u_seq_len)
        self.inputM, self.towerM = self.create_deepconn_tower(m_seq_len)
        self.joined = Concatenate()([self.towerU, self.towerM])
        self.outNeuron = Dense(1)(self.joined)

    def create_deepconn_tower(self, max_seq_len):
        input_layer = Input(shape=(max_seq_len, self.embedding_size))
        tower = Conv1D(filters=self.filters,
                       kernel_size=self.kernel_size,
                       activation="tanh")(input_layer)
        tower = MaxPooling1D()(tower)
        tower = Flatten()(tower)
        tower = Dense(self.hidden_size, activation="relu")(tower)
        return input_layer, tower

    def create_deepconn_dp(self):
        dotproduct = Dot(axes=1)([self.towerU, self.towerM])
        output = Add()([self.outNeuron, dotproduct])
        self.model = Model(inputs=[self.inputU, self.inputM], outputs=[output])
        self.model.compile(optimizer='Adam', loss='mse')
        
    def train(self, train_data, batch_size, epochs=3500):
        #tensorboard = TensorBoard(log_dir="../tf_logs/{}".format(pd.Timestamp(int(time()), unit="s")))
        tensorboard = TensorBoard(log_dir="../tf_logs")
        self.create_deepconn_dp()
        print(self.model.summary())
        
        user_reviews = np.array(list(train_data.loc[:, "userReviews"]))
        movie_reviews = np.array(list(train_data.loc[:, "movieReviews"]))

        self.train_inputs = [user_reviews, movie_reviews]
        self.train_outputs = train_data.loc[:, "overall"]
        
        self.history = self.model.fit(self.train_inputs,
                                      self.train_outputs,
                                      callbacks=[tensorboard],
                                      validation_split=0.05,
                                      batch_size=batch_size,
                                      epochs=epochs)
        
        


# In[ ]:


hidden_size = 64
deepconn = DeepCoNN(emb_size, hidden_size, u_seq_len, i_seq_len)

batch_size = 32
deepconn.train(train_embedded, batch_size, epochs=20)

deepconn.model.save("cnn.h5")


# In[ ]:


user_reviews = np.array(list(test_embedded.loc[:, "userReviews"]))
movie_reviews = np.array(list(test_embedded.loc[:, "movieReviews"]))

test_inputs = [user_reviews, movie_reviews]

dat = pd.DataFrame(test_inputs)
dat.to_csv("../data/test_data.csv")

true_rating = np.array(list(test_embedded.loc[:, "overall"])).reshape((-1, 1))

predictions = deepconn.model.predict(test_inputs)

error = np.square(predictions - true_rating)

print("MSE:", np.average(error))

