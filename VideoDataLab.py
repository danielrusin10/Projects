#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas


# In[20]:


open(r'C:\Users\danie\Desktop\USvideos.csv')
vids = pandas.read_csv(r'C:\Users\danie\Desktop\USvideos.csv')
vids.fillna(0, inplace=True)
print(vids.columns.values)


# In[21]:


import math
selected_vid = vids[vids["title"] == "WE WANT TO TALK ABOUT OUR MARRIAGE"].iloc[0]

# Choose only the numeric columns (we'll use these to compute euclidean distance)
distance_columns = [ 'views' ,'likes' ,'dislikes' ,'comment_count']
def euclidean_distance(row):
    """
    A simple euclidean distance function
    """
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_vid[k]) ** 2
    return math.sqrt(inner_value)

# Find the distance from each video to the vid selected
video_selec_distance = vids.apply(euclidean_distance, axis=1)


# In[22]:


vid_numeric = vids[distance_columns]  
# Normalize all of the numeric columns 
vid_normalized = (vid_numeric - vid_numeric.mean()) / vid_numeric.std()


# In[25]:


from scipy.spatial import distance

# Fill in NA values in vid_normalized
vid_normalized.fillna(0, inplace=True)

# Find the normalized vector for ."WE WANT TO TALK ABOUT OUR MARRIAGE"
# had to do iloc here since there are multiple videos with the same name
video_selec_normalized = vid_normalized[vids["title"] == "WE WANT TO TALK ABOUT OUR MARRIAGE"].iloc[0]
print(video_selec_normalized)
# Find the distance between selected video and the rest
euclidean_distances = vid_normalized.apply(lambda row: distance.euclidean(row, video_selec_normalized), axis=1)

# Create a new dataframe with distances.
distance_frame = pandas.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
distance_frame.sort_values("dist", inplace=True)
# Find the most similar video to the given vid
second_smallest = distance_frame.iloc[1]["idx"]
most_similar_to_vid = vids.loc[int(second_smallest)]["title"]


# In[26]:


import random
from numpy.random import permutation

random_indices = permutation(vids.index)

# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(vids)/3)

# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test = vids.loc[random_indices[1:test_cutoff]]

# Generate the train set with the rest of the data.
train = vids.loc[random_indices[test_cutoff:]]


# In[32]:


x_columns=['likes' ,'dislikes' ,'comment_count']


# In[33]:


y_column=["views"]


# In[34]:


from sklearn.neighbors import KNeighborsRegressor
# Create the knn model.
# Look at the five closest neighbors.
knn = KNeighborsRegressor(n_neighbors=5)
# Fit the model on the training data.


# In[35]:


knn.fit(train[x_columns], train[y_column])
# Make point predictions on the test set using the fit model.
predictions = knn.predict(test[x_columns])


# In[36]:


# Get the actual values for the test set.
actual = test[y_column]

# Compute the mean squared error of our predictions.
mse = (((predictions - actual) ** 2).sum()) / len(predictions)


# In[38]:


#output the final prediction
print(mse)


# In[ ]:




