#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import models,layers
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


IMAGE_SIZE=226
BATCH_SIZE=32
CHANNELS=3


# In[4]:


load_data = tf.keras.preprocessing.image_dataset_from_directory(
   r'C:\Users\admin\Desktop\pest1\pest with ants1',
    shuffle=True,
    image_size = (IMAGE_SIZE,IMAGE_SIZE),
    batch_size = BATCH_SIZE)


# In[5]:


class_names= load_data.class_names
class_names


# In[6]:


len(load_data)


# In[7]:


plt.figure(figsize=(20,20))
for image_batch,label_batch in load_data.take(1):
    for i in range(0,21):
        plt.subplot(6,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint32"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")


# In[8]:


def get_dataset_partition_tf(data,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=10000):
    data_size=len(data)
    if shuffle:
        data=data.shuffle(shuffle_size,seed=12)
    train_size=int(train_split*data_size)
    val_size=int(val_split*data_size)
    train_data=data.take(train_size)
    
    val_data=data.skip(train_size).take(val_size)
    test_data=data.skip(train_size).skip(val_size)
    return train_data,val_data,test_data
   


# In[9]:


train_data,val_data,test_data=get_dataset_partition_tf(load_data)


# In[10]:


train_data=train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_data=val_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data=test_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[11]:


resize_and_rescale =tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1.0/225)])


# In[12]:


data_augmentation =tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2)])


# In[13]:


input_shape = (BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3)

overall_model = models.Sequential([
    resize_and_rescale,data_augmentation,
    layers.Conv2D(64,(3,3),activation = "relu", input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,kernel_size = (3,3),activation= "relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,kernel_size = (3,3),activation= "relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation= "relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation= "relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(len(class_names),activation='softmax')])


# In[14]:


overall_model.build(input_shape=input_shape)


# In[15]:


overall_model.summary()


# In[16]:


overall_model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])


# In[17]:


history=overall_model.fit(train_data,epochs=5,batch_size=32,verbose=1,validation_data=val_data)


# In[38]:


for image_batch,lables_batch in test_data.take(1):
    image=image_batch[0].numpy().astype("uint8")
    label=lables_batch[0].numpy()
    
    plt.imshow(image)
    print("Actual label : ",class_names[label])
    
    batch_prediction=overall_model.predict(image_batch)
    print("predicted label : ",class_names[np.argmax(batch_prediction[0])])
    plt.axis("off")


# In[19]:


def predict (overall_model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(image[i].numpy())
    img_array = tf.expand_dims(img_array,0)
    
    prediction = overall_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100*(np.max(prediction[0])),2)
    return predicted_class,confidence


# In[37]:


plt.figure(figsize=(15,15))
for image, labels in test_data:
    for i in range(9):
        plt.imshow(image[i].numpy().astype("uint8"))
        plt.subplot(3,3,i+1)
        predicted_class= predict(overall_model,image[i].numpy())
        actual_class = class_names[labels[i]]
        plt.title(f"predicted : {actual_class}")
        plt.axis("off")
       


# In[ ]:




