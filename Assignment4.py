# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 05:03:45 2023

@author: AutoLab
"""

import keras
from keras.models import Sequential, Model 
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dropout, Flatten,Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import os
from matplotlib import image,patches,patheffects
import matplotlib.pyplot as plt
from PIL import Image
#%%
from pathlib import Path
PATH = Path('VOCdevkit/VOC2007/')
for i in PATH.iterdir(): print(i)
JPEGS = PATH/'JPEGImages'
#%%
import json
BD = json.load((PATH/'pascal_train2007.json').open()) # it loads a dictionary
print('the dictionary of keys: ',BD.keys())
#%%
import collections

def hw_bb(bb): return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])

# we convert categories into dictionary
data_category = dict((o['id'],o['name']) for o in BD['categories']) # all the categories
data_filename = dict((o['id'],o['file_name']) for o in BD['images']) # image id to image filename
data_ids = [o['id'] for o in BD['images']] # list of all the image IDs

annotations = collections.defaultdict(lambda:[])
for o in BD['annotations']:
    if not o['ignore']:
        bb = o['bbox']
        bb = hw_bb(bb)
        annotations[o['image_id']].append((bb,o['category_id']))
        
print('we have',len(BD['annotations']),'annotations')
#%%
def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])

def show_img(im, figsize=None, ax=None):
    """show images"""
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_outline(o, lw):
  
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_rect(ax, b):
    """Draw rectangle around the object of interest"""
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)
    
def draw_text(ax, xy, txt, sz=14):
    """Write the text on the upper right corner of rectangle surrounding the object"""
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)    
    
def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8)) # That's why the image with draw_im is zoomed in
    for b,c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], data_category[c], sz=16)
        
def draw_idx(i):
    im_a = annotations[i]
    im = image.imread(JPEGS/data_filename[i])
    draw_im(im, im_a)
#%%
im = image.imread(JPEGS/data_filename[data_ids[0]])
    
ax = show_img(im)
bbox = annotations[data_ids[0]][0][0]
clas_id = annotations[data_ids[0]][0][1]
bbox = bb_hw(bbox)

draw_rect(ax, bbox)
#%%
import pandas as pd

def get_largest_annotation(b):
    if not b: raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:]-x[0][:2]), reverse=True)
    return b[0]


filename=[]
C=[]
for image_id,annotation in annotations.items():
    filename.append(data_filename[image_id])
    
    C.append(data_category[get_largest_annotation(annotation)[1]])
    
df = pd.DataFrame({'filename': filename, 'class': C}, columns=['filename','class'])
#%%
def Split_Train_Valid(df,Split_train_val=0.7):
    # step 1: shuffle the data
    df = df.reindex(np.random.permutation(df.index))
    df=df.set_index(np.arange(len(df)))
    
    # step 2: split in training and testing
    df_train = df[:int(len(df)*Split_train_val)]
    df_valid = df[int(len(df)*Split_train_val):]
    df_train=df_train.set_index(np.arange(len(df_train)))
    df_valid=df_valid.set_index(np.arange(len(df_valid)))
    
    return df_train,df_valid

df_train, df_valid = Split_Train_Valid(df,0.7)
#%%
def train_val_test_dataset(filename,df):
    my_file = open(filename, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    print(data_into_list)
    my_file.close()
    for i in range(len(data_into_list)):
        data_into_list[i] = data_into_list[i]+".jpg"
    return df[df['filename'].isin(data_into_list)].reset_index(drop=True)
#%%
# our batch size
bs=32
# define the size of our input data
sz=224

# preprocess_input is for VGG16 in our case
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True) 

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 


train_batches = train_datagen.flow_from_dataframe(df_train, # The df
                                                  JPEGS, # Place on desk
                                                  x_col='filename', # The column to get x
                                                  y_col='class', # The column to get y
                                                  has_ext=True, 
                                                  target_size=(sz, sz), 
                                                  color_mode='rgb', 
                                                  classes=None, 
                                                  class_mode='categorical', 
                                                  batch_size=bs, 
                                                  shuffle=True)



valid_batches = valid_datagen.flow_from_dataframe(df_valid, 
                                                  JPEGS, 
                                                  x_col='filename', 
                                                  y_col='class', 
                                                  has_ext=True, 
                                                  target_size=(sz, sz), 
                                                  color_mode='rgb', 
                                                  classes=list(train_batches.class_indices), 
                                                  class_mode='categorical', 
                                                  batch_size=bs, 
                                                  shuffle=False)

NbClasses = len(train_batches.class_indices)

#%%
def unpreprocess(x, data_format,mode):
    """unpreprocesses a Numpy array encoding a batch of images.
    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.
        mode: One of "caffe", "tf" or "torch".
            - caffe: will convert the images from RGB to BGR,
                then will zero-center each color channel with
                respect to the ImageNet dataset,
                without scaling.
            - tf: will scale pixels between -1 and 1,
                sample-wise.
            - torch: will scale pixels between 0 and 1 and then
                will normalize each channel with respect to the
                ImageNet dataset.
    # Returns
        unreprocessed Numpy array.
    """
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    im = np.copy(x) 

    if mode == 'tf':
        im += 1.
        im *= 127.5
        im = np.clip(im, 0, 255)
        return im.astype(np.uint8)

    if mode == 'torch':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if im.ndim == 3:
            if std is not None:
                im[0, :, :] *= std[0]
                im[1, :, :] *= std[1]
                im[2, :, :] *= std[2]
                
            im[0, :, :] += mean[0]
            im[1, :, :] += mean[1]
            im[2, :, :] += mean[2]

        else:
            if std is not None:
                im[:, 0, :, :] *= std[0]
                im[:, 1, :, :] *= std[1]
                im[:, 2, :, :] *= std[2]
                
            im[:, 0, :, :] += mean[0]
            im[:, 1, :, :] += mean[1]
            im[:, 2, :, :] += mean[2]

    else:
        if std is not None:
            im[..., 0] *= std[0]
            im[..., 1] *= std[1]
            im[..., 2] *= std[2]        
        im[..., 0] += mean[0]
        im[..., 1] += mean[1]
        im[..., 2] += mean[2]

    if mode == 'torch':
        im *= 255.
    else:
        if data_format == 'channels_first':
            # 'RGB'->'BGR'
            if im.ndim == 3:
                im = im[::-1, ...]
            else:
                im = im[:, ::-1, ...]
        else:
            # 'RGB'->'BGR'
            im = im[..., ::-1]
         
    im = np.clip(im, 0, 255)

    return im.astype(np.uint8) 

#%%
'''
Object Detection
'''
filename=[]
bbox=[]
for image_id,annotation in annotations.items():
    filename.append(data_filename[image_id])
    bbox.append(get_largest_annotation(annotation)[0])
    
df = pd.DataFrame({'filename': filename, 'bbox': bbox}, columns=['filename','bbox'])

filename=[]
bbox=[]
for image_id,annotation in annotations.items():
    filename.append(data_filename[image_id])
    bbox.append(get_largest_annotation(annotation)[0])
    

#%%
class DataFrame_Generator(keras.utils.all_utils.Sequence):
    'Generates data from a Dataframe'
    def __init__(self, df, folder,preprocess_fct,batch_size=32, dim=(32,32), shuffle=True):
        'Initialization'
        self.preprocess_fct = preprocess_fct
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.folder = folder
    
        # Load the dataframe
        # the database is not so big, especially when resized in 224*224.
        # so we have either the option to load images online for each batch or
        # we can load all image at once 
        self.df = df
        self.n = len(df)            
        self.nb_iteration = int(np.floor(self.n  / self.batch_size))
        
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nb_iteration

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, 3))
        Y = np.zeros((self.batch_size,4))

        # Generate data
        for i, ID in enumerate(index):
            # Read the image
            img = Image.open(self.folder/self.df['filename'][ID])
            bb = self.df['bbox'][ID]
                 
            # Resize according to the required size
            width, height = img.size
            RatioX = width/self.dim[0]
            RatioY = height/self.dim[1]
                                                        
            img = np.asarray(img.resize(self.dim))
            
            # Resize the bbox accordingly
            bb = [bb[0]/RatioY,bb[1]/RatioX,bb[2]/RatioY,bb[3]/RatioX]
                                 
            # Same as done for VGG16
            X[i,] = self.preprocess_fct(np.asarray(img))
            
            Y[i] = bb

        return X, Y 
#%%
# train_gen = DataFrame_Generator(df_train,JPEGS,preprocess_input,bs,(sz,sz),True)
# valid_gen = DataFrame_Generator(df_valid,JPEGS,preprocess_input,bs,(sz,sz),False)
#%%
df = pd.DataFrame({'filename': filename, 'cat':C, 'bbox': bbox}, columns=['filename','cat','bbox'])
df_train = train_val_test_dataset('train.txt',df)
df_valid = train_val_test_dataset('val.txt',df)
df_test =train_val_test_dataset('test.txt',df)
#%%
class GeneratorSingleObject(keras.utils.all_utils.Sequence):
    """Generates data from a Dataframe"""

    def __init__(self, df, folder, preprocess_fct, batch_size=32, dim=(32, 32),
                 shuffle=True):
        'Initialization'
        self.preprocess_fct = preprocess_fct
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.folder = folder
        self.class_name = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                           'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                           'person', 'pottedplant', 'sheep',
                           'sofa', 'train', 'tvmonitor']
        self.NbClasses = len(self.class_name)
        self.class_dict = dict(
            (self.class_name[o], o) for o in range(self.NbClasses))

        self.df = df
        self.n = len(df)
        self.nb_iteration = int(np.floor(self.n / self.batch_size))

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.nb_iteration

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        """Generates data containing batch_size samples"""
        # Initialization
        # X: (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim, 3))
        Y_bb = np.zeros((self.batch_size, 4))
        Y_clas = np.zeros((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(index):
            # Read the image
            img = Image.open(self.folder / self.df['filename'][ID])
            bb = self.df['bbox'][ID]
            #bb = np.fromstring(bb, dtype=np.int, sep=' ')

            width, height = img.size
            RatioX = width / self.dim[0]
            RatioY = height / self.dim[1]

            img = np.asarray(img.resize(self.dim))
            bb = [bb[0] / RatioY, bb[1] / RatioX, bb[2] / RatioY, bb[3] / RatioX]

            X[i] = self.preprocess_fct(np.asarray(img))
            Y_bb[i] = bb
            Y_clas[i] = self.class_dict[self.df['cat'][ID]]

        Y_clas = keras.utils.all_utils.to_categorical(Y_clas, self.NbClasses)

        return X, [Y_bb,Y_clas]
#%%
train_gen = GeneratorSingleObject(df_train,JPEGS,preprocess_input,bs,(sz,sz),True)
valid_gen = GeneratorSingleObject(df_valid,JPEGS,preprocess_input,bs,(sz,sz),False)
#%%
fig, axes = plt.subplots(3, 4, figsize=(12, 12))

for i,ax in enumerate(axes.flat):
    x_batch,y_batch = next(iter(valid_gen))
    bb = y_batch[0][i]
    cat = y_batch[1][i]    
    image = x_batch[i]

    c = np.argmax(cat)
    ax = show_img(unpreprocess(image,'none','none'), ax=ax)
    draw_rect(ax, bb_hw(bb))
    draw_text(ax, [bb[1],bb[0]], train_gen.class_name[c], sz=16)

plt.tight_layout()
#%%
sz=224
bs=64

net = VGG16(include_top=False, weights='imagenet', input_shape=(sz,sz,3))
for layer in net.layers:
        layer.trainable=False
 
y = net.output
y = Flatten()(y)
y = Dropout(0.5)(y)

# branch for the regression --> BBox
output_layer_bbox = Dense(4, activation='linear', name='layer_bbox')(y)

# Branch for the classification --> Category
output_layer_class = Dense(train_gen.NbClasses, activation='softmax', name='layer_class')(y)

model = Model(inputs=net.input, outputs=[output_layer_bbox,output_layer_class])
#%%
model.compile(optimizer='adam',loss=['mean_absolute_error','categorical_crossentropy'], metrics=['accuracy'],loss_weights=[1., 5.])
model.summary()
#%%
epochs = 20

history = model.fit_generator(train_gen, steps_per_epoch=train_gen.nb_iteration,
                              epochs = epochs,
                              validation_data=valid_gen, validation_steps=valid_gen.nb_iteration)

print(history.history.keys())
#%%
# summarize history for accuracy for Class
plt.plot(history.history['layer_class_accuracy'])
plt.plot(history.history['val_layer_class_accuracy'])
plt.title('model accuracy - Class prediction')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='upper left')
plt.show()

# summarize history for accuracy for bbox regression
plt.plot(history.history['layer_bbox_accuracy'])
plt.plot(history.history['val_layer_bbox_accuracy'])
plt.title('model accuracy - BBOX regression')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='upper left')
plt.show()

# summarize history for loss for Class
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_layer_class_loss'])
plt.plot(history.history['layer_class_loss'])
plt.title('model loss - Class cross-entropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Global_Training', 'Global_Validation','validation','training'], loc='upper left')
plt.show()

# summarize history for loss for BBOX
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_layer_bbox_loss'])
plt.plot(history.history['layer_bbox_loss'])
plt.title('model loss - BBOX L1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Global_Training', 'Global_Validation','validation','training'], loc='upper left')
plt.show()
#%%
fig, axes = plt.subplots(3, 4, figsize=(12, 12))

for i,ax in enumerate(axes.flat):
    x_batch,y_batch = next(iter(iter(valid_gen)))
    image = x_batch[i]
    model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)

    bb = model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)[0][0]
    cat =model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)[1][0]   
    

    c = np.argmax(cat)
    ax = show_img(unpreprocess(image,'none','none'), ax=ax)
    draw_rect(ax, bb_hw(bb))
    draw_text(ax, [bb[1],bb[0]], train_gen.class_name[c], sz=16)

plt.tight_layout()
#%%
net = VGG16(include_top=False, weights='imagenet', input_shape=(sz,sz,3))
FREEZE_LAYERS = 14

# free the first layers
for layer in model.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model.layers[FREEZE_LAYERS:]:
    layer.trainable = True
    
y = net.output
y = Flatten()(y)
y = Dropout(0.5)(y)

# branch for the regression --> BBox
output_layer_bbox = Dense(4, activation='linear', name='layer_bbox')(y)

# Branch for the classification --> Category
output_layer_class = Dense(train_gen.NbClasses, activation='softmax', name='layer_class')(y)

model = Model(inputs=net.input, outputs=[output_layer_bbox,output_layer_class])

model.compile(optimizer='adam',loss=['mean_absolute_error','categorical_crossentropy'], 
              metrics=['accuracy'],loss_weights=[1., 20.])
 

epochs = 20

history = model.fit_generator(train_gen, steps_per_epoch=train_gen.nb_iteration,
                              epochs = epochs,
                              validation_data=valid_gen, validation_steps=valid_gen.nb_iteration)
#%%
# summarize history for accuracy for Class
plt.plot(history.history['layer_class_accuracy'])
plt.plot(history.history['val_layer_class_accuracy'])
plt.title('model accuracy - Class prediction')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='upper left')
plt.show()

# summarize history for accuracy for bbox regression
plt.plot(history.history['layer_bbox_accuracy'])
plt.plot(history.history['val_layer_bbox_accuracy'])
plt.title('model accuracy - BBOX regression')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training','validation'], loc='upper left')
plt.show()

# summarize history for loss for Class
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_layer_class_loss'])
plt.plot(history.history['layer_class_loss'])
plt.title('model loss - Class cross-entropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Global_Training', 'Global_Validation','validation','training'], loc='upper left')
plt.show()

# summarize history for loss for BBOX
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_layer_bbox_loss'])
plt.plot(history.history['layer_bbox_loss'])
plt.title('model loss - BBOX L1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Global_Training', 'Global_Validation','validation','training'], loc='upper left')
plt.show()
#%%
fig, axes = plt.subplots(3, 4, figsize=(12, 12))

for i,ax in enumerate(axes.flat):
    x_batch,y_batch = next(iter(iter(valid_gen)))
    image = x_batch[i]
    model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)

    bb = model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)[0][0]
    cat =model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)[1][0]   
    

    c = np.argmax(cat)
    ax = show_img(unpreprocess(image,'none','none'), ax=ax)
    draw_rect(ax, bb_hw(bb))
    draw_text(ax, [bb[1],bb[0]], train_gen.class_name[c], sz=16)

plt.tight_layout()
#%%
test_gen = GeneratorSingleObject(df_test,JPEGS,preprocess_input,bs,(sz,sz),False)
fig, axes = plt.subplots(3, 4, figsize=(12, 12))

for i,ax in enumerate(axes.flat):
    x_batch,y_batch = next(iter(iter(test_gen)))
    image = x_batch[i]
    model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)

    bb = model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)[0][0]
    cat =model.predict(np.expand_dims(image, axis=0), batch_size=None, verbose=0, steps=None)[1][0]   
    

    c = np.argmax(cat)
    ax = show_img(unpreprocess(image,'none','none'), ax=ax)
    draw_rect(ax, bb_hw(bb))
    draw_text(ax, [bb[1],bb[0]], train_gen.class_name[c], sz=16)

plt.tight_layout()