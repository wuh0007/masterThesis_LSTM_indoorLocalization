#!/usr/bin/env python
# coding: utf-8

# In[43]:
#%matplotlib qt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import matplotlib.ticker as mticker
import matplotlib.animation

# In[44]:

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.regularizers import l1,l2,l1_l2
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.externals.joblib import dump, load

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# In[45]:

df = pd.read_csv('df_test.csv')
df


# In[46]:

unit_dict = {0 : 1,
             1 : 2,
             2 : 4,
             3 : 8,
             4 : 10}

def route_initializer(df, 
                      forward_order = [0, 3, 6, 9, 12, 15, 16, 13, 10, 7, 4, 1, 2, 5, 8, 11, 14, 17],
                      backward_order = [17, 14, 11, 8, 5, 2, 1, 4, 7, 10, 13, 16, 15, 12, 9, 6, 3, 0],
                      foward_size_of_units = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      backward_size_of_units = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
   
    # split the dataframe and store each class as element in a list
    df_list = []
    for i in range(18):
        df_split = df[df['target'] == i]
        df_split.reset_index(drop = True, inplace = True)
        df_list.append(df_split)
        
    # specify the order of the each class based on route
    test_list_forward = [df_list[i] for i in forward_order]
    test_list_backward = [df_list[i] for i in backward_order]
    
    # pick random number of samples for each class based on size of units
    test_list_processed_forward = []
    test_list_processed_backward = []
    for i in range(18):
        test_list_processed_forward.append(test_list_forward[i].sample(n = unit_dict[foward_size_of_units[i]]))
        test_list_processed_backward.append(test_list_backward[i].sample(n = unit_dict[backward_size_of_units[i]]))
    
    return pd.concat(test_list_processed_forward), pd.concat(test_list_processed_backward)


# In[47]:
#####################################################################################################
df_test_forward, df_test_backward = route_initializer(df, 
            forward_order = [0, 3, 6, 9, 12, 15, 16, 13, 10, 7, 4, 1, 2, 5, 8, 11, 14, 17],
            backward_order = [17, 14, 11, 8, 5, 2, 1, 4, 7, 10, 13, 16, 15, 12, 9, 6, 3, 0],
#            foward_size_of_units = [1, 1, 2, 2, 2, 4, 1, 3, 0, 2, 1, 1, 1, 0, 3, 0, 2, 3],   
            foward_size_of_units = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],                          
            backward_size_of_units = [2, 1, 2, 1, 0, 1, 4, 2, 0, 1, 2, 0, 0, 1, 1, 0, 3, 3])
####################################################################################################
df_test = pd.concat([df_test_forward, df_test_backward])
#df_test 

# In[47]:

df_med = df_test.drop(df_test.iloc[:, :36], axis = 1) 
df_med.drop(df_med.iloc[:, 72:108], inplace = True, axis = 1) 

df_small = df_test.drop(df_test.iloc[:, :54], axis = 1) 
df_small.drop(df_small.iloc[:, 36:90], inplace = True, axis = 1) 

df_smaller = df_test.drop(df_test.iloc[:, :64], axis = 1) 
df_smaller.drop(df_smaller.iloc[:, 18:80], inplace = True, axis = 1) 


df_smooth = df_test.T
df_med = df_med.T
df_small = df_small.T
df_smaller = df_smaller.T


sequences_smooth = list()
for i in range(df_smooth.shape[1]):
    values = df_smooth.iloc[:-1,i].values
    sequences_smooth.append(values)
targets_smooth = df_smooth.iloc[-1, :].values

sequences_med = list()
for i in range(df_med.shape[1]):
    values = df_med.iloc[:-1,i].values
    sequences_med.append(values)
targets_med = df_med.iloc[-1, :].values

sequences_small = list()
for i in range(df_small.shape[1]):
    values = df_small.iloc[:-1,i].values
    sequences_small.append(values)
targets_small = df_small.iloc[-1, :].values

sequences_smaller = list()
for i in range(df_smaller.shape[1]):
    values = df_smaller.iloc[:-1,i].values
    sequences_smaller.append(values)
targets_smaller = df_smaller.iloc[-1, :].values

targets = targets_smooth


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(targets)
encoded_y = encoder.transform(targets)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)
targets = dummy_y


X_test_smooth, X_test_med, X_test_small, X_test_smaller, y_test = sequences_smooth, sequences_med, sequences_small, sequences_smaller, targets

# Feature Scaling
sc1 = load('std_scaler_smooth.bin')
X_test_smooth = sc1.transform(X_test_smooth)

sc2 = load('std_scaler_med.bin')
X_test_med = sc2.transform(X_test_med)

sc3 = load('std_scaler_small.bin')
X_test_small = sc3.transform(X_test_small)

sc4 = load('std_scaler_smaller.bin')
X_test_smaller = sc4.transform(X_test_smaller)


#X_test_smooth.shape, X_test_med.shape, X_test_small.shape, X_test_smaller.shape


X_test_smooth = np.reshape(X_test_smooth, (X_test_smooth.shape[0], X_test_smooth.shape[1], 1))
X_test_med = np.reshape(X_test_med, (X_test_med.shape[0], X_test_med.shape[1], 1))
X_test_small = np.reshape(X_test_small, (X_test_small.shape[0], X_test_small.shape[1], 1))
X_test_smaller = np.reshape(X_test_smaller, (X_test_smaller.shape[0], X_test_smaller.shape[1], 1))


n_timesteps_smooth, n_features_smooth = X_test_smooth.shape[1], X_test_smooth.shape[2]
# reshape into subsequences (samples, time steps, rows, cols, channels)
n_steps_smooth, n_length_smooth = 4, 36
X_test_smooth = X_test_smooth.reshape((X_test_smooth.shape[0], n_steps_smooth, 1, n_length_smooth, n_features_smooth))

n_timesteps_med, n_features_med = X_test_med.shape[1], X_test_med.shape[2]
# reshape into subsequences (samples, time steps, rows, cols, channels)
n_steps_med, n_length_med = 3, 24
X_test_med = X_test_med.reshape((X_test_med.shape[0], n_steps_med, 1, n_length_med, n_features_med))

n_timesteps_small, n_features_small = X_test_small.shape[1], X_test_small.shape[2]
# reshape into subsequences (samples, time steps, rows, cols, channels)
n_steps_small, n_length_small = 2, 18
X_test_small = X_test_small.reshape((X_test_small.shape[0], n_steps_small, 1, n_length_small, n_features_small))

n_timesteps_smaller, n_features_smaller = X_test_smaller.shape[1], X_test_smaller.shape[2]
# reshape into subsequences (samples, time steps, rows, cols, channels)
n_steps_smaller, n_length_smaller = 1, 18
X_test_smaller = X_test_smaller.reshape((X_test_smaller.shape[0], n_steps_smaller, 1, n_length_smaller, n_features_smaller))


#X_test_smooth.shape

#y_test.argmax(axis=1)
# In[47]:

class Model:
    def __init__(self, path_model, path_weight):
        self.model = self.loadmodel(path_model, path_weight)
        self.graph = tf.get_default_graph()    
    
    @staticmethod
    def loadmodel(path_model, path_weight):
        json_file = open(path_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(path_weight)
        return model

    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)


work_dir_model = '/home/hongyu/Documents/Spring2020/ECE_research/signal_analysis/data_18points/3_section_sliding5/MSLSTM_models/'
work_dir_weight = '/home/hongyu/Documents/Spring2020/ECE_research/signal_analysis/data_18points/3_section_sliding5/MSLSTM_weights/'

# work_dir_model = '/home/wuh007/Desktop/signal/signal_analysis/data_18points/3_section/models_mixed/'
# work_dir_weight = '/home/wuh007/Desktop/signal/signal_analysis/data_18points/3_section/weights_mixed/'

model_convlstm = Model(work_dir_model + 'MSLSTM_18ptsconvlstm_model.json', work_dir_weight + 'model-030-0.984992-0.947412-0.172179-18ptsconvlstm.h5')
model_lstm_simple = Model(work_dir_model + 'lstm_simplemodels.json', work_dir_weight + 'model-256-0.060712-0.031487-simple.h5')
model_lstm_simple2 = Model(work_dir_model + 'lstm_simplemodels.json', work_dir_weight + 'model-237-0.067243-0.042658-simple2.h5')


y_pred_convlstm = model_convlstm.predict([X_test_small, X_test_med, X_test_smaller, X_test_smooth])


#len(y_pred_convlstm), len(y_test)


#y_pred_convlstm


#y_pred_convlstm.argmax(axis=1), len(y_pred_convlstm.argmax(axis=1))


#y_pred_convlstm[y_pred_convlstm.max(axis=1) > 0.65].argmax(axis=1), len(y_pred_convlstm[y_pred_convlstm.max(axis=1) > 0.65].argmax(axis=1))


#y_test[y_pred_convlstm.max(axis=1) > 0.65].argmax(axis=1), len(y_test[y_pred_convlstm.max(axis=1) > 0.65].argmax(axis=1))


sc_lstm = load('std_scaler_nextMove.bin')


# sliding window one step for all conditions

from collections import deque 
  
# sequential voting stategy
def sequential_voting(sequence = y_pred_convlstm[y_pred_convlstm.max(axis=1) > 0.65].argmax(axis=1),
                      n_steps = 3):
    y_pred_new = list()
    i = 0
    stack_counter = list()
    for _ in range(len(sequence)):
#        print(i, len(sequence))
        if i >= len(sequence)-2:
            break
        
        stack = deque([sequence[i], sequence[i+1],sequence[i+2]]) 
#        print(stack)
        stack_counter.append(stack)
        
        if len(set(stack)) == 1 or len(set(stack)) == 2:
            for item in set(stack):
                if stack.count(item) > 1:
                    y_pred_new.append(item) 
            i += 1
            
        else:
            test = sc_lstm.transform([stack])
            test = test.reshape(1, 3, 1)
            out, out2 = int(np.round(model_lstm_simple.predict(test))), int(np.round(model_lstm_simple2.predict(test)))
#            if out == out2:
#                y_pred_new.append(out)
#            else:
#                y_pred_new.append([out, out2])
            y_pred_new.append([out, out2])    
#            print([int(np.round(model_lstm_simple.predict(test))), 
#                int(np.round(model_lstm_simple2.predict(test)))])
            i += 1
            
    
#        print(y_pred_new)
    return y_pred_new, stack_counter
    
y_pred_new, stack_counter = sequential_voting()
# y_pred_new, len(y_pred_new)


# In[48]:
#####################
# Array preparation
#####################

#input array
a = np.array([[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17]])
# kernel
kernel = np.array([[0,0]])

# input seq
input_seq = np.array([[0,0,0]])

# visualization array (2 bigger in each direction)
va = np.zeros((a.shape[0], a.shape[1]), dtype=int)
va[:,:] = a

#output array
res = np.zeros_like(a)

#colorarray
va_color = np.zeros((a.shape[0], a.shape[1])) 
va_color[:,:] = 0.0

#####################
# Create inital plot
#####################
fig = plt.figure(figsize=(11,10))

def add_axes_inches(fig, rect):
    w,h = fig.get_size_inches()
    return fig.add_axes([rect[0]/w, rect[1]/h, rect[2]/w, rect[3]/h])

# set matrix size
axwidth = 2.75
cellsize = axwidth/va.shape[1]
axheight = cellsize*va.shape[0]

# add time 
axtext = fig.add_axes([0.4,0.55,0.1,0.05])
# turn the axis labels/spines/ticks off
axtext.axis("off")
time = axtext.text(0.5,0.5, str(0), ha="left", va="top")

# add input sequence
ax_seq = add_axes_inches(fig, [4.5,
                               4.2,
                               2,  
                               1.5])
ax_seq.set_title("Input Sequence", size=10)
      
ax_va  = add_axes_inches(fig, [cellsize, 
                               cellsize, 
                               axwidth, 
                               axheight+1.75])
ax_kernel  = add_axes_inches(fig, [cellsize*2+axwidth,
                                   (2+res.shape[0])*cellsize-kernel.shape[0]*cellsize,
                                   kernel.shape[1]*cellsize,  
                                   kernel.shape[0]*cellsize])
ax_res = add_axes_inches(fig, [cellsize*3+axwidth+kernel.shape[1]*cellsize,
                               2*cellsize, 
                               res.shape[1]*cellsize,  
                               res.shape[0]*cellsize])
    
ax_va.set_title("True Route", size=10)
ax_res.set_title("Predicted Route", size=10)        
ax_kernel.set_title("Current Prediction", size=10)

im_va = ax_va.imshow(va_color, vmin=0., vmax=1.3, cmap="Blues")
for i in range(va.shape[0]):
    for j in range(va.shape[1]):
        ax_va.text(j,i, va[i,j], va="center", ha="center")
        
im_seq = ax_seq.imshow(np.zeros_like(input_seq), vmin=-1, vmax=1, cmap="Pastel1")
seq_texts = []
for i in range(input_seq.shape[0]):
    row = []
    for j in range(input_seq.shape[1]):
        row.append(ax_seq.text(j,i, "", va="center", ha="center"))
    seq_texts.append(row)  

im_kernel = ax_kernel.imshow(np.zeros_like(kernel), vmin=-1, vmax=1, cmap="Pastel1")
kernel_texts = []
for i in range(kernel.shape[0]):
    row = []
    for j in range(kernel.shape[1]):
        row.append(ax_kernel.text(j,i, "", va="center", ha="center"))
    kernel_texts.append(row)    

im_res = ax_res.imshow(res, vmin=0, vmax=1.3, cmap="Greens")
for i in range(res.shape[0]):
    for j in range(res.shape[1]):
        ax_res.text(j,i, va[i,j], va="center", ha="center")     

for ax  in [ax_seq, ax_va, ax_kernel, ax_res]:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.yaxis.set_major_locator(mticker.IndexLocator(1,0))
    ax.xaxis.set_major_locator(mticker.IndexLocator(1,0))
    ax.grid(color="k")

###############
# Animation
###############
route_dict = {0 : (0,0),
            1 : (0,1),
            2 : (0,2),
            3 : (1,0),
            4 : (1,1),
            5 : (1,2),
            6 : (2,0),
            7 : (2,1),
            8 : (2,2),
            9 : (3,0),
            10 : (3,1),
            11 : (3,2),
            12 : (4,0),
            13 : (4,1),
            14 : (4,2),
            15 : (5,0),
            16 : (5,1),
            17 : (5,2)}

true_route = np.array(y_test[y_pred_convlstm.max(axis=1) > 0.65].argmax(axis=1))
pred_route = np.array(y_pred_new) 


def run_animation():
    anim_running = True
    
    def onClick(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True
 
    def init():
        global last_i,last_j, threshold
        last_i = 0
        last_j = 0
        threshold = 1
        for row in kernel_texts:
            for text in row:
                text.set_text("")
        for row in seq_texts:
            for text in row:
                text.set_text("")  
        
    def animate(i):
        global last_i,last_j, threshold
        print(i)        
        print('last_pos {},{}'.format(last_i, last_j)) 
        
        time.set_text('Prediction = {} '.format(str(i)))
        tr_i, tr_j = route_dict[true_route[i+2]]
        
        
        if isinstance(pred_route[i], list): 
            kernel_texts[0][0].set_text(pred_route[i][0])
            kernel_texts[0][1].set_text(pred_route[i][1])
            diff_pred1 = (abs(route_dict[pred_route[i][0]][0] - last_i)+abs(route_dict[pred_route[i][0]][1] - last_j))
            diff_pred2 = (abs(route_dict[pred_route[i][1]][0] - last_i)+abs(route_dict[pred_route[i][1]][1] - last_j))
            if (diff_pred1 > threshold and diff_pred2 > threshold) or (diff_pred1 == diff_pred2):
               pr_i, pr_j = last_i, last_j 
            elif diff_pred1 <  diff_pred2:
               pr_i, pr_j = route_dict[pred_route[i][0]][0],route_dict[pred_route[i][0]][1]     
            else:
               pr_i, pr_j = route_dict[pred_route[i][1]][0],route_dict[pred_route[i][1]][1]   
               
        else:
            pr_i, pr_j = route_dict[pred_route[i]]
            kernel_texts[0][0].set_text(pred_route[i])
            kernel_texts[0][1].set_text('null')
            seq_texts[0][0].set_text(stack_counter[i][0])
            seq_texts[0][1].set_text(stack_counter[i][1])
            seq_texts[0][2].set_text(stack_counter[i][2])
        last_i, last_j = pr_i, pr_j 

        print('stack {} {} {}'.format(stack_counter[i][0],stack_counter[i][1],stack_counter[i][2]))
        print('true_route,pred_route {},{}'.format(true_route[i], pred_route[i]))
        print('pred_pos {},{}'.format(pr_i, pr_j))
        print('-----')
        
        c = va_color.copy()
        c[tr_i,tr_j] = 1
        im_va.set_array(c)
    
        r = res.copy()
        r[pr_i,pr_j] = 1
        im_res.set_array(r)

    fig.canvas.mpl_connect('button_press_event', onClick)
    anim = matplotlib.animation.FuncAnimation(fig, animate,
                                             frames=len(pred_route),
                                             interval=500,
                                             repeat=True,
                                             init_func=init)
    anim.save("algo.gif", writer="imagemagick")
run_animation()

#plt.show()

#%% 






