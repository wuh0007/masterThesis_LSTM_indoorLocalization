import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation

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
axtext = fig.add_axes([0.41,0.8,0.1,0.05])
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

true_route = np.array([0,  0,  3,  3,  3,  3,  6,  6,  6,  6,  6,  6,  9,  9,  9,  9,  9,
         9, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        16, 16, 16, 16, 13, 13, 13, 13, 10, 10,  7,  7,  7,  7,  7,  7,  4,
         4,  4,  4,  1,  1,  1,  1,  2,  2,  5,  5,  8,  8,  8,  8,  8,  8,
        11, 11, 14, 14, 14, 14, 14, 14, 17, 17, 17, 17, 17, 17, 17, 17, 17,
        17, 17, 17, 17, 17, 14, 14, 14, 14, 11, 11, 11, 11, 11,  8,  8,  8,
         5,  5,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  4,  4,
         7,  7, 10, 10, 10, 10, 13, 13, 13, 13, 13, 13, 16, 16, 15, 15, 12,
        12, 12, 12,  9,  9,  9,  9,  6,  6,  3,  3,  3,  3,  3,  3,  3,  0,
         0,  0,  0,  0,  0])

# 2&3 clear stack & 1 sliding   
#pred_route = np.array([0, 3, 6, 6, 9, 9, 12, 12, 15, 15, 15, 16, 16, 13, 10,
#                       7, 7, 4, 1, 2, 5, 8, 4, 14, 14, 17, 17, 17, 17, 17, 
#                       14, [6, 5], 11, 11, 8, [-2, 3], 2, 1, 1, 1, 4, 7, 10, 
#                       13, 13, 16, 12, 12, 9, 6, 3, 3, 0, 0])

# sliding 3    
pred_route = np.array([0, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 9, 9, 9, 9, 9, 9, 12,
                       12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 15, 15, 15,
                       15, 16, 16, 16, 16, 13, 13, 13, 13, 10, 10, 7, 7, 7, 7,
                       7, 7, 4, 4, 4, 4, 1, 1, 1, 1, 2, 2, 5, 5, 8, 8, 8, 8, 
                       4, 4, 11, 11, 14, 14, 14, 14, 14, 14, 17, 17, 17, 17, 
                       17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 14, 14, 14, 
                       [9, 6], [6, 5], 11, 11, 11, 11, 8, 8, 8, 5, 5, [-2, 3],
                       2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 7, 7, 10, 10,
                       10, 10, 13, 13, 13, 13, 13, 13, 16, 16, 15, 15, 12, 12,
                       12, 12, 9, 9, 9, 9, 6, 6, 3, 3, 3, 3, 3, 3, [4, 1], 0,
                       0, 0, 0, 0])
   
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
    time.set_text('Prediction = {} '.format(str(i)))
    tr_i, tr_j = route_dict[true_route[2+i]]
    
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
        last_i, last_j = pr_i, pr_j
        kernel_texts[0][0].set_text(pred_route[i])
        kernel_texts[0][1].set_text('null')
        seq_texts[0][0].set_text('null')
        seq_texts[0][1].set_text('null')
        seq_texts[0][2].set_text('null')
    
    print(i)
    print(last_i, last_j)
    print(true_route[i], pred_route[i])
    print('-----')
    
    c = va_color.copy()
    c[tr_i,tr_j] = 1
    im_va.set_array(c)

    r = res.copy()
    r[pr_i,pr_j] = 1
    im_res.set_array(r)

ani = matplotlib.animation.FuncAnimation(fig, animate,
                                         frames=len(pred_route),
                                         interval=1000,
                                         repeat=True,
                                         init_func=init)
#ani.save("algo.gif", writer="imagemagick")
plt.show()