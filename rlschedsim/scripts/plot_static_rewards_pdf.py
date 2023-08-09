import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import sys
import numpy as np
style.use('fivethirtyeight')

fig = plt.figure(figsize=(18,6))
ax1 = fig.add_subplot(1,1,1)


def plot(fname):
    graph_data = open(fname,'r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    # ax1.clear()
    plt.plot(xs, ys)
    counter = 0
    xt=[]
    yt=[]
    x_avg = []
    y_avg = []
    temp = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            yt.append(float(y))
            xt.append(float(x))
            counter +=1    
        if counter == 500:
            mean = np.mean(np.asarray(yt)) 
            y_avg.append(mean)
            x_avg.append(xt[-1])
            counter = 0
            xt=[]
            yt=[]
        

    a = plt.plot(x_avg, y_avg, color='#ffff00')
    ax = a[0].axes
    ax.set_xticks(range(0,81001,8100))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(0,11):
        labels[i] = i
    ax.set_xticklabels(labels)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.xlabel('Training Epochs', fontsize=35)
    plt.ylabel('Average Rewards', fontsize=35)
    plt.yticks(fontsize=35)
    plt.xticks(fontsize=35)
    # plt.axis('off')
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    plt.show()

file_name = sys.argv[1]
plot(file_name)
save_as = sys.argv[2]
fig.savefig(save_as+".pdf",bbox_inches='tight',transparent=True,pad_inches=0)
