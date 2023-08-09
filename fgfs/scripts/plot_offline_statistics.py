import os
import subprocess
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rcParams.update({'font.size':16})

def compute_speedup(command_template,argument,x,y):
    command = command_template + "complete_makespans_lc_ml_correct_depth4_"+str(x)+".stats"
    n_command = command  + " " + argument + "=" +str(y)
    print n_command
    ps = subprocess.Popen(n_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output = ps.communicate()[0].split("\n")
    print output
    return float(output[0])

def compute_speedup_tasks_nature(command_template,argument,x):
    command = command_template + "complete_makespans_lc_ml_correct_depth4_"+str(x)+".stats"
    n_command = command  + " " + argument 
    print n_command
    ps = subprocess.Popen(n_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    output = ps.communicate()[0].split("\n")
    print output
    return float(output[0])


def plot_bar(command_template,argument,x,y,labels,filename):
    X=np.arange(3)
    delay = 0.0
    c=0
    colors =['b','r','g','c']
    for n in y:
        z = [compute_speedup(command_template,argument,xs,n) for xs in x]
        plt.bar(X + delay, z, color = colors[c], edgecolor='black', width = 0.2)
        delay +=0.2
        c +=1
    X_shifted = [x+0.3 for x in X]
    plt.xticks(X_shifted, ('A1', 'A2', 'A3'))
    plt.xlabel('Architecture')
    plt.ylabel('Average Speedup')
    plt.legend(labels,loc='upper left',bbox_to_anchor=(1, 0.5))
    # plt.legend(labels,loc='upper center', ncol=4, bbox_to_anchor=(0.5,1.05))
    plt.savefig(filename,bbox_inches='tight')
    plt.clf()

def plot_bar_task_nature(command_template,filename):
    X=np.arange(3)
    delay = 0.0
    c=0
    labels = []
    x=[2,3,4]
    colors =['b','r','g','c','y','m','gray','darkred','navy']
    for cd in [2,4,6]:
        for cdr in [0.25,0.5,0.75]:
            argument = "cd="+str(cd)+",cdr="+str(cdr)
            labels.append(argument)
            z = [compute_speedup_tasks_nature(command_template,argument,xs) for xs in x]
            plt.bar(X + delay, z, color = colors[c], edgecolor='black', width = 0.1)
            delay +=0.1
            c +=1
    X_shifted = [x+0.3 for x in X]
    plt.xticks(X_shifted, ('A1', 'A2', 'A3'))
    plt.xlabel('Architecture')
    plt.ylabel('Average Speedup')
    plt.legend(labels,loc='upper left',bbox_to_anchor=(1, 1))
    # plt.legend(labels,loc="upper center",bbox_to_anchor=(0.5,1.05))
    plt.savefig(filename,bbox_inches='tight')
    plt.clf()



if __name__ == '__main__':
    command_template = "python get_offline_statistics.py "
    n_list = []
    w_list = []
    work_items_list = []

    
    #Plot for n
    x=[2,3,4]
    y_n=[50,100,150,200]

    labels_n = ["n=50", "n=100", "n=150", "n=200"]
    plot_bar(command_template,"n",x,y_n,labels_n,"speedup_vs_n_correct.pdf")
    

    
    #Plots for w
    x=[2,3,4]
    y_w=[0.3,0.5,0.7]
    labels_w = ["w=0.3", "w=0.5", "w=0.7"]
    plot_bar(command_template,"w",x,y_w,labels_w,"speedup_vs_w_correct.pdf")
    
    #Plot for W
    x=[2,3,4]
    y_W=[1024,2048,4096,8192]
    labels_W = ["W=1024", "W=2048", "W=4096", "W=8192"]
    plot_bar(command_template,"work_items",x,y_W,labels_W,"speedup_vs_W_correct.pdf")
     
    
    #Plots for r
    x=[2,3,4]
    y_r=[0.4,0.6,0.8]
    labels_r = ["r=0.4", "r=0.6", "r=0.8"]
    plot_bar(command_template,"r",x,y_r,labels_r,"speedup_vs_r_correct.pdf")
    

    plot_bar_task_nature(command_template,"speedup_vs_tasknature_correct.pdf")
    
    x=[2,3,4]
    y_od=[2,3,4,5]
    labels_od = ["od=2", "od=3", "od=4", "od=5"]
    plot_bar(command_template,"od",x,y_od,labels_od,"speedup_vs_od_correct.pdf")
    
    '''
    X=np.arange(3)
    delay = 0.0
    c=0
    colors =['b','r','g','w']
    for n in [50,100,150,200]:
        z = [compute_speedup(command_template,"n",xs,n) for xs in x]
        plt.bar(X + delay, z, color = colors[c], edgecolor='black', width = 0.2)
        delay +=0.2
        c +=1
    X_shifted = [x+0.3 for x in X]
    plt.xticks(X_shifted, ('A1', 'A2', 'A3'))
    plt.xlabel('Architecture')
    plt.ylabel('Average Speedup')
    labels = ["n=50", "n=100", "n=150", "n=200"]
    plt.legend(labels)
    plt.savefig('speedup_vs_n.pdf',bbox_inches='tight')

    '''

    
    '''
    X, Y = np.meshgrid(x, y_n)
    z_n=np.array([compute_speedup(command_template,"n",xs,ys) for xs in x for ys in y_n])
    Z=z_n.reshape(X.shape)
    fig = plt.figure(0)
    plt.xticks([2,3,4], ('A1', 'A2', 'A3'))
    ax= fig.add_subplot(111,projection='3d')
    ax.scatter(X,Y,Z)
    # ax.plot_surface(X,Y,Z, rstride=1, cstride=1, edgecolor='black')
    # ax.bar3d(X, Y, Z, width, depth, top, shade=True)
    plt.savefig("3d_n.pdf")
'''
    
'''
    for A in range(2,5):
        command = command_template + "complete_makespans_lc_ml_depth4_"+str(A)+".stats"
        n_l = []
        for n in [50,100,150,200]:
            n_command = command + " n="+str(n)
            print n_command
            n_l.append(compute_speedup(n_command))
        n_list.append(n_l)
    
    print n_list
'''