import sys
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.patches as mpatches

plt.rcParams.update({'font.size': 16})

def make_dict(lines):
    x = dict()
    for line in lines:
        kvs = line.split(':')
        x[kvs[0]] = float(kvs[1].strip("\n"))
    return x

def calculate_average_makespan(lines):
    counter = 0
    avg_makespan = 0.0
    for line in lines:
        
        kvs = line.split(':')
        m = float(kvs[1].strip("\n"))
        avg_makespan += m
        counter +=1
    # print counter
    return avg_makespan,counter #/counter


def plot_bar(speedup_dict,mixture_name,A,rates,filename):
    X=np.arange(3)
    delay = 0.0
    c=0
    colors =['b','r','g','c']
    z = []
    for i in range(3):
        z = []
        for xs in A:
            r = rates[(mixture_name,xs)]
            speedup_value,list_wins,lc_wins = speedup_dict[(mixture_name,xs,r[i])]
            z.append(speedup_value)
        print z
        plt.bar(X + delay, z, color = colors[c], edgecolor='black', width = 0.2)
        delay +=0.2
        c +=1
    X_shifted = [x+0.3 for x in X]
    plt.xticks(X_shifted, ('A1', 'A2', 'A3'))
    plt.xlabel('Architecture')
    plt.ylabel('Average Speedup')
    labels = ["I","II","III"]
    plt.legend(labels,loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(filename,bbox_inches='tight')
    plt.clf()


def plot_lines(speedup_dict,rates,filename):
    X=np.arange(3)
    A = [2,3,4]
    colors =['b','g','r','c','k']
    z = []
    c = 0
    for mixture in Mixture_names:
        i = 0
        for a in A:
            z = []
            for r in rates[(mixture,a)]:
                speedup_value,list_wins,lc_wins = speedup_dict[(mixture,a,r)]
                z.append(speedup_value)
            
            x = a-1 
            X_prime = None
            if x == 1:
                X_prime = [1,1.5,2,2.5,3,3.5]
                # X_prime = [1,1.5,2,2.5,3,3.5,4,4.5,5]
            if x ==2:
                X_prime = [5,5.5,6,6.5,7,7.5]
                # X_prime = [7,7.5,8,8.5,9,9.5,10,10.5,11]
            if x == 3:
                X_prime = [9,9.5,10,10.5,11,11.5]
                # X_prime = [13,13.5,14,14.5,15,15.5,16,16.5,17]
            # X_prime = [x,x+0.5,x+1]
            plt.plot(X_prime,z,marker='+',color=colors[c])
            i+=1
        c +=1
    X_shifted = [2.25,6.25,10.25]
    # X_shifted = [3,9,15]
    plt.xticks(X_shifted, ('A1', 'A2', 'A3'))
    plt.xlabel('Architecture')
    plt.ylabel('Average Speedup')
    m1_patch = mpatches.Patch(color='b', label='M1')
    m2_patch = mpatches.Patch(color='g', label='M2')
    m3_patch = mpatches.Patch(color='r', label='M3')
    m4_patch = mpatches.Patch(color='c', label='M4')
    m5_patch = mpatches.Patch(color='k', label='M5')

    plt.legend(handles=[m1_patch,m2_patch,m3_patch,m4_patch,m5_patch],loc='upper left',bbox_to_anchor=(1, 0.5))
    # labels = ["I","II","III"]
    # plt.legend(labels,loc='upper left',bbox_to_anchor=(1, 0.5))
    plt.savefig(filename,bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    
  

    avg_makespan = {}

    avg_makespan[(1024,2)]=2554466.55314
    avg_makespan[(1024,3)]=1990329.99926
    avg_makespan[(1024,4)]=1755498.53456
    avg_makespan[(2048,2)]=1314847.03575
    avg_makespan[(2048,3)]=1004051.83256
    avg_makespan[(2048,4)]=886076.971836  
    avg_makespan[(4096,2)]=3377686.39976
    avg_makespan[(4096,3)]=2478076.23624
    avg_makespan[(4096,4)]=2089081.37767
    avg_makespan[(8192,2)]=12307547.0019  
    avg_makespan[(8192,3)]=9119072.58587
    avg_makespan[(8192,4)]=7707189.8796

    m1_w = [0.4,0.1,0.1,0.4]
    m2_w = [0.1,0.4,0.4,0.1]
    m3_w = [0.25,0.25,0.25,0.25]
    m4_w = [0.2,0.2,0.3,0.3]
    m5_w = [0.3,0.3,0.2,0.2]
    work_items = [1024,2048,4096,8192]
    Mixtures =[m1_w,m2_w,m3_w,m4_w,m5_w]
    Mixture_names =["m1", "m2", "m3", "m4", "m5"]
    
    MixtureOutputData = {}
    rate_dict = {}
    
    for mixture_set,mixture_name in zip(Mixtures,Mixture_names):
        weights = mixture_set
        for A in [2,3,4]:
            weighted_average = 0.0
            for w,W in zip(weights,work_items):
                # print w,W,A,avg_makespan[(W,A)]
                weighted_average += w*avg_makespan[(W,A)]
            # weighted_average = weighted_average/4
            default_rate = 1e6/weighted_average
            
            for rate in [0.5*default_rate, 0.9*default_rate, 1.3*default_rate, 1.7*default_rate, 2.1*default_rate, 2.5*default_rate]: #, 2.9*default_rate, 3.3*default_rate, 3.7*default_rate]:
                if (mixture_name,A) not in rate_dict.keys():
                    rate_dict[(mixture_name,A)] = [rate]
                else:
                    rate_dict[(mixture_name,A)].append(rate)
                avg_speedup = 0.0
                lc_wins = 0
                list_wins = 0
                for num_experiments in range(24):
                    list_makespan_file_name = "OnlineMakespansMMCorrect/list_individual_makespan_online_configuration_"+mixture_name+"_"+str(round(rate,4))+"_"+str(A)+"_"+str(num_experiments)+'.stats'
                    lc_makespan_file_name="OnlineMakespansMMCorrect/lc_individual_makespan_online_configuration_"+mixture_name+"_"+str(round(rate,4))+"_"+str(A)+"_"+str(num_experiments)+'.stats'
                    list_file_contents = open(list_makespan_file_name,'r').readlines()
                    lc_file_contents = open(lc_makespan_file_name,'r').readlines()
                    makespan_list = make_dict(list_file_contents)
                    makespan_lc = make_dict(lc_file_contents)
                    for key in makespan_list.keys():
                        avg_speedup +=  makespan_list[key]/makespan_lc[key]
                        if makespan_lc[key] <= makespan_list[key]:
                            lc_wins +=1
                        else:
                            list_wins +=1
                
                avg_speedup = avg_speedup/(list_wins+lc_wins)
                MixtureOutputData[(mixture_name,A,rate)] = (avg_speedup,list_wins,lc_wins)
                
                print round(default_rate,4), mixture_name + "_" + str(A) + "_" + str(round(rate,4)) + ": " + str(avg_speedup) + "." + str(list_wins) + "," + str(lc_wins)
    # A = [2,3,4]
    # for mixture_name in Mixture_names:
    #     filename = "barplot_"+mixture_name+".pdf"
    #     plot_bar(MixtureOutputData,mixture_name,A,rate_dict,filename)
    plot_lines(MixtureOutputData,rate_dict,"mm_lineplot_correct.pdf")