import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':14})
data_optimal_list = [4562286.61822/1000, 3372844.05663/1000, 2862251.81292/1000]
data_ml_list = [4959461.54377/1000, 3692638.49823/1000, 3147260.14706/1000]
data_optimal_lc = [3804411.53827/1000, 2890237.70445/1000, 2498606.36151/1000]
data_ml_lc = [4110963.19075/1000, 3162752.11492/1000, 2756864.23408/1000]

for i in range(len(data_optimal_list)):
    print data_ml_list[i]/data_optimal_list[i]

for i in range(len(data_optimal_lc)):
    print data_ml_lc[i]/data_optimal_lc[i]
    

data =[data_optimal_list,data_ml_list,data_optimal_lc,data_ml_lc]
labels = ["OPT List", "ML List", "OPT Clustering", "ML Clustering"]



X = np.arange(3)
plt.bar(X + 0.00, data[0], color = 'b',edgecolor='black', width = 0.2)
plt.bar(X + 0.2, data[1], color = 'b', edgecolor='black', hatch = "/", width = 0.2)
plt.bar(X + 0.4, data[2], color = 'r', edgecolor='black', width = 0.2)
plt.bar(X + 0.6, data[3], color = 'r', edgecolor='black', hatch = "/", width = 0.2)
X_shifted = [x+0.3 for x in X]
plt.xticks(X_shifted, ('A1', 'A2', 'A3'))
plt.xlabel('Architecture')
plt.ylabel('Average Makespan (ms)')
plt.legend(labels)#,loc='upper left',bbox_to_anchor=(1, 0.5))
plt.savefig('ml_vs_optimal_bold.pdf',bbox_inches='tight')

