import sys
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()
file_contents = open(sys.argv[1]).readlines()

n,e =map(int,file_contents[0].split(" "))

ex_cpu = {}
ex_gpu ={}
id = 0
for line in file_contents[1:n+1]:
    print line
    line_contents = line.split(":")
    t_cpu,t_gpu = map(float, line_contents[1].split(","))
    ex_gpu[id] = t_gpu
    ex_cpu[id] = t_cpu
    id +=1
fused_timings = {}
for line in file_contents[n+e+1:]:
    print line
    ids,t = line.split(":")
    ids = tuple(map(int,ids.split(",")))
    fused_timings[ids]=(float(t.split(",")[0]),float(t.split(",")[1]))

# print fused_timings
# print ex_gpu
speedups_c = []
speedups_g = []
for ids in fused_timings.keys():
    id_list = list(ids)
    t_fused=fused_timings[ids]
    t_gpu = 0.0
    for id in id_list:
        t_gpu+=ex_gpu[id]
    print(t_gpu,t_fused[1],t_gpu/t_fused[1])
    speedups_g.append(t_gpu/t_fused[1])

    t_cpu = 0.0
    for id in id_list:
        t_cpu+=ex_cpu[id]
    print(t_cpu,t_fused[0],t_cpu/t_fused[0])
    speedups_c.append(t_cpu/t_fused[0])

# for ids,speedup in zip(fused_timings.keys(),speedups_c):
#     print ids,speedup

# for ids,speedup in zip(fused_timings.keys(),speedups_g):
#     print ids,speedup

fused_ids = tuple([str(ids) for ids in fused_timings.keys()])
#sorted_fused_ids = sorted(fused_ids)
sorted_fused_ids = fused_ids
# print sorted_fused_ids
y_pos = np.arange(len(sorted_fused_ids))
# print y_pos

if sys.argv[2]=='cpu':
    performance = speedups_c
if sys.argv[2]=='gpu':
    performance = speedups_g


ax.barh(y_pos, performance, align='center',
        color='#1d8e99', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_fused_ids)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Speedup')
ax.set_title('Fusion Speedups')

plt.show()
