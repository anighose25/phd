from engine import *
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches



use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# Number of training updates done uptil now
steps_done = 0

import argparse

def parse_arg(args=None):
    parser = argparse.ArgumentParser(
        description='Gantt Chart Plotting Module')
    
    parser.add_argument('-ng', '--mGPU',
                        help='Number of GPUs',
                        default='1')
    parser.add_argument('-nc', '--nCPU',
                        help='Number of CPUs',
                        default='1')
    parser.add_argument('-d', '--dag_history',
                        help='DAG History File Name',
                        default='dag_history_0.stats')
    parser.add_argument('-ts', '--task_history',
                        help='Task History Filename',
                        default='task_history_0.stats')
    parser.add_argument('-t', '--trace_f',
                        help='Trace Folder Name')
    
    parser.add_argument('-f', '--filename',
                        help='Gantt Chart File Name')
    
    
    return parser.parse_args(args)


args = parse_arg(sys.argv[1:])


def check_correctness(dag_num_tasks,global_dag_id_dag_stats_map, global_dag_id_task_stats_map):
    number_of_global_dags = len(global_dag_id_dag_stats_map.keys())
    for d_key in global_dag_id_task_stats_map.keys():
        d = global_dag_id_task_stats_map[d_key]
        job_id = -1
        num_tasks = 0
        for t in d:
            print t
            job_id,task_component,device_type,device_id,start,finish,terminal = t
            if len(task_component) == 1:
                num_tasks +=1
            else:
                num_tasks += len(task_component.split("-"))
        print num_tasks, dag_num_tasks[job_id]
        if num_tasks == dag_num_tasks[job_id]:
            number_of_global_dags -=1
        else:
            raise Exception('global dag_id not completed {}'.format(d_key))
    
    if number_of_global_dags == 0:
        print "Schedule is correct"
    else:
        print "Schedule is not correct"
                

        


def plot_gantt_chart(global_dag_id_dag_stats_map, device_history, global_dag_id_job_instance_map,plot_colors_map, nCPU, mGPU,filename):

    fig, ax = plt.subplots(1,figsize=(10,10))
    
    heights = {}
    start_y = 5
    for i in range(nCPU):
        heights[("cpu",i)] = start_y + i*5
    
    for i in range(mGPU):
        heights[("gpu",i)] = start_y + nCPU*5 + i*5
    max_finish_time = 0.0
    patch_list = []
    for device_type in device_history.keys():
        for device_id in device_history[device_type].keys():
            for task_info in device_history[device_type][device_id]:
                global_dag_id, job_id, task_component, start, finish = task_info
                print global_dag_id, job_id, task_component, start, finish
                x=start
                width = finish - start
                max_finish_time = max(max_finish_time,finish)
                y = heights[(device_type,device_id)]
                height = 5
                j,i = global_dag_id_job_instance_map[global_dag_id]
                patch_color = plot_colors_map[(j,i)]
                cx = x + width/2.0
                cy = y + height/2.0
                ax.annotate(task_component,(cx,cy),color='black',weight='bold',fontsize=10,ha='center',va='center')
                if patch_color not in patch_list:
                    label_value = str(j) +"," +str(i)
                    ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=patch_color,edgecolor='black',  label=label_value))
                    patch_list.append(patch_color)
                else:
                    ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=patch_color,edgecolor='black',  label=""))
    ax.set_xlim(0,max_finish_time*1.5)
    ax.set_ylim(0,5*(nCPU+mGPU+2))
    # labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = []
    for i in range(nCPU+mGPU+1):
        labels.append("")
    counter = 0
    y_ticks=[0]
    y_tick_start = 7.5
    for i in range(nCPU):
        labels[counter+1] = "CPU " + str(counter)
        counter += 1
        y_ticks.append(y_tick_start)
        y_tick_start += 5
    for i in range(mGPU):
        labels[counter+1] = "GPU " + str(counter-nCPU)
        counter += 1
        y_ticks.append(y_tick_start)
        y_tick_start += 5

    print labels, y_ticks
    plt.yticks(y_ticks,labels)
    ax.set_yticklabels(labels)
    plt.legend()
    plt.savefig(filename, bbox_inches='tight')



    


if __name__ == '__main__':

    trace_folder = "Results/" + args.trace_f
    trace_file = "Results/" + args.trace_f + "/" + args.task_history
    
    nCPU = int(args.nCPU)
    mGPU = int(args.mGPU)

    global_map = {}
    global_file_list = open("global_map.txt", "r").readlines()
    global_map = make_dict(global_file_list)
    kernel_info_list = [obtain_kernel_info(key) for key in global_map]
    ex_cpu = {}
    ex_gpu = {}
    ex_map = (ex_cpu, ex_gpu)
    DC = DAGCreator()
    
    input_file1="ADAS_Graphs/dag_HOG_odroid.graph"
    input_file2="ADAS_Graphs/dag_matrix_vector_op_512_odroid.graph"
    
    dag0 = DC.create_adas_job_from_file_with_fused_times(input_file1, 0, global_map, ex_map, 0,0.0)
    dag1 = DC.create_adas_job_from_file_with_fused_times(input_file2, 1, global_map, ex_map, 1,0.0)
    dag2 = DC.create_adas_job_from_file_with_fused_times(input_file1, 2, global_map, ex_map, 2,0.0)
    dag3 = DC.create_adas_job_from_file_with_fused_times(input_file2, 3, global_map, ex_map, 3,0.0)

    dag_num_tasks = {}
    
    dag_num_tasks[0] = len(dag0.tasks.keys())
    dag_num_tasks[1] = len(dag1.tasks.keys())
    dag_num_tasks[2] = len(dag2.tasks.keys())
    dag_num_tasks[3] = len(dag3.tasks.keys())



    global_dag_id_dag_stats_map = {}
    global_dag_id_job_instance_map = {}
    job_count = {0:0, 1:0, 2:0, 3:0}
    dag_filename = args.dag_history
    dag_file_contents = open(dag_filename,'r').readlines()
    for d in dag_file_contents:
        global_dag_id, job_id, release, deadline = map(float,d.strip("\n").split(" "))
        global_dag_id = int(global_dag_id)
        job_id = int(job_id)
        global_dag_id_dag_stats_map[global_dag_id] = (job_id, release, deadline)
        if global_dag_id not in global_dag_id_job_instance_map.keys():
            global_dag_id_job_instance_map[global_dag_id] = (job_id, job_count[job_id])
            job_count[job_id] += 1
    
    
    global_dag_id_task_stats_map = {}

    device_history = {'cpu': {}, 'gpu': {}}
    
    for i in range(nCPU):
        device_history['cpu'][i]=[]
    
    for i in range(mGPU):
        device_history['gpu'][i]=[]
        

    for key in global_dag_id_dag_stats_map.keys():
        global_dag_id_task_stats_map[key] = []
    
    
    task_filename = trace_file
    task_file_contents = open(task_filename, 'r').readlines()
    for t in task_file_contents:
        global_dag_id, job_id, task_component, device_type, device_id, start, finish, terminal = t.strip("\n").split(",") # File format for task set history
        start = float(start)
        finish = float(finish)
        global_dag_id = int(global_dag_id)
        job_id = int(job_id)
        device_id = int(device_id)
        terminal = int(terminal)
        task_tuple = (job_id,task_component,device_type,device_id,start,finish,terminal)
        global_dag_id_task_stats_map[global_dag_id].append(task_tuple)
        device_stats_tuple = (global_dag_id, job_id, task_component, start, finish)
        device_history[device_type][device_id].append(device_stats_tuple)


    check_correctness(dag_num_tasks,global_dag_id_dag_stats_map,global_dag_id_task_stats_map)

    plot_colors_map = {}

    plot_colors_map[(0,0)] = "#67D33D"
    plot_colors_map[(0,1)] = "#84DB5F"
    plot_colors_map[(0,2)] = "#B2E897"
    plot_colors_map[(0,3)] = "#DAF4CB"

    plot_colors_map[(1,0)] = "#FC5E49"
    plot_colors_map[(1,1)] = "#FF725B"
    plot_colors_map[(1,2)] = "#FF9681"
    plot_colors_map[(1,3)] = "#FFC0B1"
    
    plot_colors_map[(2,0)] = "#4B48F9"
    plot_colors_map[(2,1)] = "#6356FA"
    plot_colors_map[(2,2)] = "#836EFC"
    plot_colors_map[(2,3)] = "#A58DFE"
    
    plot_colors_map[(3,0)] = "#F5FF49"
    plot_colors_map[(3,1)] = "#FAFF76"
    plot_colors_map[(3,2)] = "#FFFFAF"
    plot_colors_map[(3,3)] = "#FFFFDC"


    filename = args.filename

    plot_gantt_chart(global_dag_id_dag_stats_map,device_history,global_dag_id_job_instance_map,plot_colors_map, nCPU,mGPU,filename)

