import sys
import os
import json
from cnn import *
from collections import OrderedDict
import numpy 


if len(sys.argv)<2:
    network_name='edlenet'
else:
    network_name=sys.argv[1]

if len(sys.argv)<3:
    height=32
    width=height
else:
    height=int(sys.argv[2])
    width=height

if len(sys.argv)<4:
    profile_mode = "interference"
else:
    profile_mode = sys.argv[3]



def create_profile_map_fused(network,width,profile_mode):
    
    fused_variant="./" + network + "_fused_configurations.txt"
    cpu_timing_file = "../profile/"+network+str(width)+"/"+profile_mode+"/"+network+"_"+str(width)+"_" + "cpu" + ".timing"
    gpu_timing_file = "../profile/"+network+str(width)+"/"+profile_mode+"/"+network+"_"+str(width)+"_" + "gpu" + ".timing"
    # print "Opening CPU File ",cpu_timing_file
    with open(cpu_timing_file,'r') as g:
        fused_cpu=json.loads(g.read())
    # print "Opening GPU File ",gpu_timing_file
    with open(gpu_timing_file,'r') as g:
        fused_gpu=json.loads(g.read())


    execution_map = {}
    with open(fused_variant,'r') as f:
        for fused in f:
            kernel_name=fused.rstrip()
            for k in kernel_name.split(','):
                fused_variant_kernel=kernel_name+"_"+k
                # print fused_variant_kernel
                cpu_time_end     = float(fused_cpu[fused_variant_kernel]["ndrange"]["device_end"])
                cpu_time_start   = float(fused_cpu[fused_variant_kernel]["ndrange"]["device_start"])
                # gpu_time_end     = float(fused_gpu[fused_variant_kernel]["read"]["device_end"])
                # if gpu_time_end==0:
                #     gpu_time_end = float(fused_gpu[fused_variant_kernel]["ndrange"]["device_end"])
                # gpu_time_start   = float(fused_gpu[fused_variant_kernel]["write"]["device_queued"])
                # if gpu_time_end==0:
                #     gpu_time_end = float(fused_gpu[fused_variant_kernel]["ndrange"]["device_start"])

                gpu_time_end     = float(fused_gpu[fused_variant_kernel]["ndrange"]["device_end"])
                gpu_time_start   = float(fused_gpu[fused_variant_kernel]["ndrange"]["device_start"])

                execution_map[fused_variant_kernel]=(cpu_time_end,cpu_time_start,gpu_time_end,gpu_time_start)
                
                # print fused_variant_kernel, cpu_time_end, cpu_time_start, gpu_time_end, gpu_time_start
                # print "------------------------"
    return execution_map


# def compute_time(cnn,kernel,execution_map):
    
#     extime_list_cpu=[]
#     extime_list_gpu=[]
#     for i in range(kernel):
#         fused_variant=""

#         for j in range(i, kernel):
#             fused_variant += str(j) + ","
        
#         for j in range(kernel,len(cnn)):
#             fused_variant += str(j)
#             fused_key = fused_variant+"_"+str(kernel)
#             fused_variant += ","
#             cpu_time_end,cpu_time_start,gpu_time_end,gpu_time_start = execution_map[fused_key]
#             cpu_time=(cpu_time_end-cpu_time_start)*1e-6
#             gpu_time=(gpu_time_end-gpu_time_start)*1e-6
#             extime_list_cpu.append(cpu_time)
#             extime_list_gpu.append(gpu_time)
#             print fused_key, cpu_time, gpu_time

#     fused_variant = str(kernel)+","
#     for j in range(kernel+1,len(cnn)):
#         fused_variant += str(j)
#         fused_key = fused_variant+"_"+str(kernel)
#         fused_variant += ","
#         cpu_time_end,cpu_time_start,gpu_time_end,gpu_time_start = execution_map[fused_key]
#         cpu_time=(cpu_time_end-cpu_time_start)*1e-6
#         gpu_time=(gpu_time_end-gpu_time_start)*1e-6
#         extime_list_cpu.append(cpu_time)
#         extime_list_gpu.append(gpu_time)
#         print fused_key, cpu_time, gpu_time    

        
#     avg_extime_cpu = numpy.mean(extime_list_cpu)
#     avg_extime_gpu = numpy.mean(extime_list_gpu)
#     std_dev_extime_cpu = numpy.std(extime_list_cpu) 
#     std_dev_extime_gpu = numpy.std(extime_list_gpu) 

#     print kernel, avg_extime_cpu, avg_extime_gpu
#     print kernel, std_dev_extime_cpu, std_dev_extime_gpu
#     print "\n"
#     return avg_extime_cpu, avg_extime_gpu


def compute_fused_time_cpu(fused_variant,execution_map):
    fused_nodes=fused_variant.split(',')
    node_start=fused_variant+"_"+fused_nodes[0]
    node_end  =fused_variant+"_"+fused_nodes[-1]
    
    cpu_time_end,_,_,_ = execution_map[node_end]
    _,cpu_time_start,_,_ = execution_map[node_start]
    extime_cpu =cpu_time_end-cpu_time_start
    return extime_cpu*1e-6

def compute_fused_time_gpu(fused_variant,execution_map):
    fused_nodes=fused_variant.split(',')
    node_start=fused_variant+"_"+fused_nodes[0]
    node_end  =fused_variant+"_"+fused_nodes[-1]
    
    _,_,gpu_time_end,_ = execution_map[node_end]
    _,_,_,gpu_time_start = execution_map[node_start]
    extime_gpu =gpu_time_end-gpu_time_start
    return extime_gpu*1e-6


def create_profile_map(network,width,profile_mode):
    info_folder = "./CNN/json/"+network+"_"+str(width)
    #print info_folder
    json_files=[os.path.join(info_folder,f) for f in os.listdir(info_folder) if f.endswith('.json')]
    execution_map = {}
    for kernel in json_files:
        # print "Opening JSON File ",kernel
        with open(kernel,'r') as g:
            json_kernel=json.loads(g.read())
        kernel_src = kernel.split("/")[-1]
        kernel_folder = kernel_src[:-5].split("_")[0]+str(width)
        cpu_timing_file = "../profile/"+kernel_folder+"/"+profile_mode+"/"+kernel_src[:-5]+"_" + "cpu" + ".timing"
        gpu_timing_file = "../profile/"+kernel_folder+"/"+profile_mode+"/"+kernel_src[:-5]+"_" + "gpu" + ".timing"
        # print "Opening CPU File ",cpu_timing_file
        with open(cpu_timing_file,'r') as g:
            json_cpu=json.loads(g.read())
        # print "Opening GPU File ",kernel
        with open(gpu_timing_file,'r') as g:
            json_gpu=json.loads(g.read())
        kernel_name=json_kernel["name"]
        # print kernel_name
        cpu_time = float(json_cpu[kernel_name]["ndrange"]["device_end"])-float(json_cpu[kernel_name]["ndrange"]["device_start"])
        gpu_time = float(json_gpu[kernel_name]["ndrange"]["device_end"])-float(json_gpu[kernel_name]["ndrange"]["device_start"])
        h2d_time = float(json_gpu[kernel_name]["write"]["device_end"])-float(json_gpu[kernel_name]["write"]["device_start"])
        d2h_time = float(json_gpu[kernel_name]["read"]["device_end"])-float(json_gpu[kernel_name]["read"]["device_start"])

        execution_map[kernel_src]=(cpu_time*1e-6,gpu_time*1e-6,h2d_time*1e-6,d2h_time*1e-6)
    return execution_map

def get_nodes_vertices(cnn):
    return len(cnn), len(cnn)-1

def get_edges(cnn):
    edges = []
    for i in range(len(cnn)-1):
        e = str(i) + " " + str(i+1)
        edges.append(e)
    return edges

def generate_dag_configuration(network, cnn, execution_map,execution_map_fused, fused_times_map, file_name):
    f = open(file_name,'w')
    v,e= get_nodes_vertices(cnn)
    print >>f, v,e
    edges = get_edges(cnn)
    
    for k in range(len(cnn)):
        layer_type = cnn[k].cfg[k][0]
        json_name = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ "1" + "_"+ "1" +".json"
        cpu_time,gpu_time,h2d_time,d2h_time = execution_map[json_name]
        kernel_id = network+"_"+ str(k) 
        node_times = kernel_id +":" +str(cpu_time) + "," + str(gpu_time + d2h_time)
        print >>f, node_times

    for edge in edges:
        print >>f, edge

    for key in fused_times_map:
        if("," in key):
            cpu_time,gpu_time = fused_times_map[key]
            dump_string = key+":"+str(cpu_time)+","+str(gpu_time)
            print >>f , dump_string


if __name__ == "__main__":
    
    # Initialise configurations for a network 

    configuration=generate_configuration_for_network(network_name,height,width)

    cnn = populate_cnn_info(configuration)



    #Initialize execution profile maps
    execution_map_fused = create_profile_map_fused(network_name,width,profile_mode)

    fused_times_map = OrderedDict()

    print "-----Fused variant timings--------"
    fused_variant="./" + network_name + "_fused_configurations.txt"
    with open(fused_variant,'r') as f:
        for fused in f:
            fused=fused.rstrip()

   
            ex_cpu  = compute_fused_time_cpu(fused,execution_map_fused)
            ex_gpu  = compute_fused_time_gpu(fused,execution_map_fused)

            fused_times_map[fused] = (ex_cpu,ex_gpu)

            print fused , ex_cpu, ex_gpu


    print "----------------"


    execution_map = create_profile_map(network_name,width,profile_mode)


    print ("Generating dag files...\n\n")
    file_name = "./graphs_overhead/" + network_name + "_" + str(height) + "_" + profile_mode + ".graph"
    generate_dag_configuration(network_name,cnn,execution_map,execution_map_fused,fused_times_map,file_name)
    
    
    print ("Finished execution...\n\n")



    

   

