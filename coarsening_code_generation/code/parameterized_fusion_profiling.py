import sys
import os
import json
from cnn import *
from collections import OrderedDict

if len(sys.argv)<2:
    network_name='nameplatemini'
else:
    network_name=sys.argv[1]

if len(sys.argv)<3:
    height=32
    width=height
else:
    height=int(sys.argv[2])

if len(sys.argv)<4:
    width=height
else:
    width=int(sys.argv[3])

if len(sys.argv)<5:
    profile_mode = "normal"
else:
    profile_mode = sys.argv[4]



def create_profile_map(network,height,profile_mode):
    info_folder = "./CNN/json/"+network+"_"+str(height)
    #print info_folder
    json_files=[os.path.join(info_folder,f) for f in os.listdir(info_folder) if f.endswith('.json')]
    execution_map = {}
    for kernel in json_files:
        print "Opening JSON File ",kernel
        with open(kernel,'r') as g:
            json_kernel=json.loads(g.read())
        kernel_src = kernel.split("/")[-1]
        kernel_folder = kernel_src[:-5].split("_")[0]+str(height)
        cpu_timing_file = "../profile/"+kernel_folder+"/"+profile_mode+"/"+kernel_src[:-5]+"_" + "cpu" + ".timing"
        gpu_timing_file = "../profile/"+kernel_folder+"/"+profile_mode+"/"+kernel_src[:-5]+"_" + "gpu" + ".timing"
        print cpu_timing_file,gpu_timing_file
        if os.path.exists(cpu_timing_file):
            print "Opening CPU File ",cpu_timing_file
            with open(cpu_timing_file,'r') as g:
                json_cpu=json.loads(g.read())
        
        if os.path.exists(gpu_timing_file):
            print "Opening GPU File ",kernel
            with open(gpu_timing_file,'r') as g:
                json_gpu=json.loads(g.read())
        
        if os.path.exists(cpu_timing_file) & os.path.exists(gpu_timing_file) :
            kernel_name=json_kernel["name"]
            print kernel_name
            cpu_time = float(json_cpu[kernel_name]["ndrange"]["device_end"])-float(json_cpu[kernel_name]["ndrange"]["device_start"])
            gpu_time = float(json_gpu[kernel_name]["ndrange"]["device_end"])-float(json_gpu[kernel_name]["ndrange"]["device_start"])
            h2d_time = float(json_gpu[kernel_name]["write"]["device_end"])-float(json_gpu[kernel_name]["write"]["device_start"])
            d2h_time = float(json_gpu[kernel_name]["read"]["device_end"])-float(json_gpu[kernel_name]["read"]["device_start"])

            execution_map[kernel_src]=(cpu_time*1e-6,gpu_time*1e-6,h2d_time*1e-6,d2h_time*1e-6)
        print(execution_map)
    return execution_map

def modify_data_transfer_time(cnn,execution_map):
    for k in range(len(cnn)):
        layer_type = cnn[k].cfg[k][0]
        max_htd_overhead = 0
        max_dth_overhead = 0
        for cf in [1,2,4,8,16]:
            for ct in ['T','B']:
                json_name = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(cf if ct == 'T' else 1 ) + "_"+ str(cf if ct == 'B' else 1) +".json"
                if (execution_map.has_key(json_name)) and (execution_map[json_name][2]>max_htd_overhead):
                    max_htd_overhead=execution_map[json_name][2]
                if (execution_map.has_key(json_name)) and (execution_map[json_name][3]>max_dth_overhead):
                    max_dth_overhead=execution_map[json_name][3]
        for cf in [1,2,4,8,16]:
            for ct in ['T','B']:
                json_name = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(cf if ct == 'T' else 1 ) + "_"+ str(cf if ct == 'B' else 1) +".json"
                # print execution_map[json_name]
                if json_name in execution_map:
                    execution_map[json_name] = execution_map[json_name][0],execution_map[json_name][1],max_htd_overhead,max_dth_overhead
                    # print execution_map[json_name]


def uncoarsened_kernel_times(cnn,execution_map):
    for k in range(len(cnn)):
        layer_type = cnn[k].cfg[k][0]
        json_name = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ "1" + "_"+ "1" +".json"
        print json_name, execution_map[json_name]



def coarsened_kernel_times(cnn,execution_map):
    for k in range(len(cnn)):
        print "============================================"
        for cf in [1,2,4,8,16]:
            layer_type = cnn[k].cfg[k][0]
            json_name_1 = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(cf) + "_"+ "1" +".json"
            if json_name_1 in execution_map:
                print json_name_1, execution_map[json_name_1]
            json_name_2 = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ "1" + "_"+ str(cf) +".json"
            if json_name_2 in execution_map:
                print json_name_2, execution_map[json_name_2]
            print "------------------------------------"
            

def coarsened_kernel_speedup(cnn,execution_map):
    for k in range(len(cnn)):
        print "============================================"

        layer_type = cnn[k].cfg[k][0]
        json_name_1 = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ "1" + "_"+ "1" +".json"
        cpu_time1,gpu_time1,h2d_time1,d2h_time1 = execution_map[json_name_1]

        for cf in [2,4,8,16]:
            layer_type = cnn[k].cfg[k][0]
            json_name_T = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(cf) + "_"+ "1" +".json"
            if json_name_T in execution_map:
                cpu_timeT,gpu_timeT,h2d_timeT,d2h_timeT = execution_map[json_name_T]
                print json_name_T, cpu_time1/cpu_timeT, gpu_time1/gpu_timeT,h2d_time1/h2d_timeT, d2h_time1/d2h_timeT
            json_name_B = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ "1" + "_"+ str(cf) +".json"
            if json_name_B in execution_map:
                cpu_timeB,gpu_timeB,h2d_timeB,d2h_timeB = execution_map[json_name_B]
                print json_name_B, cpu_time1/cpu_timeB, gpu_time1/gpu_timeB,h2d_time1/h2d_timeB, d2h_time1/d2h_timeB
            
            # print "------------------------------------"



def compute_time_cpu(coarsened_kernel_list,execution_map):
    extime_cpu = 0
    for k in coarsened_kernel_list:
        cpu_time,_,_,_ = execution_map[k]
        extime_cpu +=cpu_time
    return extime_cpu

def compute_time_gpu(coarsened_kernel_list,execution_map):
    extime_gpu = 0
    for k in coarsened_kernel_list:
        _,gpu_time,_,_ = execution_map[k]
        extime_gpu +=gpu_time
    _,_,h2d_time,_ = execution_map[coarsened_kernel_list[0]]
    _,_,_,d2h_time = execution_map[coarsened_kernel_list[-1]]
    return extime_gpu+h2d_time+d2h_time


def get_nodes_vertices(cnn):
    return len(cnn), len(cnn)-1

def get_edges(cnn):
    edges = []
    for i in range(len(cnn)-1):
        e = str(i) + " " + str(i+1)
        edges.append(e)
    return edges

def generate_dag_configuration(network, cnn, execution_map, fused_times_map, file_name):
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

def find_best_coarsened_version(node,execution_map,device):
    k=node
    layer_type = cnn[k].cfg[k][0]
    best_cf=1
    best_ct='T'
    min_latency=execution_map[network_name +  "_" + str(k) + "_"+  layer_type + "_1_1.json"][0 if device=='cpu' else 1]
    for cf in [1,2,4,8,16]:
        for ct in ['T','B']:
            json_name = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(cf if ct == 'T' else 1 ) + "_"+ str(cf if ct == 'B' else 1) +".json"
            # print json_name
            # print execution_map[json_name][0 if device=='cpu' else 1] ,min_latency
            if execution_map.has_key(json_name) and execution_map[json_name][0 if device=='cpu' else 1]<min_latency:
                min_latency=execution_map[json_name][0 if device=='cpu' else 1]
                best_cf=cf
                best_ct=ct
    # print node,best_cf,best_ct
    return best_cf,best_ct

def get_coarsened_kernel_list(fused_kernels,execution_map,device):

    coarsened_kernel_list = []
    for k in fused_kernels:
        layer_type = cnn[k].cfg[k][0]
        best_cf, best_ct= find_best_coarsened_version(k,execution_map,device)
        
        jsonname_coarsened = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(best_cf if best_ct == 'T' else 1) + "_"+ str(best_cf if best_ct == 'B' else 1) +".json"
        
        coarsened_kernel_list.append(jsonname_coarsened)
    	
    return coarsened_kernel_list

def generate_dag_dot_graph(network_name,cnn,file_name):
	f = open(file_name,'w')

	for k in range(len(cnn)):
		layer_type = cnn[k].cfg[k][0]

		best_cf, best_ct = find_best_coarsened_version(k,execution_map,'cpu')
		cpu_json = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(best_cf if best_ct == 'T' else 1) + "_"+ str(best_cf if best_ct == 'B' else 1) +".json"
        
		best_cf, best_ct  = find_best_coarsened_version(k,execution_map,'gpu')
		gpu_json = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(best_cf if best_ct == 'T' else 1) + "_"+ str(best_cf if best_ct == 'B' else 1) +".json"
        
		node_define=str(k)+	" " + cpu_json + " " + gpu_json

		print >>f, node_define

	print >>f, "---"	

	for k in range(len(cnn)-1):
		# print cnn[k].cfg[k] [1]
		flow = str(k) + " " + str(len(cnn[k].cfg[k][1])-1) + "-" + str(k+1) + " " + "0"
		print >>f, flow

	print >>f, "---"	


if __name__ == "__main__":
    
    # Initialise configurations for a network 

    configuration=generate_configuration_for_network(network_name,height,width)

    cnn = populate_cnn_info(configuration)


    #Initialize execution profile maps
    execution_map = create_profile_map(network_name,height,profile_mode)

    # print execution_map
    modify_data_transfer_time(cnn,execution_map)
    # print execution_map
    fused_times_map = OrderedDict()

    

    # Generate all possinle fused variants

    fused_variant=[]

    for start_node in range(len(configuration)): 
        for depth in range(len(configuration)-start_node): 
            nodes=[]
            for node in range(depth+1): 
                nodes.append(node+start_node)
            fused_variant.append(nodes)
        
    # print   fused_variant
    
    
    
    for fused_kernels in fused_variant:


        # ########### CF based on minimum global work size
        # find min work item size in the fused variant
        # min_work_item_size = int(cnn[fused_kernels[0]].global_work_size[0])
        # for k in fused_kernels:
        #     if(int(cnn[k].global_work_size[0])<min_work_item_size):
        #         min_work_item_size=int(cnn[k].global_work_size[0])
        # # print min_work_item_size

        # # populate list with proper coarsened version of each kernel
        # coarsened_kernel_list = []
        # for k in fused_kernels:
        #     cf = cnn[k].global_work_size[0]/min_work_item_size
        #     # print cf
        #     if cf>16:
        #         cf=16
        #     layer_type = cnn[k].cfg[k][0]
        #     jsonname = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(cf if coarsening_type == 'T' else 1) + "_"+ str(cf if coarsening_type == 'B' else 1) +".json"
        #     # print jsonname
        #     coarsened_kernel_list.append(jsonname)

        #print coarsened_kernel_list


        coarsened_kernel_list_cpu = get_coarsened_kernel_list(fused_kernels,execution_map,'cpu')
        coarsened_kernel_list_gpu = get_coarsened_kernel_list(fused_kernels,execution_map,'gpu')

        print coarsened_kernel_list_cpu
        print ("\n\n")
        print coarsened_kernel_list_gpu

        ex_cpu  = compute_time_cpu(coarsened_kernel_list_cpu,execution_map)
        ex_gpu  = compute_time_gpu(coarsened_kernel_list_gpu,execution_map)

        fused_variant_string=""
        for k in fused_kernels:
            fused_variant_string = fused_variant_string + str(k)+","
        fused_variant_string = fused_variant_string[:-1]
        fused_times_map[fused_variant_string] = (ex_cpu,ex_gpu)

        #print "-----"

    coarsened_kernel_times(cnn,execution_map)
        
    print ("Generating dag files...\n\n")
    file_name = "./graphs/" + network_name + "_" + str(height) + "_" + profile_mode + ".graph"
    generate_dag_configuration(network_name,cnn,execution_map,fused_times_map,file_name)
    
    print ("Generating dag.graph file.\n\n")
    file_name = "./dags/" + network_name + "_" + str(height) + "_" + profile_mode + "_dag.graph"
    generate_dag_dot_graph(network_name,cnn,file_name)

    
    print ("Finished execution...\n\n")



    

   

