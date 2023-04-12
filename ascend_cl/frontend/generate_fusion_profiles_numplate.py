import os
import sys
import json

from cnn import *

if len(sys.argv)<2:
    network="numplateyololite"
else:
    network=sys.argv[1]

if len(sys.argv)<3:
    coarsening_type='T'
else:
    coarsening_type=sys.argv[2]




def create_profile_map(network):
    info_folder = "json/"+network
    json_files=[os.path.join(info_folder,f) for f in os.listdir(info_folder) if f.endswith('.json')]
    execution_map = {}
    for kernel in json_files:
        print "Opening JSON File ",kernel
        with open(kernel,'r') as g:
            json_kernel=json.loads(g.read())
        kernel_src = kernel.split("/")[-1]
        cpu_timing_file = "../profiling/"+kernel_src[:-5]+"_" + "cpu" + ".timing"
        gpu_timing_file = "../profiling/"+kernel_src[:-5]+"_" + "gpu" + ".timing"
        print "Opening CPU File ",cpu_timing_file
        with open(cpu_timing_file,'r') as g:
            json_cpu=json.loads(g.read())
        print "Opening GPU File ",kernel
        with open(gpu_timing_file,'r') as g:
            json_gpu=json.loads(g.read())
        kernel_name=json_kernel["name"]
        print kernel_name
        cpu_time = float(json_cpu[kernel_name]["ndrange"]["device_end"])-float(json_cpu[kernel_name]["ndrange"]["device_start"])
        gpu_time = float(json_gpu[kernel_name]["ndrange"]["device_end"])-float(json_gpu[kernel_name]["ndrange"]["device_start"])
        h2d_time = float(json_gpu[kernel_name]["write"]["device_end"])-float(json_gpu[kernel_name]["write"]["device_start"])
        d2h_time = float(json_gpu[kernel_name]["read"]["device_end"])-float(json_gpu[kernel_name]["read"]["device_start"])

        execution_map[kernel_src]=(cpu_time,gpu_time,h2d_time,d2h_time)
    return execution_map


def uncoarsened_kernel_times(cnn,execution_map):
    for k in range(len(cnn)):
        layer_type = cnn[k].cfg[k][0]
        json_name = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ "1" + "_"+ "1" +".json"
        print json_name, execution_map[json_name]

def coarsened_kernel_times(cnn,execution_map):
    for k in range(len(cnn)):
        print "============================================"
        for cf in [1]:
            layer_type = cnn[k].cfg[k][0]
            json_name_1 = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(cf) + "_"+ "1" +".json"
            print json_name_1, execution_map[json_name_1]
            json_name_2 = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ "1" + "_"+ str(cf) +".json"
            print json_name_2, execution_map[json_name_2]
            print "------------------------------------"



if __name__ == "__main__":
    configuration = []
    execution_map = create_profile_map(network)
    if network=="synthetic":
        # synthetic CNNinference dag
        configuration.append(('C',((3,32,32),(16,3,3,3),(16),(16,32,32)))) #(ic*h*w),(oc*f*f*ic),(oc),(oc*h'*w')
        configuration.append(('P',((16,32,32),(16,16,16))))  #(ic,h,w),(ic,h',w')
        configuration.append(('C',((16,16,16),(32,3,3,16),(32),(32,16,16))))
        configuration.append(('P',((32,16,16),(32,8,8))))
        configuration.append(('C',((32,8,8),(64,3,3,32),(64),(64,8,8))))
        configuration.append(('P',((64,8,8),(64,4,4))))
        configuration.append(('L',((1024),(1024,512),(512))))
        configuration.append(('L',((512),(512,128),(128)))) #(flattened version in 1d of previous output:l, bias:b, (l)(l,b))(b))
        configuration.append(('L',((128),(128,16),(16))))
        configuration.append(('S',((16),(16))))

        network_name="synthetic"

    if network=="edlenet":
        # adelnet CNN inferencing
        configuration.append(('C',((1,32,32),(32,3,3,1),(32),(32,32,32)))) 
        configuration.append(('P',((32,32,32),(32,16,16)))) 
        configuration.append(('C',((32,16,16),(64,3,3,32),(64),(64,16,16))))
        configuration.append(('P',((64,16,16),(64,8,8))))
        configuration.append(('C',((64,8,8),(128,3,3,64),(128),(128,8,8))))
        configuration.append(('P',((128,8,8),(128,4,4))))
        configuration.append(('L',((2048),(2048,128),(128)))) 
        configuration.append(('L',((128),(128,16),(16))))
        configuration.append(('S',((16),(16))))

        network_name="edlenet"


    if network=="yololite":
        # adelnet CNN inferencing
        configuration.append(('C',((1,32,32),(16,3,3,1),(16),(16,32,32)))) #(ic*h*w),(oc*f*f*ic),(oc),(oc*h'*w')
        configuration.append(('P',((16,32,32),(16,16,16)))) #(ic,h,w),(ic,h',w')
        configuration.append(('C',((16,16,16),(32,3,3,16),(32),(32,16,16))))
        configuration.append(('P',((32,16,16),(32,8,8))))
        configuration.append(('C',((32,8,8),(64,3,3,32),(64),(64,8,8))))
        configuration.append(('P',((64,8,8),(64,4,4))))
        configuration.append(('C',((64,4,4),(128,3,3,64),(128),(128,4,4))))
        configuration.append(('P',((128,4,4),(128,2,2))))
        configuration.append(('C',((128,2,2),(128,3,3,128),(128),(128,2,2))))
        configuration.append(('P',((128,2,2),(128,1,1))))
        configuration.append(('C',((128,1,1),(256,3,3,128),(256),(256,1,1))))
        configuration.append(('C',((256,1,1),(125,3,3,256),(125),(125,1,1))))
        
        network_name="yololite"


    if network=="numplateyololite":
        # yololite for numplate
        
        configuration.append(('C',((1,256,128),(16,3,3,3),(16),(16,256,128)))) #(ic*h*w),(oc*f*f*ic),(oc),(oc*h'*w')
        configuration.append(('P',((16,256,128),(16,128,64))))  #(ic,h,w),(ic,h',w')
        configuration.append(('C',((16,128,64),(32,3,3,16),(32),(32,128,64))))
        configuration.append(('P',((32,128,64),(32,64,32))))
        configuration.append(('C',((32,64,32),(64,3,3,32),(64),(64,64,32))))
        configuration.append(('P',((64,64,32),(64,32,16))))
        configuration.append(('C',((64,32,16),(128,3,3,64),(128),(128,32,16))))
        configuration.append(('P',((128,32,16),(128,16,8))))
        configuration.append(('C',((128,16,8),(128,3,3,128),(128),(128,16,8))))
        configuration.append(('P',((128,16,8),(128,8,4))))
        configuration.append(('C',((128,8,4),(4,3,3,128),(4),(4,8,4))))
        configuration.append(('P',((4,8,4),(4,4,2))))
        configuration.append(('L',((32),(32,4),(4))))

        network_name="numplateyololite"
 

    # Initialise configurations for a network 

    cnn = []
    for start_node in range(len(configuration)): 
        
        uid = ""
        tid = ""
        # variables=[]
        cfg=[]
        uid += str(start_node)
        tid += str(start_node)
        cfg.append(configuration[start_node])
        # print cfg

        dag = 1
        depth = 1
        ipbuffsize = 0
        opbuffsize = 0

        # ipbuffsize = 2 * depth + 1
        # opbuffsize = 1 * depth
        varsize = 2 * depth
        buf_index = 0
        var_index = 0
        datatype = "float"
        cnn.append(
            CNN(
                uid,
                dag,
                start_node,
                depth,
                configuration,
                buf_index,
                var_index,
                datatype,
            )
        )

        cnn[-1].initialise_kernel_info()


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
        print fused_kernels

    
    # fused ="1,2,3,4,5"
    # fused_kernels=[]     
    # for k in fused.strip().split(","):
    #     fused_kernels.append(int(k))

        # find min work item size in the fused variant
        min_work_item_size = int(cnn[fused_kernels[0]].global_work_size[0])
        for k in fused_kernels:
            if(int(cnn[k].global_work_size[0])<min_work_item_size):
                min_work_item_size=int(cnn[k].global_work_size[0])
        # print min_work_item_size

        # populate list with proper coarsened version of each kernel
        coarsened_kernel_list = []
        for k in fused_kernels:
            cf = cnn[k].global_work_size[0]/min_work_item_size
            # print cf
            if cf>16:
                cf=16
            layer_type = cnn[k].cfg[k][0]
            jsonname = network_name +  "_" + str(k) + "_"+  layer_type + "_"+ str(cf if coarsening_type == 'T' else 1) + "_"+ str(cf if coarsening_type == 'B' else 1) +".json"
            # print jsonname
            coarsened_kernel_list.append(jsonname)

        for coarsened_k in coarsened_kernel_list:
            print coarsened_k,execution_map[coarsened_k]
        # print coarsened_kernel_list

        print "-----"



    uncoarsened_kernel_times(cnn,execution_map)
    coarsened_kernel_times(cnn,execution_map)
    
