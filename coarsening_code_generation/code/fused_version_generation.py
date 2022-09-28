import sys
from cnn import *

if len(sys.argv)<2:
    network_name='yololite'
else:
    network_name=sys.argv[1]

if len(sys.argv)<3:
    coarsening_type='T'
else:
    coarsening_type=sys.argv[2]

if len(sys.argv)<4:
    height=32
else:
    height=int(sys.argv[3])

if len(sys.argv)<5:
    width=height
else:
    width=int(sys.argv[4])

if __name__ == "__main__":
    
    
    # Initialise configurations for a network 

    configuration=generate_configuration_for_network(network_name,height,width)

    cnn = populate_cnn_info(configuration)


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

        print coarsened_kernel_list

        print "-----"



