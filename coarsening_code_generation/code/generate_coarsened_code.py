import sys
from cnn import *

if len(sys.argv)<2:
    network_name="edlenet"
else:
    network_name=sys.argv[1]

if len(sys.argv)<3:
    target="AScEnD"
else:
    target=sys.argv[2]
    
if len(sys.argv)<4:
    height=32
else:
    height=int(sys.argv[3])

if len(sys.argv)<5:
    width=height
else:
    width=int(sys.argv[4])

if __name__ == "__main__":
    

    configuration=generate_configuration_for_network(network_name,height,width)

    
    total_nodes=len(configuration)
    for start_node in range(total_nodes): 
        cnn = []
        # for i in range(1, total_nodes - start_node + 1):
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

        layer_type,layer_info = cnn[-1].cfg[cnn[-1].start_node]
        
        coarsening_type=['T','B']
        coarsening_factors=[1,2,4,8,16]
        #coarsening_type=['T']
        #coarsening_factors=[1]
        for ct in coarsening_type: 
            for cf in coarsening_factors:    

                cnn[-1].initialise_kernel_info()

                cnn[-1].generate_coarsened_source_code(ct,cf,8 if ct == 'T' else 1 )

                src_file = "./CNN/src/" + network_name+ "_" +str(height) + "/" +network_name +  "_" + str(start_node) + "_"+  layer_type + "_"+ str(cf if ct == 'T' else 1) + "_"+ str(cf if ct == 'B' else 1) +".cl"   
                print src_file
                file = open(src_file, "w")
                file.writelines(cnn[-1].source_code)
                file.close()
                cnn[-1].source_code=""

                target='AScEnD'
                if target=='AScEnD':
                    
                    jsonname = "./CNN/json/" + network_name+ "_" +str(height) + "/"  +network_name +  "_" + str(start_node) + "_"+  layer_type + "_"+ str(cf if ct == 'T' else 1) + "_"+ str(cf if ct == 'B' else 1) +".json"
                    print jsonname
                    dimension = 1
                    src_file_name = network_name+"_" + str(start_node) + "_"+  layer_type + "_"+ str(cf if ct == 'T' else 1) + "_"+ str(cf if ct == 'B' else 1) +".cl"
                    if layer_type=='C':
                        layer_name="convolution"
                    elif layer_type=='P':
                        layer_name="pooling"
                    elif layer_type=='L':
                        layer_name="linear"
                    elif layer_type=='S':
                        layer_name="softmax"
                    kernel_name = layer_name + "_"+ uid
                    cnn[-1].dump_json(kernel_name, src_file_name, dimension, jsonname,cf)

                target='EScheduler'
                if target=='EScheduler':

                    tinfoname = "./CNN/tinfo/"  + network_name + "_" +str(height) +"/" +network_name +  "_"  + str(start_node) + "_"+  layer_type + "_"+ str(cf if ct == 'T' else 1) + "_"+ str(cf if ct == 'B' else 1)
                    # tinfoname = "./CNN/tinfo/node_" + str(start_node) + ":" + tid
                    print tinfoname
                    dimension = 1

                    src_file_name =  network_name+ "_" + str(start_node) + "_"+  layer_type + "_"+ str(cf if ct == 'T' else 1) + "_"+ str(cf if ct == 'B' else 1) +".cl"
                    if layer_type=='C':
                        layer_name="convolution"
                    elif layer_type=='P':
                        layer_name="pooling"
                    elif layer_type=='L':
                        layer_name="linear"
                    elif layer_type=='S':
                        layer_name="softmax"
                    kernel_name = layer_name + "_"+ uid
                    
                    # kernel_name = "cnn_" + uid
                    cnn[-1].dump_tinfo(kernel_name, src_file_name, dimension, tinfoname,cf)
