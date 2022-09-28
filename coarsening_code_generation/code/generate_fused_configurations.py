import sys
from cnn import *

if len(sys.argv)<2:
    network_name="edlenet"
else:
    network_name=sys.argv[1]

    

configuration=generate_configuration_for_network(network_name,32,32)
cnn = populate_cnn_info(configuration)

filename=network_name+"_fused_configurations.txt"
f = open(filename,'w')
for start_node in range(len(configuration)): 
    for depth in range(len(configuration)-start_node): 
        if depth>0:
	        nodes=""
	        for node in range(depth+1): 
	            nodes+=str(node+start_node)+ str( "," if node!=depth else "")
	        print >>f, nodes

