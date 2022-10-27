import sys

def create_configuration_file(filename,l,b,r,e):
    f = open(filename,'w')
    layer = "layer="+str(l)+"\n"
    f.write(layer)
    f.write("num_states=10\n")
    f.write("num_actions=19\n")
    f.write("replay_size="+str(r)+'\n')
    f.write("BATCH_SIZE="+str(b)+'\n')
    f.write("GAMMA=1\n")
    f.write("EPS_START=0.9\n")
    f.write("EPS_END=0.05\n")
    f.write("EPS_DECAY="+str(e)+"\n")

if __name__ == '__main__':
    
    num_states = int(sys.argv[1])
    num_actions = int(sys.argv[2])
    num_layers = int(sys.argv[3])+1
    hidden_layers = [(2*num_states)/3+num_actions, (num_states + num_actions)/2]
    print hidden_layers
    layers = []
    for i in range(1,num_layers):
        for h in hidden_layers:
            layer = []
            for j in range(0,i):
                layer.append(('L',h))
            layer.append(('L',num_actions))
            layers.append(layer)
    batch_size=[8,16,32,64]
    replay_size=[10000,20000,30000]
    eps_decay=[100,200,300,400,500,600,700,800]

    configuration_file_number = 0
    dump_config_file = open("Configurations_per/dump_config.stats",'w')
    
    
    for l in layers:
        for b in batch_size:
            for r in replay_size:
                for e in eps_decay:
                    filename = "configuration_per_"+str(configuration_file_number)+".stats"
                    dump_string = filename + ":" + str(l) + "," + str(b) + "," + str(r) + "," + str(e) +"\n" 
                    dump_config_file.write(dump_string)   
                    configuration_file_number += 1 
                    create_configuration_file("Configurations_per/"+filename,l,b,r,e)
