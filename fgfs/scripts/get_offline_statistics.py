import matplotlib.pyplot as plt
import sys

def make_dict(lines):
    x = dict()
    for line in lines:
        kvs = line.split(':')
        x[kvs[0].split("/")[1][:-6]] = kvs[1].strip("\n")
    return x

def graph_query(graph,query,properties):
    
    if query == "default":
        return True
    keys = "n,w,r,od,cd,cdr,work_items,dimension"
    counter = 0
    graph_keys_dict = {}
    for key in keys.split(","):
        graph_keys_dict[key] = counter
        counter +=1

    # graph="expt_"+str(graph)
    graph_param_values = properties
    queries = query.split(",")
    flag = True
    for q in queries:
        param,value = q.split("=")
        key_index = graph_keys_dict[param]
        # print "KEY INDEX", key_index
        # print properties
        if graph_param_values[key_index] != value:
            flag = False
    # print "query is ", flag
    return flag

if __name__ == '__main__':
    makespan_file = open(sys.argv[1],'r')
    query =str(sys.argv[2])
    global_makespan_lc = 0.0
    global_makespan_list = 0.0
    global_speedup = 0.0
    counter = 0 
    for data_line in makespan_file.readlines():
        data_stats,makespan=data_line.split(":")
        data_point = data_stats.strip('\n').split(',')
        makespan_list,makespan_lc = makespan.strip(" ").strip("\n").split(",")
        graph_name = data_point[0]
        properties = data_point[1:9]
        if graph_query(graph_name, query, properties):
            global_makespan_list +=float(makespan_list)
            global_makespan_lc += float(makespan_lc)
            global_speedup += float(makespan_list)/float(makespan_lc)
            counter +=1
    

    #print global_makespan_list/counter, global_makespan_lc/counter, global_speedup/counter
    print global_speedup/counter
            
	        
