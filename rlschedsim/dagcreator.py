from simtaskdag import *
class DAGCreator(object):
    def create_dag(self, n, global_map, ex_map, dag_id):
        G = nx.fast_gnp_random_graph(4, 0.5)
        DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])
        task_dict = dict()
        for i in range(0, n):
            key = random.choice(global_map.keys())
            feat_dict = create_feat_dict(key, global_map)

            # extime = ex_map[key]
            extime = random.randint(1, 10)
            task_dict[i] = SimTask(key, i, dag_id, feat_dict, extime)
        TaskGraph = SimTaskDAG(task_dict, DAG, dag_id)
        return TaskGraph

    def create_dag_from_graph(self, DAG, global_map, ex_map, dag_id):
        task_dict = dict()
        n = len(DAG.nodes())
        for i in range(0, n):
            key = random.choice(global_map.keys())
            feat_dict = create_feat_dict(key, global_map)
            # extime = ex_map[key]
            extime = random.randint(1, 10)
            task_dict[i] = SimTask(key, i, dag_id, feat_dict, extime)
        TaskGraph = SimTaskDAG(task_dict, DAG, dag_id)
        return TaskGraph

    

    def create_adas_job_from_file(self, input_file, dag_id, global_map, ex_map, job_id,release_time,period=0.0,tolerance=0.0):
        
        ex_cpu,ex_gpu=ex_map
        task_dict = dict()
        file_contents = open(input_file, "r").readlines()
        n, e = file_contents[0].strip("\n").split(" ")
        n=int(n)
        e=int(e)
        
        DAG = nx.DiGraph()
        for i in range(0, n):
            DAG.add_node(i)
        for i in range(1, n + 1):
            key = file_contents[i].strip("\n").split(":")[0]
            # print key
            
            extime_cpu,extime_gpu = map(float, file_contents[i].strip("\n").split(":")[1].split(","))
            ex_cpu[key]=extime_cpu/1000.0
            ex_gpu[key]=extime_gpu/1000.0
            feat_dict = create_feat_dict(key, global_map)
            if extime_cpu > extime_gpu:
                extime=extime_cpu
            else:
                extime=extime_gpu
            task_dict[i - 1] = SimTask(key, i - 1, dag_id, feat_dict, extime)
        DAG = nx.DiGraph()
        # print file_contents[n + 1:]
        for x in range(n+1,n+1+e):
            edge = file_contents[x]
            # print "EDGE", edge
            u, v = edge.strip("\n").split(" ")
            u = int(u)
            v = int(v.strip("\n"))
            DAG.add_edge(u, v, weight=0.0, time=0.0)
        fused_kernel_timings={}
        for tuple_times in file_contents[n+1+e:]:
            t,timing=tuple_times.split(":")
            t=tuple(map(int,t.split(",")))
            timing=float(timing.strip("\n"))
            fused_kernel_timings[t]=timing/1000.0

        name = input_file.split("/")[1]
        
        d = SimTaskDAG(task_dict, DAG, dag_id, ex_map,name=name,deadline=True)
        d.period=period
        d.job_id=job_id
        d.fused_kernel_timings=fused_kernel_timings
        d.release=release_time
        # print d.job_id
        
        print d.calculate_wcet()
        # d.deadline=d.release+max(d.wcet,d.period)
        d.deadline=d.release+d.period
        d.adas=True
        d.tolerance = tolerance
        # print d.job_id, d.release, d.wcet, d.deadline
        return d 
    
    def create_adas_job_from_file_with_fused_times(self, input_file, dag_id, global_map, ex_map, job_id, release_time,period=0.0,tolerance=0.0):
        
        ex_cpu,ex_gpu=ex_map
        task_dict = dict()
        file_contents = open(input_file, "r").readlines()
        n, e = file_contents[0].strip("\n").split(" ")
        n=int(n)
        e=int(e)
        
        DAG = nx.DiGraph()
        for i in range(0, n):
            DAG.add_node(i)
        for i in range(1, n + 1):
            key = file_contents[i].strip("\n").split(":")[0]
            # print key
            
            extime_cpu,extime_gpu = map(float, file_contents[i].strip("\n").split(":")[1].split(","))
            ex_cpu[key]=extime_cpu*1000.0
            ex_gpu[key]=extime_gpu*1000.0
            # feat_dict = create_feat_dict(key, global_map)
            feat_dict = {}
            if extime_cpu > extime_gpu:
                extime=extime_cpu*1000.0
            else:
                extime=extime_gpu*1000.0
            task_dict[i - 1] = SimTask(key, i - 1, dag_id, feat_dict, extime,job_id)
        DAG = nx.DiGraph()
        # print file_contents[n + 1:]
        for x in range(n+1,n+1+e):
            edge = file_contents[x]
            # print "EDGE", edge
            u, v = edge.strip("\n").split(" ")
            u = int(u)
            v = int(v.strip("\n"))
            DAG.add_edge(u, v, weight=0.0, time=0.0)
        fused_kernel_timings={'cpu': {}, 'gpu': {}}
        for tuple_times in file_contents[n+1+e:]:
            t,timing=tuple_times.split(":")
            t=tuple(map(int,t.split(",")))
            timing_cpu, timing_gpu=map(float,timing.strip("\n").split(","))
            fused_kernel_timings['cpu'][t]=timing_cpu*1000.0
            fused_kernel_timings['gpu'][t]=timing_gpu*1000.0

        name = input_file.split("/")[1]
        
        d = SimTaskDAG(task_dict, DAG, dag_id, ex_map,name=name,deadline=True,job_id=job_id)
        d.period=period
        d.job_id=job_id
        d.fused_kernel_timings=fused_kernel_timings
        d.release=release_time
        # print d.job_id
        d.calculate_wcet()
        # d.deadline=d.release+max(d.wcet,d.period)
        d.deadline=d.release+d.period
        d.tolerance = tolerance
        d.adas=True
        # print d.job_id, d.release, d.wcet, d.deadline
        return d 
    



    
    def create_dag_from_file(self, input_file, dag_id, global_map, ex_map, ml=False):
        ex_cpu, ex_gpu = ex_map
        task_dict = dict()
        # print input_file
        # print input_file
        filter_list = []
        
        file_contents = open(input_file, "r").readlines()
        n = int(file_contents[0].strip("\n"))
        DAG = nx.DiGraph()
        for i in range(0, n):
            DAG.add_node(i)
        # print "TOTAL_NODES  " + str(len(DAG.nodes()))
        for i in range(1, n + 1):
            key = file_contents[i].strip("\n")
            # print key
            # extime = file_contents[i].split(" ")[1]
            filter_list.append(key)
            feat_dict = create_feat_dict(key, global_map)

            #Case for optimal

            
            if ex_cpu[key] < ex_gpu[key]:
                extime = ex_cpu[key]
                feat_dict['Class']= "ZERO"
            else:
                extime = ex_gpu[key]
                feat_dict['Class']= "TEN"
            

            
            # if (value_int(feat_dict['Class']) < 5):
            #     extime = ex_cpu[key]
            # else:
            #     extime = ex_gpu[key]
            
         
            task_dict[i - 1] = SimTask(key, i - 1, dag_id, feat_dict, extime)
        DAG = nx.DiGraph()
        # print file_contents[n + 1:]
        for edge in file_contents[n + 1:]:
            # print edge
            u, v = edge.split(" ")
            u = int(u)
            v = int(v.strip("\n"))
            DAG.add_edge(u, v, weight=0.0, time=0.0)
        name = input_file.split("/")[1]
        ml_classifier = None
        if ml:
            ml_classifier = CLTrainer(ex_map,global_map, list(set(filter_list)),name)
         

        return SimTaskDAG(task_dict, DAG, dag_id, ex_map, ml_classifier,name)

    def get_task_info(self, devtype):
        key = random.choice(global_map.keys())
        feat_dict = create_feat_dict(key, global_map)
        extime = 0.0
        while partition_class_value(feat_dict['Class']) is not devtype:
            key = random.choice(global_map.keys())
            feat_dict = create_feat_dict(key, global_map)

        if (devtype is "gpu"):
            extime = ex_gpu[key]
        else:
            extime = ex_cpu[key]
        return key, feat_dict, extime

    def create_sample_dag(self, dag_id, ex_map, filename):
        ex_cpu, ex_gpu = ex_map
        task_dict = dict()
        DAG = nx.DiGraph()
        for i in range(0, 21):
            DAG.add_node(i)
        DAG.add_edge(0, 1)
        DAG.add_edge(0, 2)
        DAG.add_edge(1, 3)
        DAG.add_edge(1, 4)
        DAG.add_edge(2, 5)
        DAG.add_edge(3, 6)
        DAG.add_edge(3, 7)
        DAG.add_edge(4, 8)
        DAG.add_edge(5, 9)
        DAG.add_edge(6, 10)
        DAG.add_edge(7, 11)
        DAG.add_edge(8, 12)
        DAG.add_edge(9, 13)
        DAG.add_edge(10, 14)
        DAG.add_edge(10, 15)
        DAG.add_edge(10, 16)
        DAG.add_edge(11, 17)
        DAG.add_edge(12, 18)
        DAG.add_edge(12, 19)
        DAG.add_edge(13, 20)
        for node in [0, 1, 2, 5, 6, 7, 9, 12, 13, 14, 15, 16, 18, 19]:
            print "Node" + str(node) + " --> gpu"
            key, feat_dict, extime = self.get_task_info("gpu")
            print " " + feat_dict['Class']
            task_dict[node] = SimTask(key, node, dag_id, feat_dict, extime)

        for node in [3, 4, 8, 10, 19, 11, 17, 20]:
            print str(node) + " --> cpu"
            key, feat_dict, extime = self.get_task_info("cpu")
            print " " + feat_dict['Class']
            task_dict[node] = SimTask(key, node, dag_id, feat_dict, extime)

        f = open(filename, "w")
        f.write(str(21))
        f.write("\n")
        for i in range(0, 21):
            f.write(task_dict[i].name)
            f.write("\n")
        for e in DAG.edges():
            u, v = e
            edge = str(u) + " " + str(v)
            f.write(edge)
            f.write("\n")
        f.close()
        return SimTaskDAG(task_dict, DAG, dag_id, ex_map)



    def select_database_dimension(self, buf_rand_pool, required_dimension):
        output = []
        for key in buf_rand_pool:
            kernel_info = key.split("_")
            kernelName = kernel_info[0]
            worksize = kernel_info[1].strip("\n")
            if kernelName == 'uncoalesced' or kernelName == 'shared':
                kernelName = kernelName + "_copy"
                worksize = kernel_info[2].strip("\n")
            if kernelName == 'transpose':
                kernelName = kernelName + "_naive"
                worksize = kernel_info[2].strip("\n")
            dataset = int(worksize)
            dimension = obtain_kernel_dimension(key)
            if required_dimension == dimension:
                output.append(key)
        return output

    def select_database_devicetype(self, buf_rand_pool, required_device_type, global_map):
        output = []
        device_type = ""
        for key in buf_rand_pool:
            kernel_info = key.split("_")
            kernelName = kernel_info[0]
            worksize = kernel_info[1].strip("\n")
            if kernelName == 'uncoalesced' or kernelName == 'shared':
                kernelName = kernelName + "_copy"
                worksize = kernel_info[2].strip("\n")
            if kernelName == 'transpose':
                kernelName = kernelName + "_naive"
                worksize = kernel_info[2].strip("\n")
            dataset = int(worksize)
            dimension = obtain_kernel_dimension(key)
            feat_dict = create_feat_dict(key, global_map)
            if value_int(feat_dict['Class']) > 5:
                device_type = "gpu"
            else:
                device_type = "cpu"

            if device_type == required_device_type:
                output.append(key)

        return output


    def randomly_select_SimTask(self, node_index, dag_id, ex_map, rand_task_pool, global_map, workitem_range, dimension):
        workitem_list = [workitem_range[0]]
        workitem = workitem_range[0]
        while(workitem < workitem_range[1]):
            workitem *=2
            workitem_list.append(workitem)

        workitem = random.choice(workitem_list)
        kernelName = random.choice(rand_task_pool[workitem][dimension])
        ex_cpu, ex_gpu = ex_map
        if kernelName is not None:
            key = kernelName+"_"+str(workitem)
            feat_dict = create_feat_dict(key, global_map)
            extime = 0.0
            if value_int(feat_dict['Class']) > 5:
                extime = ex_cpu[key]
            else:
                extime = ex_gpu[key]
            return SimTask(key, node_index, dag_id, feat_dict, extime)
        else:
            self.randomly_select_SimTask(node_index, dag_id, rand_task_pool, global_map, workitem_range, dimension)

    def randomly_select_SimTask_key(self, node_index, dag_id, ex_map, rand_task_pool, global_map, workitem_range, dimension):
        workitem_list = [workitem_range[0]]
        workitem = workitem_range[0]
        while(workitem < workitem_range[1]):
            workitem *=2
            workitem_list.append(workitem)

        workitem = random.choice(workitem_list)
        kernelName = random.choice(rand_task_pool[workitem][dimension])
        ex_cpu, ex_gpu = ex_map
        if kernelName is not None:
            key = kernelName+"_"+str(workitem)
            feat_dict = create_feat_dict(key, global_map)
            extime = 0.0
            if value_int(feat_dict['Class']) > 5:
                extime = ex_cpu[key]
            else:
                extime = ex_gpu[key]
            return key
            # return SimTask(key, node_index, dag_id, feat_dict, extime)
        else:
            self.randomly_select_SimTask_key(node_index, dag_id, rand_task_pool, global_map, workitem_range, dimension)





    def create_dag_randomly(self, global_map, dag_id, ex_map, n, width, regular, density, workitem_range, graph_file):

        # Create random task pool indexed by data set size and work dimension
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        ex_cpu, ex_gpu = ex_map
        rand_task_pool = {k: {} for k in [128 * 2 ** (r - 1) for r in range(1, 8)]}
        rand_bufsize_pool = {}
        for k in rand_task_pool.keys():
            rand_task_pool[k] = {1: [], 2: []}
        for key in global_map.keys():
            kernel_info = key.split("_")
            kernelName = kernel_info[0]
            worksize = kernel_info[1].strip("\n")
            if kernelName == 'uncoalesced' or kernelName == 'shared':
                kernelName = kernelName + "_copy"
                worksize = kernel_info[2].strip("\n")
            if kernelName == 'transpose':
                kernelName = kernelName + "_naive"
                worksize = kernel_info[2].strip("\n")
            dataset = int(worksize)
            dimension = obtain_kernel_dimension(key)
            rand_task_pool[dataset][dimension].append(kernelName)
            print kernel_info
            feat_dict = create_feat_dict(key, global_map)
            extime = 0.0
            if value_int(feat_dict['Class']) > 5:
                extime = ex_cpu[key]
            else:
                extime = ex_gpu[key]
            simtask = SimTask(key, 0, dag_id, feat_dict, extime)
            buf_sizes = simtask.get_input_buffer_sizes()
            for buf in buf_sizes:
                if buf not in rand_bufsize_pool:
                    rand_bufsize_pool[buf] = []
                    rand_bufsize_pool[buf].append(key)
                else:
                    rand_bufsize_pool[buf].append(key)



        ex_cpu, ex_gpu = ex_map
        task_dict = dict()
        avg_tasks_per_level = exp(width * log(n))
        print avg_tasks_per_level
        level_sizes = []
        total_tasks = 0


        print "Generating Random DAG"

        # Get number of nodes per level

        while (True):
            temp = get_random_integer_around(avg_tasks_per_level, regular)
            print "Number of nodes in one level " + str(temp)
            if total_tasks + temp > n:
                temp = n - total_tasks
            level_sizes.append(temp)
            total_tasks += temp
            if (total_tasks >= n):
                break
        index = 0
        node_index = 0
        buffer_levels = []
        task_levels = []
        graph_levels = []
        buffer_dict = {}
        tasks = []
        buffers = []
        node_mapping = {}

        dag = nx.DiGraph()
        levels = []
        level_nodes = []

        for i in range(0, level_sizes[0]):
            dimension = random.choice([1, 2])
            simtask = self.randomly_select_SimTask(node_index, dag_id, ex_map, rand_task_pool,
                                                                 global_map, workitem_range, dimension)
            task_dict[node_index] = simtask
            node_mapping[index] = node_index
            G.add_node(index, shape='circle', label=simtask.name)
            task_index = index
            level_nodes.append(task_index)
            dag.add_node(node_index)
            node_index +=1
            index += 1
            input_buffers = simtask.get_input_buffer_sizes()
            output_buffers = simtask.get_output_buffer_sizes()
            print "BUFFER STATS"
            print input_buffers
            print output_buffers
            for buf_size in input_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(index, task_index)
                buffer_dict[index] = buf_size
                index+=1

            for buf_size in output_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(task_index, index)
                buffer_dict[index] = buf_size
                buffers.append(index)
                index += 1


        buffer_levels.append(buffers)
        levels.append(level_nodes)

        print "BUFFER SIZES LEVEL 0"
        print [buffer_dict[buffer_index] for buffer_index in buffers]
        print buffer_levels

        for i in range(1, len(level_sizes)):
            level_nodes = []
            bufs = buffer_levels[-1]
            input_buffer_level = []
            output_buffer_level = []

            # Number of output buffers in previous level is less than or equal to number of nodes in current level
            print "level" + str(i)
            print bufs
            buf_sizes = [buffer_dict[buffer_index] for buffer_index in bufs]
            print buf_sizes
            if len(bufs) <= level_sizes[i]:
                for buffer_index in bufs:
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool
                    key = random.choice(task_pool)
                    level_nodes.append(index)

                    dag.add_node(node_index)


                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()
                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers

                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1

                remaining = level_sizes[i] - len(bufs)


                while(remaining > 0):
                    remaining -= 1
                    buffer_index = random.choice(bufs)
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool
                    key = random.choice(task_pool)
                    level_nodes.append(index)
                    dag.add_node(node_index)

                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()

                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers

                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1
                print "BUFFER SIZES PREVIOUS LEVEL ITERATION"
                print [buffer_dict[buffer_index] for buffer_index in bufs]
                print "INPUT BUF SIZES LEVEL ITERATION"
                input_buf_sizes = [buffer_dict[buffer_index] for buffer_index in input_buffer_level]
                print input_buf_sizes
                level_edges = 0
                if 1 + int(density * len(input_buffer_level)) >= len(input_buffer_level):
                    level_edges = len(input_buffer_level)
                else:
                    level_edges = np.random.randint(1 + int(density * len(input_buffer_level)), len(input_buffer_level))
                if len(bufs) <= len(input_buffer_level):
                    edge_count = 0
                    for buffer_index in bufs:

                        pred_task = int(G.predecessors(buffer_index)[0])
                        succ_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False
                        while True:
                            required_buf_index = random.choice(input_buffer_level)
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(required_buf_index) == 0:
                                succ_task = int(G.successors(required_buf_index)[0])
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]
                                if(not dag.has_edge(source_node, target_node)):
                                    add_edge = True
                                    print "Edge does not exist"
                                    break
                                else:
                                    print "Edge exists"
                            probe_count += 1
                            if probe_count > len(input_buffer_level):
                                break
                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(buffer_index, required_buf_index)
                            edge_count += 1
                        if edge_count > level_edges:
                            break

                    if edge_count < level_edges:
                        for buffer_index in bufs:
                            for required_buf_index in input_buffer_level:
                                if buffer_dict[buffer_index] == buffer_dict[required_buf_index] and G.in_degree(required_buf_index) == 0:
                                    succ_task = int(G.successors(required_buf_index)[0])
                                    pred_task = int(G.predecessors(buffer_index)[0])
                                    if (not dag.has_edge(source_node, target_node)):
                                        dag.add_edge(source_node, target_node)
                                        G.add_edge(buffer_index, required_buf_index)
                                        edge_count +=1
                                if edge_count > level_edges:
                                    break
                            if edge_count > level_edges:
                                break



                    # for buffer_index in input_buffer_level:
                    #     if G.in_degree(buffer_index) == 0:
                    #         for required_buffer_index in bufs:
                    #             if buffer_dict[buffer_index] == buffer_dict[required_buffer_index]:
                    #                 succ_task = int(G.successors(required_buf_index)[0])
                    #                 pred_task = int(G.predecessors(buffer_index)[0])
                    #                 if (not dag.has_edge(source_node, target_node)):
                    #                     dag.add_edge(source_node, target_node)
                    #                     G.add_edge(buffer_index, required_buf_index)
                    #                     edge_count +=1
                        if edge_count >level_edges:
                            break









                bufs.append(output_buffer_level)
            else:
                buf_counter = 0
                while buf_counter < level_sizes[i]:
                    buf_counter +=1
                    buffer_index = random.choice(bufs)
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool
                    key = random.choice(task_pool)
                    level_nodes.append(index)

                    dag.add_node(node_index)


                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()

                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers
                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1
                print "BUFFER SIZES PREVIOUS LEVEL ITERATION"
                print [buffer_dict[buffer_index] for buffer_index in bufs]
                print "INPUT BUF SIZES LEVEL ITERATION"
                input_buf_sizes = [buffer_dict[buffer_index] for buffer_index in input_buffer_level]

                print input_buf_sizes

                level_edges = 0
                if 1 + int(density * len(input_buffer_level)) >= len(input_buffer_level):
                    level_edges = len(input_buffer_level)
                else:
                    level_edges = np.random.randint(1 + int(density * len(input_buffer_level)), len(input_buffer_level))
                edge_count = 0
                if len(bufs) <= len(input_buffer_level):
                    for buffer_index in bufs:
                        pred_task = int(G.predecessors(buffer_index)[0])
                        succ_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False
                        while True:
                            required_buf_index = random.choice(input_buffer_level)
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(required_buf_index) == 0:
                                succ_task = int(G.successors(required_buf_index)[0])
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]
                                if (not dag.has_edge(source_node, target_node)):
                                    add_edge = True
                                    break
                            probe_count +=1

                            if probe_count > len(input_buffer_level):
                                break

                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(buffer_index, required_buf_index)
                            edge_count += 1
                        if edge_count > level_edges:
                            break

                    if edge_count < level_edges:
                        for buffer_index in bufs:
                            for required_buf_index in input_buffer_level:
                                if buffer_dict[buffer_index] == buffer_dict[required_buf_index] and G.in_degree(
                                        required_buf_index) == 0:
                                    succ_task = int(G.successors(required_buf_index)[0])
                                    pred_task = int(G.predecessors(buffer_index)[0])
                                    if (not dag.has_edge(source_node, target_node)):
                                        dag.add_edge(source_node, target_node)
                                        G.add_edge(buffer_index, required_buf_index)
                                        edge_count += 1
                                if edge_count > level_edges:
                                    break
                            if edge_count > level_edges:
                                break







                else:

                    counter = 0
                    for buffer_index in input_buffer_level:

                        succ_task = int(G.successors(buffer_index)[0])
                        pred_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False

                        while True:
                            required_buf_index = random.choice(bufs)
                            # print buffer_index
                            # print required_buf_index
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(buffer_index) == 0:
                                print "match found"
                                pred_task = int(G.predecessors(required_buf_index)[0])
                                print pred_task
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]

                                if (not dag.has_edge(source_node, target_node)):
                                    print "Edge does not exist"
                                    add_edge = True
                                    break
                                else:
                                    print "Edge exists"
                            probe_count += 1
                            if probe_count > len(bufs):

                                break
                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(required_buf_index, buffer_index)
            print "OUTPUT BUFFER LEVEL"
            print output_buffer_level
            buffer_levels.append(output_buffer_level)


        print levels
        print buffer_levels


        G_file = "buf_task_" + graph_file[:-5] + ".png"
        G.layout(prog='dot')
        print "Dumping buffer tasks graph"
        G.draw(G_file)
        filename = graph_file
        dag_dump_file = "complete_dump_"+filename
        dag_dump = open(dag_dump_file, 'w')
        dag_dump.write(str(len(G.nodes())) + "\n")
        for edge in G.edges():
            dag_dump.write(str(edge[0]) + " " + str(edge[1]) + "\n")
        dag_dump.write("\n")
        for buffer_index in buffer_dict:
            dag_dump.write(str(buffer_index) + " " + str(buffer_dict[buffer_index]) + "\n")
        dag_dump.write("\n")
        inv_node_map = {v: k for k, v in node_mapping.iteritems()}
        for task_index in task_dict:
            dag_dump.write(str(inv_node_map[task_index]) + " " +  str(task_dict[task_index].name) + "\n")
        # dag_dump.write(str(buffer_dict))
        # dag_dump.write(str(task_dict))
        dag_dump.close()
        f = open(filename, "w")
        f.write(str(n))
        f.write("\n")
        for i in range(0, n):
            f.write(task_dict[i].name)
            f.write("\n")
        for e in dag.edges():
            u, v = e
            edge = str(u) + " " + str(v)
            f.write(edge)
            f.write("\n")
        f.close()
        simtaskdag = SimTaskDAG(task_dict, dag, dag_id, ex_map)
        viz_graph = "names_"+graph_file[:-5] + ".png"
        print "Dumping Graph"
        self.dump_graph_names(simtaskdag, viz_graph)

    def create_fan_dag_updated(self, global_map, dag_id, ex_map, outdegree, cluster_depth, cluster_devratio, n, width, regular,
                       workitem_range, graph_file, dimension):

        class DisjointSet:

            def __init__(self, node_id_list=None):

                self.num_nodes = 0
                self.current_index = 0
                self.sizes = []
                self.node_index = {}
                self.num_components = 0
                self.node_ids = []
                self.parents = []
                self.node_depth = {}
                self.component_mappings = {}
                self.size_components = []
                self.component_device_ratio = {}
                self.task_dict = {}
                self.dag_id = dag_id

                if node_id_list is None:
                    node_id_list = []
                for node_id in node_id_list:
                    self.add(node_id)

            def insert(self, x, depth, task):
                if x in self.node_ids:
                    return

                self.node_depth[x] = depth
                self.task_dict[x] = task

                self.node_ids.append(x)
                self.node_index[x] = self.current_index
                self.component_mappings[self.current_index] = [x]
                self.parents.append(self.current_index)
                self.sizes.append(1)
                self.current_index += 1

                self.num_nodes += 1
                self.num_components += 1

            def find(self, x):

                if x not in self.node_index:
                    raise ValueError('{} is not an element'.format(x))

                p = self.node_index[x]
                while p != self.parents[p]:
                    q = self.parents[p]
                    self.parents[p] = self.parents[q]
                    p = q
                return p

            def union(self, x, y):

                xroot = self.find(x)
                yroot = self.find(y)

                if xroot == yroot:
                    return
                if self.sizes[xroot] < self.sizes[yroot]:
                    self.parents[xroot] = yroot
                    self.sizes[yroot] += self.sizes[xroot]
                    for node_id in self.component_mappings[xroot]:
                        self.component_mappings[yroot].append(node_id)
                    del self.component_mappings[xroot]

                else:
                    self.parents[yroot] = xroot
                    self.sizes[xroot] += self.sizes[yroot]
                    for node_id in self.component_mappings[yroot]:
                        self.component_mappings[xroot].append(node_id)
                    del self.component_mappings[yroot]
                self.num_components -= 1

            def get_component(self, x):
                if x not in self.node_ids:
                    raise ValueError('{} is not an element'.format(x))
                nodes = np.array(self.node_ids)
                vfind = np.vectorize(self.find)
                roots = vfind(nodes)
                return set(nodes[roots == self.find(x)])

            def get_component_stats(self, x):
                node_ids = self.get_component(x)
                num_cpu = 0.0
                num_gpu = 0.0
                for node in node_ids:
                    task = self.task_dict[node]
                    if partition_class(task) is "cpu":
                        num_cpu += 1
                    else:
                        num_gpu += 1

                percent_cpu = num_cpu / (num_gpu + num_cpu)
                percent_gpu = num_gpu / (num_gpu + num_cpu)
                return percent_cpu, percent_gpu

            def print_stats(self):
                print DJ.component_mappings

        import pygraphviz as pgv

        # Initializing graphs and task pools

        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        dag = nx.DiGraph()
        ex_cpu, ex_gpu = ex_map
        rand_task_pool = {k: {} for k in [128 * 2 ** (r - 1) for r in range(1, 8)]}
        rand_bufsize_pool = {}
        for k in rand_task_pool.keys():
            rand_task_pool[k] = {1: [], 2: []}
        for key in global_map.keys():
            kernel_info = key.split("_")
            kernelName = kernel_info[0]
            worksize = kernel_info[1].strip("\n")
            if kernelName == 'uncoalesced' or kernelName == 'shared':
                kernelName = kernelName + "_copy"
                worksize = kernel_info[2].strip("\n")
            if kernelName == 'transpose':
                kernelName = kernelName + "_naive"
                worksize = kernel_info[2].strip("\n")
            dataset = int(worksize)
            dimension = obtain_kernel_dimension(key)
            rand_task_pool[dataset][dimension].append(kernelName)
            # print kernel_info
            feat_dict = create_feat_dict(key, global_map)
            extime = 0.0
            if value_int(feat_dict['Class']) > 5:
                extime = ex_cpu[key]
            else:
                extime = ex_gpu[key]
            simtask = SimTask(key, 0, dag_id, feat_dict, extime)
            buf_sizes = simtask.get_input_buffer_sizes()
            for buf in buf_sizes:
                if buf not in rand_bufsize_pool:
                    rand_bufsize_pool[buf] = []
                    rand_bufsize_pool[buf].append(key)
                else:
                    rand_bufsize_pool[buf].append(key)

            # DJ = DisjointSet()
            #
            # for id in range(0,10):
            #     key = random.choice(rand_bufsize_pool[random.choice(rand_bufsize_pool.keys())])
            #     DJ.insert(id,3,obtain_SimTask_object(key, global_map, ex_map))
            #
            # DJ.union(0,1)
            # DJ.union(1,2)
            # DJ.print_stats()
            # print DJ.get_component_stats(0)

        avg_tasks_per_level = exp(width * log(n))
        # print "Generating Random DAG"

        # Get number of nodes per level

        level_sizes = []
        total_tasks = 0
        # print "regular", regular
        while (True):
            temp = get_random_integer_around(avg_tasks_per_level, regular)
            # print "Number of nodes in one level " + str(temp)
            if len(level_sizes) > 0:
                if temp > level_sizes[-1] * outdegree:
                    temp = level_sizes[-1] * outdegree
            if total_tasks + temp > n:
                temp = n - total_tasks
            level_sizes.append(temp)
            total_tasks += temp
            if (total_tasks >= n):
                break
        # print "DAGLevelSizes ",level_sizes

        import time
        # time.sleep(5)
        # indexes for buffers and nodes
        index = 0
        node_index = 0
        buffer_levels = []
        task_levels = []
        graph_levels = []
        buffer_dict = {}
        tasks = []
        buffers = []
        node_mapping = {}
        task_dict = dict()
        levels = []
        level_nodes = []

        # Initialize first level

        DJ = DisjointSet()
        cluster_depth_map = {}
        for i in range(0, level_sizes[0]):
            # dimension = random.choice([1, 2])
            simtask = self.randomly_select_SimTask(node_index, dag_id, ex_map, rand_task_pool,
                                                   global_map, workitem_range, dimension)
            task_dict[node_index] = simtask
            node_mapping[index] = node_index
            G.add_node(index, shape='circle', label=simtask.name + "_" + partition_class(simtask))
            task_index = index
            cluster_depth_map[task_index] = cluster_depth
            level_nodes.append(task_index)
            dag.add_node(node_index)
            DJ.insert(task_index, cluster_depth, simtask)
            node_index += 1
            index += 1
            input_buffers = simtask.get_input_buffer_sizes()
            output_buffers = simtask.get_output_buffer_sizes()
            # print "BUFFER STATS"
            # print input_buffers
            # print output_buffers
            for buf_size in input_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(index, task_index)
                buffer_dict[index] = buf_size
                index += 1

            for buf_size in output_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(task_index, index)
                buffer_dict[index] = buf_size
                buffers.append(index)
                index += 1

        levels.append(level_nodes)
        for i in range(0, len(level_sizes) - 1):
            levels.append([])
        actual_n = n
        n = n - level_sizes[0]
        num_tasks = 0
        curr_level = 0
        next_level = 1
        num_next_level_tasks = 0
        curr_level_task = 0
        next_level_task = 0
        # print num_tasks
        # print "DAG nodes remaining", n

        while num_tasks < n:

            if curr_level == len(level_sizes) - 1:
                # print "Boundary case"
                num_tasks += 1
            else:

                if random.uniform(0, 1) < 0.5:
                    # print "DAG testing current level: ", curr_level, "curr_level_task: ", curr_level_task, "max_level_size: ", level_sizes[curr_level]
                    # print "FanOut Phase", num_tasks, n

                    # print curr_level_task, level_sizes[curr_level]

                    # print "Curr Level Update" ,curr_level_task,level_sizes[curr_level]
                    pending_task_creation = False
                    num_pending_tasks = 0
                    if curr_level_task == level_sizes[curr_level]:

                        if len(levels[next_level]) < level_sizes[next_level]:
                            # print "NextLevel Tasks Left to be added to next level"
                            num = 0
                            while (level_sizes[next_level] - len(levels[next_level]) > 0):



                                # key = random.choice(
                                #     self.select_database_devicetype(task_pool, random.choice(["cpu", "gpu"]),
                                #                                     global_map))
                                key = self.randomly_select_SimTask_key(node_index, dag_id, ex_map, rand_task_pool,
                                                                       global_map, workitem_range, dimension)

                                target_task = index
                                levels[next_level].append(target_task)
                                dag.add_node(node_index)
                                DJ.insert(target_task, cluster_depth, simtask)
                                cluster_depth_map[target_task] = cluster_depth
                                feat_dict = create_feat_dict(key, global_map)
                                extime = 0.0
                                if value_int(feat_dict['Class']) > 5:
                                    extime = ex_cpu[key]
                                else:
                                    extime = ex_gpu[key]
                                simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                                task_dict[node_index] = simtask
                                G.add_node(index, shape='circle', label=simtask.name + "_" + partition_class(simtask))
                                task_index = index
                                node_mapping[task_index] = node_index
                                node_index += 1
                                index += 1
                                # Generate buffer to node relationships

                                input_buffers = simtask.get_input_buffer_sizes()
                                output_buffers = simtask.get_output_buffer_sizes()

                                # print "BUFFER STATS LEVEL ITERATION"
                                # print input_buffers
                                # print output_buffers

                                input_buffer_level = []
                                output_buffer_level = []
                                # Generate input buffers to node relationships for target task

                                for buf_size in input_buffers:
                                    G.add_node(index, shape='square', label=str(buf_size))
                                    G.add_edge(index, task_index)
                                    buffer_dict[index] = buf_size
                                    input_buffer_level.append(index)
                                    index += 1

                                # Generate node to output buffer relationships for target task

                                for buf_size in output_buffers:
                                    G.add_node(index, shape='square', label=str(buf_size))
                                    G.add_edge(task_index, index)
                                    buffer_dict[index] = buf_size
                                    output_buffer_level.append(index)

                                    index += 1
                                num += 1

                        curr_level_task = 0
                        curr_level += 1
                        next_level += 1
                        num_next_level_tasks = 0
                        # print "DAG  update  curr level ", curr_level, "curr_level_task: ", curr_level_task, "max level size: ", \
                        # level_sizes[curr_level]
                        # continue
                    # print "level stats" ,curr_level, next_level, len(level_sizes)
                    # print "curr level stats", curr_level, curr_level_task, len(levels[curr_level]), level_sizes[curr_level]
                    # print "DAG level stats: ", "curr_level: ", levels[curr_level], "next_level: ", next_level, "curr_level_task: ", curr_level_task
                    if curr_level == len(level_sizes) - 1:
                        break
                    source_task = levels[curr_level][curr_level_task]
                    curr_level_task += 1
                    num_edges = 0
                    if curr_level_task == level_sizes[curr_level] - 1 and num_next_level_tasks < level_sizes[next_level]:
                        num_edges = level_sizes[next_level] - num_next_level_tasks
                        if num_edges > outdegree:
                            pending_task_creation = True
                            num_pending_tasks = num_edges - outdegree
                            num_edges = outdegree




                    else:
                        num_edges = np.random.randint(1, max(min(outdegree, level_sizes[next_level]), 2))
                    # print "num edges" , num_edges
                    source_buffer_index = int(G.successors(source_task)[0])
                    source_buffer_size = buffer_dict[source_buffer_index]

                    if len(levels[next_level]) == 0:
                        # Generate new tasks and add outgoing edges for task in current level
                        # print "DAG 0 nodes in next level: ", next_level, "required", level_sizes[curr_level]
                        for num_iter in range(0, num_edges):
                            num_next_level_tasks += 1
                            # Determine task to be generated based on cluster component device ratio stats

                            cpu_percent, gpu_percent = DJ.get_component_stats(source_task)
                            task_pool = rand_bufsize_pool[source_buffer_size]
                            key = ""
                            if gpu_percent <= 1 - cluster_devratio:
                                key = random.choice(self.select_database_devicetype(task_pool, "gpu", global_map))
                            else:
                                key = random.choice(self.select_database_devicetype(task_pool, "cpu", global_map))

                            # Generate task object and update DJ component

                            target_task = index
                            levels[next_level].append(target_task)
                            dag.add_node(node_index)
                            dag.add_edge(node_mapping[source_task], node_index)
                            feat_dict = create_feat_dict(key, global_map)
                            extime = 0.0
                            if value_int(feat_dict['Class']) > 5:
                                extime = ex_cpu[key]
                            else:
                                extime = ex_gpu[key]
                            simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                            task_dict[node_index] = simtask
                            if cluster_depth_map[source_task] > 0:
                                DJ.insert(target_task, cluster_depth_map[source_task] - 1, simtask)
                                DJ.union(source_task, target_task)
                                cluster_depth_map[target_task] = cluster_depth_map[source_task] - 1
                            else:
                                DJ.insert(target_task, cluster_depth, simtask)
                                cluster_depth_map[target_task] = cluster_depth
                            G.add_node(index, shape='circle', label=simtask.name + "_" + partition_class(simtask))
                            task_index = index
                            node_mapping[task_index] = node_index
                            node_index += 1
                            index += 1

                            # Generate buffer to node relationships

                            input_buffers = simtask.get_input_buffer_sizes()
                            output_buffers = simtask.get_output_buffer_sizes()

                            # print "BUFFER STATS LEVEL ITERATION"
                            # print input_buffers
                            # print output_buffers

                            input_buffer_level = []
                            output_buffer_level = []
                            # Generate input buffers to node relationships for target task

                            for buf_size in input_buffers:
                                G.add_node(index, shape='square', label=str(buf_size))
                                G.add_edge(index, task_index)
                                buffer_dict[index] = buf_size
                                input_buffer_level.append(index)
                                index += 1

                            # Generate node to output buffer relationships for target task

                            for buf_size in output_buffers:
                                G.add_node(index, shape='square', label=str(buf_size))
                                G.add_edge(task_index, index)
                                buffer_dict[index] = buf_size
                                output_buffer_level.append(index)
                                index += 1

                            buffer_index = int(G.successors(source_task)[0])
                            for required_buffer_index in input_buffer_level:
                                if buffer_dict[required_buffer_index] == buffer_dict[buffer_index]:
                                    G.add_edge(buffer_index, required_buffer_index)
                                    break
                        num_tasks += 1
                        # print "DAG 0 nodes in next level scenario now contains", levels[next_level]

                    else:
                        move_current_task = False
                        # print "DAG next level is not empty", next_level
                        for num_iter in range(0, num_edges):

                            # Determine task to be generated/selected based on cluster component device ratio stats
                            if move_current_task:
                                # print "DAG move current task"
                                break
                            cpu_percent, gpu_percent = DJ.get_component_stats(source_task)
                            task_pool = rand_bufsize_pool[source_buffer_size]
                            if not (num_next_level_tasks < level_sizes[next_level]):
                                target_task = -1
                                is_task_selected = False
                                buffer_index = int(G.successors(source_task)[0])
                                if gpu_percent <= 1 - cluster_devratio:
                                    for t_index in levels[next_level]:
                                        task = task_dict[node_mapping[t_index]]
                                        if partition_class(task) is "gpu" and buffer_dict[
                                            buffer_index] in task.get_input_buffer_sizes():
                                            target_task = t_index
                                            found = False
                                            required_buffer_indices = G.predecessors(target_task)
                                            for required_buffer_index in required_buffer_indices:
                                                if len(G.predecessors(required_buffer_index)) == 0:
                                                    found = True
                                                    break
                                            if found:
                                                break
                                            else:
                                                target_task = -1
                                else:
                                    for t_index in levels[next_level]:
                                        task = task_dict[node_mapping[t_index]]
                                        if partition_class(task) is "cpu" and buffer_dict[
                                            buffer_index] in task.get_input_buffer_sizes():
                                            target_task = t_index
                                            found = False
                                            required_buffer_indices = G.predecessors(target_task)
                                            for required_buffer_index in required_buffer_indices:
                                                if len(G.predecessors(required_buffer_index)) == 0:
                                                    found = True
                                                    break
                                            if found:
                                                break
                                            else:
                                                target_task = -1

                                if target_task != -1:

                                    DJ.union(source_task, target_task)
                                    required_buffer_indices = G.predecessors(target_task)
                                    for required_buffer_index in required_buffer_indices:
                                        if buffer_dict[buffer_index] == buffer_dict[int(required_buffer_index)] and len(
                                                G.predecessors(required_buffer_index)) == 0:
                                            G.add_edge(buffer_index, required_buffer_index)
                                            dag.add_edge(node_mapping[source_task], node_mapping[target_task])
                                            break
                                else:
                                    move_current_task = True


                            else:

                                key = ""

                                if gpu_percent <= 1 - cluster_devratio:
                                    key = random.choice(self.select_database_devicetype(task_pool, "gpu", global_map))
                                else:
                                    key = random.choice(self.select_database_devicetype(task_pool, "cpu", global_map))

                                # Generate task object and update DJ component

                                target_task = index
                                levels[next_level].append(target_task)
                                dag.add_node(node_index)
                                dag.add_edge(node_mapping[source_task], node_index)
                                feat_dict = create_feat_dict(key, global_map)
                                extime = 0.0
                                if value_int(feat_dict['Class']) > 5:
                                    extime = ex_cpu[key]
                                else:
                                    extime = ex_gpu[key]
                                simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                                task_dict[node_index] = simtask
                                if cluster_depth_map[source_task] > 0:
                                    DJ.insert(target_task, cluster_depth_map[source_task] - 1, simtask)
                                    DJ.union(source_task, target_task)
                                    cluster_depth_map[target_task] = cluster_depth_map[source_task] - 1
                                else:
                                    DJ.insert(target_task, cluster_depth, simtask)
                                    cluster_depth_map[target_task] = cluster_depth
                                G.add_node(index, shape='circle', label=simtask.name + "_" + partition_class(simtask))
                                task_index = index
                                node_mapping[task_index] = node_index
                                node_index += 1
                                index += 1

                                # Generate buffer to node relationships

                                input_buffers = simtask.get_input_buffer_sizes()
                                output_buffers = simtask.get_output_buffer_sizes()

                                # print "BUFFER STATS LEVEL ITERATION"
                                # print input_buffers
                                # print output_buffers

                                input_buffer_level = []
                                output_buffer_level = []
                                # Generate input buffers to node relationships for target task

                                for buf_size in input_buffers:
                                    G.add_node(index, shape='square', label=str(buf_size))
                                    G.add_edge(index, task_index)
                                    buffer_dict[index] = buf_size
                                    input_buffer_level.append(index)
                                    index += 1

                                # Generate node to output buffer relationships for target task

                                for buf_size in output_buffers:
                                    G.add_node(index, shape='square', label=str(buf_size))
                                    G.add_edge(task_index, index)
                                    buffer_dict[index] = buf_size
                                    output_buffer_level.append(index)
                                    index += 1

                                buffer_index = int(G.successors(source_task)[0])
                                for required_buffer_index in input_buffer_level:
                                    if buffer_dict[required_buffer_index] == buffer_dict[buffer_index]:
                                        G.add_edge(buffer_index, required_buffer_index)
                                        break
                                dont_make_new_task = True
                                num_next_level_tasks += 1
                        num_tasks += 1




                    if pending_task_creation:
                        pass






                else:
                    pass
                    # print "FanIn Phase"
                    # num_tasks +=1

        G_file = "buf_task_" + graph_file[:-5] + ".png"
        G.layout(prog='dot')
        # print "Dumping buffer tasks graph"
        # G.draw(G_file)
        simtaskdag = SimTaskDAG(task_dict, dag, dag_id, ex_map)
        viz_graph = "names_" + graph_file[:-5] + ".png"
        # print "Dumping Graph"
        # self.dump_graph_names(simtaskdag, viz_graph)
        filename = graph_file
        f = open(filename, "w")
        f.write(str(len(dag.nodes())))
        f.write("\n")
        for i in range(0, len(dag.nodes())):
            f.write(task_dict[i].name)
            f.write("\n")
        for e in dag.edges():
            u, v = e
            edge = str(u) + " " + str(v)
            f.write(edge)
            f.write("\n")
        f.close()

        return simtaskdag

    def create_fan_dag(self, global_map, dag_id, ex_map, outdegree, cluster_depth, cluster_devratio, n, width, regular, workitem_range, graph_file, dimension):

        class DisjointSet:

            def __init__(self, node_id_list=None):

                self.num_nodes = 0
                self.current_index = 0
                self.sizes = []
                self.node_index = {}
                self.num_components = 0
                self.node_ids = []
                self.parents = []
                self.node_depth = {}
                self.component_mappings = {}
                self.size_components = []
                self.component_device_ratio = {}
                self.task_dict = {}
                self.dag_id = dag_id

                if node_id_list is None:
                    node_id_list = []
                for node_id in node_id_list:
                    self.add(node_id)

            def insert(self, x, depth, task):
                if x in self.node_ids:
                    return

                self.node_depth[x] = depth
                self.task_dict[x] = task

                self.node_ids.append(x)
                self.node_index[x] = self.current_index
                self.component_mappings[self.current_index] = [x]
                self.parents.append(self.current_index)
                self.sizes.append(1)
                self.current_index += 1

                self.num_nodes += 1
                self.num_components += 1


            def find(self, x):

                if x not in self.node_index:
                    raise ValueError('{} is not an element'.format(x))

                p = self.node_index[x]
                while p != self.parents[p]:
                    q = self.parents[p]
                    self.parents[p] = self.parents[q]
                    p = q
                return p

            def union(self, x, y):

                xroot = self.find(x)
                yroot = self.find(y)

                if xroot == yroot:
                    return
                if self.sizes[xroot] < self.sizes[yroot]:
                    self.parents[xroot] = yroot
                    self.sizes[yroot] += self.sizes[xroot]
                    for node_id in self.component_mappings[xroot]:
                        self.component_mappings[yroot].append(node_id)
                    del self.component_mappings[xroot]

                else:
                    self.parents[yroot] = xroot
                    self.sizes[xroot] += self.sizes[yroot]
                    for node_id in self.component_mappings[yroot]:
                        self.component_mappings[xroot].append(node_id)
                    del self.component_mappings[yroot]
                self.num_components -= 1


            def get_component(self, x):
                if x not in self.node_ids:
                    raise ValueError('{} is not an element'.format(x))
                nodes = np.array(self.node_ids)
                vfind = np.vectorize(self.find)
                roots = vfind(nodes)
                return set(nodes[roots == self.find(x)])

            def get_component_stats(self, x):
                node_ids = self.get_component(x)
                num_cpu = 0.0
                num_gpu = 0.0
                for node in node_ids:
                    task = self.task_dict[node]
                    if partition_class(task) is "cpu":
                        num_cpu +=1
                    else:
                        num_gpu +=1

                percent_cpu = num_cpu/(num_gpu+num_cpu)
                percent_gpu = num_gpu / (num_gpu + num_cpu)
                return percent_cpu, percent_gpu

            def print_stats(self):
                print DJ.component_mappings



        import pygraphviz as pgv

        # Initializing graphs and task pools

        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        dag = nx.DiGraph()
        ex_cpu, ex_gpu = ex_map
        rand_task_pool = {k: {} for k in [128 * 2 ** (r - 1) for r in range(1, 8)]}
        rand_bufsize_pool = {}
        for k in rand_task_pool.keys():
            rand_task_pool[k] = {1: [], 2: []}
        for key in global_map.keys():
            kernel_info = key.split("_")
            kernelName = kernel_info[0]
            worksize = kernel_info[1].strip("\n")
            if kernelName == 'uncoalesced' or kernelName == 'shared':
                kernelName = kernelName + "_copy"
                worksize = kernel_info[2].strip("\n")
            if kernelName == 'transpose':
                kernelName = kernelName + "_naive"
                worksize = kernel_info[2].strip("\n")
            dataset = int(worksize)
            dimension = obtain_kernel_dimension(key)
            rand_task_pool[dataset][dimension].append(kernelName)
            # print kernel_info
            feat_dict = create_feat_dict(key, global_map)
            extime = 0.0
            if value_int(feat_dict['Class']) > 5:
                extime = ex_cpu[key]
            else:
                extime = ex_gpu[key]
            simtask = SimTask(key, 0, dag_id, feat_dict, extime)
            buf_sizes = simtask.get_input_buffer_sizes()
            for buf in buf_sizes:
                if buf not in rand_bufsize_pool:
                    rand_bufsize_pool[buf] = []
                    rand_bufsize_pool[buf].append(key)
                else:
                    rand_bufsize_pool[buf].append(key)

            # DJ = DisjointSet()
            #
            # for id in range(0,10):
            #     key = random.choice(rand_bufsize_pool[random.choice(rand_bufsize_pool.keys())])
            #     DJ.insert(id,3,obtain_SimTask_object(key, global_map, ex_map))
            #
            # DJ.union(0,1)
            # DJ.union(1,2)
            # DJ.print_stats()
            # print DJ.get_component_stats(0)

        avg_tasks_per_level = exp(width * log(n))
        # print "Generating Random DAG"

        # Get number of nodes per level

        level_sizes = []
        total_tasks = 0
        # print "regular", regular
        while (True):
            temp = get_random_integer_around(avg_tasks_per_level, regular)
            # print "Number of nodes in one level " + str(temp)

            if total_tasks + temp > n:
                temp = n - total_tasks
            level_sizes.append(temp)
            total_tasks += temp
            if (total_tasks >= n):
                break
        # print "DAGLevelSizes ",level_sizes

        import time
        # time.sleep(5)
        # indexes for buffers and nodes
        index = 0
        node_index = 0
        buffer_levels = []
        task_levels = []
        graph_levels = []
        buffer_dict = {}
        tasks = []
        buffers = []
        node_mapping = {}
        task_dict = dict()
        levels = []
        level_nodes = []


        # Initialize first level

        DJ = DisjointSet()
        cluster_depth_map = {}
        for i in range(0, level_sizes[0]):
            # dimension = random.choice([1, 2])
            simtask = self.randomly_select_SimTask(node_index, dag_id, ex_map, rand_task_pool,
                                                   global_map, workitem_range, dimension)
            task_dict[node_index] = simtask
            node_mapping[index] = node_index
            G.add_node(index, shape='circle', label=simtask.name+"_"+partition_class(simtask))
            task_index = index
            cluster_depth_map[task_index] = cluster_depth
            level_nodes.append(task_index)
            dag.add_node(node_index)
            DJ.insert(task_index,cluster_depth,simtask)
            node_index += 1
            index += 1
            input_buffers = simtask.get_input_buffer_sizes()
            output_buffers = simtask.get_output_buffer_sizes()
            # print "BUFFER STATS"
            # print input_buffers
            # print output_buffers
            for buf_size in input_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(index, task_index)
                buffer_dict[index] = buf_size
                index += 1

            for buf_size in output_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(task_index, index)
                buffer_dict[index] = buf_size
                buffers.append(index)
                index += 1

        levels.append(level_nodes)
        for i in range(0, len(level_sizes) - 1):
            levels.append([])
        actual_n = n
        n = n - level_sizes[0]
        num_tasks = 0
        curr_level = 0
        next_level = 1
        num_next_level_tasks = 0
        curr_level_task = 0
        next_level_task = 0
        # print num_tasks
        # print "DAG nodes remaining", n

        while num_tasks < n:

            if curr_level == len(level_sizes) - 1:
                # print "Boundary case"
                num_tasks +=1
            else:

                if random.uniform(0, 1) < 0.5:
                    # print "DAG testing current level: ", curr_level, "curr_level_task: ", curr_level_task, "max_level_size: ", level_sizes[curr_level]
                    # print "FanOut Phase", num_tasks, n

                    # print curr_level_task, level_sizes[curr_level]



                    # print "Curr Level Update" ,curr_level_task,level_sizes[curr_level]

                    if curr_level_task == level_sizes[curr_level]:

                        if len(levels[next_level]) < level_sizes[next_level]:
                            # print "NextLevel Tasks Left to be added to next level"
                            num = 0
                            while (level_sizes[next_level] - len(levels[next_level]) > 0):


                                key = random.choice(self.select_database_devicetype(task_pool, random.choice(["cpu","gpu"]), global_map))

                                target_task = index
                                levels[next_level].append(target_task)
                                dag.add_node(node_index)
                                DJ.insert(target_task, cluster_depth, simtask)
                                cluster_depth_map[target_task] = cluster_depth
                                feat_dict = create_feat_dict(key, global_map)
                                extime = 0.0
                                if value_int(feat_dict['Class']) > 5:
                                    extime = ex_cpu[key]
                                else:
                                    extime = ex_gpu[key]
                                simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                                task_dict[node_index] = simtask
                                G.add_node(index, shape='circle', label=simtask.name + "_" + partition_class(simtask))
                                task_index = index
                                node_mapping[task_index] = node_index
                                node_index += 1
                                index += 1
                                # Generate buffer to node relationships

                                input_buffers = simtask.get_input_buffer_sizes()
                                output_buffers = simtask.get_output_buffer_sizes()

                                # print "BUFFER STATS LEVEL ITERATION"
                                # print input_buffers
                                # print output_buffers

                                input_buffer_level = []
                                output_buffer_level = []
                                # Generate input buffers to node relationships for target task

                                for buf_size in input_buffers:
                                    G.add_node(index, shape='square', label=str(buf_size))
                                    G.add_edge(index, task_index)
                                    buffer_dict[index] = buf_size
                                    input_buffer_level.append(index)
                                    index += 1

                                # Generate node to output buffer relationships for target task

                                for buf_size in output_buffers:
                                    G.add_node(index, shape='square', label=str(buf_size))
                                    G.add_edge(task_index, index)
                                    buffer_dict[index] = buf_size
                                    output_buffer_level.append(index)

                                    index += 1
                                num += 1

                        curr_level_task = 0
                        curr_level +=1
                        next_level +=1
                        num_next_level_tasks = 0
                        # print "DAG  update  curr level ", curr_level, "curr_level_task: ", curr_level_task, "max level size: ", \
                        level_sizes[curr_level]
                        # continue
                    # print "level stats" ,curr_level, next_level, len(level_sizes)
                    # print "curr level stats", curr_level, curr_level_task, len(levels[curr_level]), level_sizes[curr_level]
                    # print "DAG level stats: ", "curr_level: ", levels[curr_level], "next_level: ", next_level, "curr_level_task: ", curr_level_task
                    if curr_level == len(level_sizes) - 1:
                        break
                    source_task = levels[curr_level][curr_level_task]
                    curr_level_task += 1
                    num_edges = 0
                    if curr_level_task == level_sizes[curr_level] - 1 and num_next_level_tasks < level_sizes[next_level]:
                        num_edges = level_sizes[next_level] - num_next_level_tasks

                    else:
                        num_edges = np.random.randint(1, max(min(outdegree, level_sizes[next_level]),2))
                    # print "num edges" , num_edges
                    source_buffer_index = int(G.successors(source_task)[0])
                    source_buffer_size = buffer_dict[source_buffer_index]

                    if len(levels[next_level]) == 0:
                        # Generate new tasks and add outgoing edges for task in current level
                        # print "DAG 0 nodes in next level: ", next_level, "required", level_sizes[curr_level]
                        for num_iter in range(0, num_edges):
                            num_next_level_tasks += 1
                            # Determine task to be generated based on cluster component device ratio stats

                            cpu_percent, gpu_percent = DJ.get_component_stats(source_task)
                            task_pool = rand_bufsize_pool[source_buffer_size]
                            key = ""
                            if gpu_percent <= 1 - cluster_devratio:
                                key = random.choice(self.select_database_devicetype(task_pool, "gpu", global_map))
                            else:
                                key = random.choice(self.select_database_devicetype(task_pool, "cpu", global_map))


                            # Generate task object and update DJ component

                            target_task = index
                            levels[next_level].append(target_task)
                            dag.add_node(node_index)
                            dag.add_edge(node_mapping[source_task], node_index)
                            feat_dict = create_feat_dict(key, global_map)
                            extime = 0.0
                            if value_int(feat_dict['Class']) > 5:
                                extime = ex_cpu[key]
                            else:
                                extime = ex_gpu[key]
                            simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                            task_dict[node_index] = simtask
                            if cluster_depth_map[source_task] > 0:
                                DJ.insert(target_task, cluster_depth_map[source_task]-1,simtask)
                                DJ.union(source_task, target_task)
                                cluster_depth_map[target_task] = cluster_depth_map[source_task]-1
                            else:
                                DJ.insert(target_task, cluster_depth, simtask)
                                cluster_depth_map[target_task] = cluster_depth
                            G.add_node(index, shape='circle', label=simtask.name+"_"+partition_class(simtask))
                            task_index = index
                            node_mapping[task_index] = node_index
                            node_index += 1
                            index += 1

                            # Generate buffer to node relationships

                            input_buffers = simtask.get_input_buffer_sizes()
                            output_buffers = simtask.get_output_buffer_sizes()

                            # print "BUFFER STATS LEVEL ITERATION"
                            # print input_buffers
                            # print output_buffers

                            input_buffer_level = []
                            output_buffer_level = []
                            # Generate input buffers to node relationships for target task

                            for buf_size in input_buffers:
                                G.add_node(index, shape='square', label=str(buf_size))
                                G.add_edge(index, task_index)
                                buffer_dict[index] = buf_size
                                input_buffer_level.append(index)
                                index += 1

                            # Generate node to output buffer relationships for target task

                            for buf_size in output_buffers:
                                G.add_node(index, shape='square', label=str(buf_size))
                                G.add_edge(task_index, index)
                                buffer_dict[index] = buf_size
                                output_buffer_level.append(index)
                                index += 1

                            buffer_index = int(G.successors(source_task)[0])
                            for required_buffer_index in input_buffer_level:
                                if buffer_dict[required_buffer_index] == buffer_dict[buffer_index]:
                                    G.add_edge(buffer_index, required_buffer_index)
                                    break
                        num_tasks += 1
                        # print "DAG 0 nodes in next level scenario now contains", levels[next_level]

                    else:
                        move_current_task = False
                        # print "DAG next level is not empty", next_level
                        for num_iter in range(0, num_edges):

                            # Determine task to be generated/selected based on cluster component device ratio stats
                            if move_current_task:
                                # print "DAG move current task"
                                break
                            cpu_percent, gpu_percent = DJ.get_component_stats(source_task)
                            task_pool = rand_bufsize_pool[source_buffer_size]
                            if not(num_next_level_tasks < level_sizes[next_level]):
                                target_task = -1
                                is_task_selected = False
                                buffer_index = int(G.successors(source_task)[0])
                                if gpu_percent <= 1 - cluster_devratio:
                                    for t_index in levels[next_level]:
                                        task = task_dict[node_mapping[t_index]]
                                        if partition_class(task) is "gpu" and buffer_dict[buffer_index] in task.get_input_buffer_sizes():
                                            target_task = t_index
                                            found = False
                                            required_buffer_indices = G.predecessors(target_task)
                                            for required_buffer_index in required_buffer_indices:
                                                if len(G.predecessors(required_buffer_index)) == 0:
                                                    found = True
                                                    break
                                            if found:
                                                break
                                            else:
                                                target_task = -1
                                else:
                                    for t_index in levels[next_level]:
                                        task = task_dict[node_mapping[t_index]]
                                        if partition_class(task) is "cpu" and buffer_dict[buffer_index] in task.get_input_buffer_sizes() :
                                            target_task = t_index
                                            found = False
                                            required_buffer_indices = G.predecessors(target_task)
                                            for required_buffer_index in required_buffer_indices:
                                                if len(G.predecessors(required_buffer_index)) == 0:
                                                    found = True
                                                    break
                                            if found:
                                                break
                                            else:
                                                target_task = -1

                                if target_task != -1:


                                    DJ.union(source_task, target_task)
                                    required_buffer_indices = G.predecessors(target_task)
                                    for required_buffer_index in required_buffer_indices:
                                        if buffer_dict[buffer_index] == buffer_dict[int(required_buffer_index)] and len(G.predecessors(required_buffer_index)) == 0:
                                            G.add_edge(buffer_index, required_buffer_index)
                                            dag.add_edge(node_mapping[source_task], node_mapping[target_task])
                                            break
                                else:
                                    move_current_task = True


                            else:


                                key = ""

                                if gpu_percent <= 1 - cluster_devratio:
                                    key = random.choice(self.select_database_devicetype(task_pool, "gpu", global_map))
                                else:
                                    key = random.choice(self.select_database_devicetype(task_pool, "cpu", global_map))


                                # Generate task object and update DJ component

                                target_task = index
                                levels[next_level].append(target_task)
                                dag.add_node(node_index)
                                dag.add_edge(node_mapping[source_task], node_index)
                                feat_dict = create_feat_dict(key, global_map)
                                extime = 0.0
                                if value_int(feat_dict['Class']) > 5:
                                    extime = ex_cpu[key]
                                else:
                                    extime = ex_gpu[key]
                                simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                                task_dict[node_index] = simtask
                                if cluster_depth_map[source_task] > 0:
                                    DJ.insert(target_task, cluster_depth_map[source_task]-1,simtask)
                                    DJ.union(source_task, target_task)
                                    cluster_depth_map[target_task] = cluster_depth_map[source_task]-1
                                else:
                                    DJ.insert(target_task, cluster_depth, simtask)
                                    cluster_depth_map[target_task] = cluster_depth
                                G.add_node(index, shape='circle', label=simtask.name+"_"+partition_class(simtask))
                                task_index = index
                                node_mapping[task_index] = node_index
                                node_index += 1
                                index += 1

                                # Generate buffer to node relationships

                                input_buffers = simtask.get_input_buffer_sizes()
                                output_buffers = simtask.get_output_buffer_sizes()

                                # print "BUFFER STATS LEVEL ITERATION"
                                # print input_buffers
                                # print output_buffers

                                input_buffer_level = []
                                output_buffer_level = []
                                # Generate input buffers to node relationships for target task

                                for buf_size in input_buffers:
                                    G.add_node(index, shape='square', label=str(buf_size))
                                    G.add_edge(index, task_index)
                                    buffer_dict[index] = buf_size
                                    input_buffer_level.append(index)
                                    index += 1

                                # Generate node to output buffer relationships for target task

                                for buf_size in output_buffers:
                                    G.add_node(index, shape='square', label=str(buf_size))
                                    G.add_edge(task_index, index)
                                    buffer_dict[index] = buf_size
                                    output_buffer_level.append(index)
                                    index += 1

                                buffer_index = int(G.successors(source_task)[0])
                                for required_buffer_index in input_buffer_level:
                                    if buffer_dict[required_buffer_index] == buffer_dict[buffer_index]:
                                        G.add_edge(buffer_index, required_buffer_index)
                                        break
                                dont_make_new_task = True
                                num_next_level_tasks += 1
                        num_tasks += 1











                else:
                    pass
                    # print "FanIn Phase"
                    # num_tasks +=1

        G_file = "buf_task_" + graph_file[:-5] + ".png"
        G.layout(prog='dot')
        # print "Dumping buffer tasks graph"
        # G.draw(G_file)
        simtaskdag = SimTaskDAG(task_dict, dag, dag_id, ex_map)
        viz_graph = "names_" + graph_file[:-5] + ".png"
        # print "Dumping Graph"
        # self.dump_graph_names(simtaskdag, viz_graph)
        filename = graph_file
        f = open(filename, "w")
        f.write(str(len(dag.nodes())))
        f.write("\n")
        for i in range(0, len(dag.nodes())):
            f.write(task_dict[i].name)
            f.write("\n")
        for e in dag.edges():
            u, v = e
            edge = str(u) + " " + str(v)
            f.write(edge)
            f.write("\n")
        f.close()

        return simtaskdag














    def create_param_dag_randomly(self, global_map, dag_id, ex_map, n, width, regular, density, workitem_range, graph_file, dimension):

        # Create random task pool indexed by data set size and work dimension
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        ex_cpu, ex_gpu = ex_map
        rand_task_pool = {k: {} for k in [128 * 2 ** (r - 1) for r in range(1, 8)]}
        rand_bufsize_pool = {}
        for k in rand_task_pool.keys():
            rand_task_pool[k] = {1: [], 2: []}
        for key in global_map.keys():
            kernel_info = key.split("_")
            kernelName = kernel_info[0]
            worksize = kernel_info[1].strip("\n")
            if kernelName == 'uncoalesced' or kernelName == 'shared':
                kernelName = kernelName + "_copy"
                worksize = kernel_info[2].strip("\n")
            if kernelName == 'transpose':
                kernelName = kernelName + "_naive"
                worksize = kernel_info[2].strip("\n")
            dataset = int(worksize)
            dimension = obtain_kernel_dimension(key)
            rand_task_pool[dataset][dimension].append(kernelName)
            print kernel_info
            feat_dict = create_feat_dict(key, global_map)
            extime = 0.0
            if value_int(feat_dict['Class']) > 5:
                extime = ex_cpu[key]
            else:
                extime = ex_gpu[key]
            simtask = SimTask(key, 0, dag_id, feat_dict, extime)
            buf_sizes = simtask.get_input_buffer_sizes()
            for buf in buf_sizes:
                if buf not in rand_bufsize_pool:
                    rand_bufsize_pool[buf] = []
                    rand_bufsize_pool[buf].append(key)
                else:
                    rand_bufsize_pool[buf].append(key)



        ex_cpu, ex_gpu = ex_map
        task_dict = dict()
        avg_tasks_per_level = exp(width * log(n))
        print avg_tasks_per_level
        level_sizes = []
        total_tasks = 0


        print "Generating Random DAG"

        # Get number of nodes per level

        while (True):
            temp = get_random_integer_around(avg_tasks_per_level, regular)
            print "Number of nodes in one level " + str(temp)
            if total_tasks + temp > n:
                temp = n - total_tasks
            level_sizes.append(temp)
            total_tasks += temp
            if (total_tasks >= n):
                break
        index = 0
        node_index = 0
        buffer_levels = []
        task_levels = []
        graph_levels = []
        buffer_dict = {}
        tasks = []
        buffers = []
        node_mapping = {}

        dag = nx.DiGraph()
        levels = []
        level_nodes = []

        for i in range(0, level_sizes[0]):
            # dimension = random.choice([1, 2])
            simtask = self.randomly_select_SimTask(node_index, dag_id, ex_map, rand_task_pool,
                                                                 global_map, workitem_range, dimension)
            task_dict[node_index] = simtask
            node_mapping[index] = node_index
            G.add_node(index, shape='circle', label=simtask.name)
            task_index = index
            level_nodes.append(task_index)
            dag.add_node(node_index)
            node_index +=1
            index += 1
            input_buffers = simtask.get_input_buffer_sizes()
            output_buffers = simtask.get_output_buffer_sizes()
            print "BUFFER STATS"
            print input_buffers
            print output_buffers
            for buf_size in input_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(index, task_index)
                buffer_dict[index] = buf_size
                index+=1

            for buf_size in output_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(task_index, index)
                buffer_dict[index] = buf_size
                buffers.append(index)
                index += 1


        buffer_levels.append(buffers)
        levels.append(level_nodes)

        print "BUFFER SIZES LEVEL 0"
        print [buffer_dict[buffer_index] for buffer_index in buffers]
        print buffer_levels

        for i in range(1, len(level_sizes)):
            level_nodes = []
            bufs = buffer_levels[-1]
            input_buffer_level = []
            output_buffer_level = []

            # Number of output buffers in previous level is less than or equal to number of nodes in current level
            print "level" + str(i)
            print bufs
            buf_sizes = [buffer_dict[buffer_index] for buffer_index in bufs]
            print buf_sizes
            if len(bufs) <= level_sizes[i]:
                for buffer_index in bufs:
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool

                    def create_dag_randomly(self, global_map, dag_id, ex_map, n, width, regular, density,
                                            workitem_range, graph_file):

                        # Create random task pool indexed by data set size and work dimension
                        import pygraphviz as pgv
                        G = pgv.AGraph(strict=False, directed=True)
                        G.node_attr['style'] = 'filled'
                        ex_cpu, ex_gpu = ex_map
                        rand_task_pool = {k: {} for k in [128 * 2 ** (r - 1) for r in range(1, 8)]}
                        rand_bufsize_pool = {}
                        for k in rand_task_pool.keys():
                            rand_task_pool[k] = {1: [], 2: []}
                        for key in global_map.keys():
                            kernel_info = key.split("_")
                            kernelName = kernel_info[0]
                            worksize = kernel_info[1].strip("\n")
                            if kernelName == 'uncoalesced' or kernelName == 'shared':
                                kernelName = kernelName + "_copy"
                                worksize = kernel_info[2].strip("\n")
                            if kernelName == 'transpose':
                                kernelName = kernelName + "_naive"
                                worksize = kernel_info[2].strip("\n")
                            dataset = int(worksize)
                            dimension = obtain_kernel_dimension(key)
                            rand_task_pool[dataset][dimension].append(kernelName)
                            print kernel_info
                            feat_dict = create_feat_dict(key, global_map)
                            extime = 0.0
                            if value_int(feat_dict['Class']) > 5:
                                extime = ex_cpu[key]
                            else:
                                extime = ex_gpu[key]
                            simtask = SimTask(key, 0, dag_id, feat_dict, extime)
                            buf_sizes = simtask.get_input_buffer_sizes()
                            for buf in buf_sizes:
                                if buf not in rand_bufsize_pool:
                                    rand_bufsize_pool[buf] = []
                                    rand_bufsize_pool[buf].append(key)
                                else:
                                    rand_bufsize_pool[buf].append(key)

                        ex_cpu, ex_gpu = ex_map
                        task_dict = dict()
                        avg_tasks_per_level = exp(width * log(n))
                        print avg_tasks_per_level
                        level_sizes = []
                        total_tasks = 0

                        print "Generating Random DAG"

                        # Get number of nodes per level

                        while (True):
                            temp = get_random_integer_around(avg_tasks_per_level, regular)
                            print "Number of nodes in one level " + str(temp)
                            if total_tasks + temp > n:
                                temp = n - total_tasks
                            level_sizes.append(temp)
                            total_tasks += temp
                            if (total_tasks >= n):
                                break
                        index = 0
                        node_index = 0
                        buffer_levels = []
                        task_levels = []
                        graph_levels = []
                        buffer_dict = {}
                        tasks = []
                        buffers = []
                        node_mapping = {}

                        dag = nx.DiGraph()
                        levels = []
                        level_nodes = []

                        for i in range(0, level_sizes[0]):
                            # dimension = random.choice([1, 2])
                            simtask = self.randomly_select_SimTask(node_index, dag_id, ex_map, rand_task_pool,
                                                                   global_map, workitem_range, dimension)
                            task_dict[node_index] = simtask
                            node_mapping[index] = node_index
                            G.add_node(index, shape='circle', label=simtask.name)
                            task_index = index
                            level_nodes.append(task_index)
                            dag.add_node(node_index)
                            node_index += 1
                            index += 1
                            input_buffers = simtask.get_input_buffer_sizes()
                            output_buffers = simtask.get_output_buffer_sizes()
                            print "BUFFER STATS"
                            print input_buffers
                            print output_buffers
                            for buf_size in input_buffers:
                                G.add_node(index, shape='square', label=str(buf_size))
                                G.add_edge(index, task_index)
                                buffer_dict[index] = buf_size
                                index += 1

                            for buf_size in output_buffers:
                                G.add_node(index, shape='square', label=str(buf_size))
                                G.add_edge(task_index, index)
                                buffer_dict[index] = buf_size
                                buffers.append(index)
                                index += 1

                        buffer_levels.append(buffers)
                        levels.append(level_nodes)

                        print "BUFFER SIZES LEVEL 0"
                        print [buffer_dict[buffer_index] for buffer_index in buffers]
                        print buffer_levels

                        for i in range(1, len(level_sizes)):
                            level_nodes = []
                            bufs = buffer_levels[-1]
                            input_buffer_level = []
                            output_buffer_level = []

                            # Number of output buffers in previous level is less than or equal to number of nodes in current level
                            print "level" + str(i)
                            print bufs
                            buf_sizes = [buffer_dict[buffer_index] for buffer_index in bufs]
                            print buf_sizes
                            if len(bufs) <= level_sizes[i]:
                                for buffer_index in bufs:
                                    buf = buffer_dict[buffer_index]
                                    task_pool = rand_bufsize_pool[buf]
                                    print task_pool
                                    key = random.choice(self.select_database_dimension(task_pool,dimension))
                                    level_nodes.append(index)

                                    dag.add_node(node_index)

                                    feat_dict = create_feat_dict(key, global_map)
                                    extime = 0.0
                                    if value_int(feat_dict['Class']) > 5:
                                        extime = ex_cpu[key]
                                    else:
                                        extime = ex_gpu[key]
                                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                                    task_dict[node_index] = simtask
                                    G.add_node(index, shape='circle', label=simtask.name)
                                    task_index = index
                                    node_mapping[task_index] = node_index
                                    node_index += 1
                                    index += 1
                                    input_buffers = simtask.get_input_buffer_sizes()
                                    output_buffers = simtask.get_output_buffer_sizes()
                                    print "BUFFER STATS LEVEL ITERATION"
                                    print input_buffers
                                    print output_buffers

                                    # Generate input buffers to node relationships for current level

                                    for buf_size in input_buffers:
                                        G.add_node(index, shape='square', label=str(buf_size))
                                        G.add_edge(index, task_index)
                                        buffer_dict[index] = buf_size
                                        input_buffer_level.append(index)
                                        index += 1

                                    # Generate node to output buffer relationships for current level

                                    for buf_size in output_buffers:
                                        G.add_node(index, shape='square', label=str(buf_size))
                                        G.add_edge(task_index, index)
                                        buffer_dict[index] = buf_size
                                        output_buffer_level.append(index)
                                        index += 1

                                remaining = level_sizes[i] - len(bufs)

                                while (remaining > 0):
                                    remaining -= 1
                                    buffer_index = random.choice(bufs)
                                    buf = buffer_dict[buffer_index]
                                    task_pool = rand_bufsize_pool[buf]
                                    print task_pool
                                    key = random.choice(task_pool)
                                    level_nodes.append(index)
                                    dag.add_node(node_index)

                                    feat_dict = create_feat_dict(key, global_map)
                                    extime = 0.0
                                    if value_int(feat_dict['Class']) > 5:
                                        extime = ex_cpu[key]
                                    else:
                                        extime = ex_gpu[key]
                                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                                    task_dict[node_index] = simtask
                                    G.add_node(index, shape='circle', label=simtask.name)
                                    task_index = index
                                    node_mapping[task_index] = node_index
                                    node_index += 1
                                    index += 1
                                    input_buffers = simtask.get_input_buffer_sizes()
                                    output_buffers = simtask.get_output_buffer_sizes()

                                    print "BUFFER STATS LEVEL ITERATION"
                                    print input_buffers
                                    print output_buffers

                                    # Generate input buffers to node relationships for current level

                                    for buf_size in input_buffers:
                                        G.add_node(index, shape='square', label=str(buf_size))
                                        G.add_edge(index, task_index)
                                        buffer_dict[index] = buf_size
                                        input_buffer_level.append(index)
                                        index += 1

                                    # Generate node to output buffer relationships for current level

                                    for buf_size in output_buffers:
                                        G.add_node(index, shape='square', label=str(buf_size))
                                        G.add_edge(task_index, index)
                                        buffer_dict[index] = buf_size
                                        output_buffer_level.append(index)
                                        index += 1
                                print "BUFFER SIZES PREVIOUS LEVEL ITERATION"
                                print [buffer_dict[buffer_index] for buffer_index in bufs]
                                print "INPUT BUF SIZES LEVEL ITERATION"
                                input_buf_sizes = [buffer_dict[buffer_index] for buffer_index in input_buffer_level]
                                print input_buf_sizes
                                level_edges = 0
                                if 1 + int(density * len(input_buffer_level)) >= len(input_buffer_level):
                                    level_edges = len(input_buffer_level)
                                else:
                                    level_edges = np.random.randint(1 + int(density * len(input_buffer_level)),
                                                                    len(input_buffer_level))
                                if len(bufs) <= len(input_buffer_level):
                                    edge_count = 0
                                    for buffer_index in bufs:

                                        pred_task = int(G.predecessors(buffer_index)[0])
                                        succ_task = -1
                                        source_node = -1
                                        target_node = -1
                                        required_buf_index = -1
                                        buffer_size = buffer_dict[buffer_index]
                                        probe_count = 0
                                        add_edge = False
                                        while True:
                                            required_buf_index = random.choice(input_buffer_level)
                                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(
                                                    required_buf_index) == 0:
                                                succ_task = int(G.successors(required_buf_index)[0])
                                                source_node = node_mapping[pred_task]
                                                target_node = node_mapping[succ_task]
                                                if (not dag.has_edge(source_node, target_node)):
                                                    add_edge = True
                                                    print "Edge does not exist"
                                                    break
                                                else:
                                                    print "Edge exists"
                                            probe_count += 1
                                            if probe_count > len(input_buffer_level):
                                                break
                                        if add_edge:
                                            dag.add_edge(source_node, target_node)
                                            G.add_edge(buffer_index, required_buf_index)
                                            edge_count += 1
                                        if edge_count > level_edges:
                                            break

                                    if edge_count < level_edges:
                                        for buffer_index in bufs:
                                            for required_buf_index in input_buffer_level:
                                                if buffer_dict[buffer_index] == buffer_dict[
                                                    required_buf_index] and G.in_degree(required_buf_index) == 0:
                                                    succ_task = int(G.successors(required_buf_index)[0])
                                                    pred_task = int(G.predecessors(buffer_index)[0])
                                                    if (not dag.has_edge(source_node, target_node)):
                                                        dag.add_edge(source_node, target_node)
                                                        G.add_edge(buffer_index, required_buf_index)
                                                        edge_count += 1
                                                if edge_count > level_edges:
                                                    break
                                            if edge_count > level_edges:
                                                break

                                        # for buffer_index in input_buffer_level:
                                        #     if G.in_degree(buffer_index) == 0:
                                        #         for required_buffer_index in bufs:
                                        #             if buffer_dict[buffer_index] == buffer_dict[required_buffer_index]:
                                        #                 succ_task = int(G.successors(required_buf_index)[0])
                                        #                 pred_task = int(G.predecessors(buffer_index)[0])
                                        #                 if (not dag.has_edge(source_node, target_node)):
                                        #                     dag.add_edge(source_node, target_node)
                                        #                     G.add_edge(buffer_index, required_buf_index)
                                        #                     edge_count +=1
                                        if edge_count > level_edges:
                                            break

                                bufs.append(output_buffer_level)
                            else:
                                buf_counter = 0
                                while buf_counter < level_sizes[i]:
                                    buf_counter += 1
                                    buffer_index = random.choice(bufs)
                                    buf = buffer_dict[buffer_index]
                                    task_pool = rand_bufsize_pool[buf]
                                    print task_pool
                                    key = random.choice(self.select_database_dimension(task_pool, dimension))
                                    level_nodes.append(index)

                                    dag.add_node(node_index)

                                    feat_dict = create_feat_dict(key, global_map)
                                    extime = 0.0
                                    if value_int(feat_dict['Class']) > 5:
                                        extime = ex_cpu[key]
                                    else:
                                        extime = ex_gpu[key]
                                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                                    task_dict[node_index] = simtask
                                    G.add_node(index, shape='circle', label=simtask.name)
                                    task_index = index
                                    node_mapping[task_index] = node_index
                                    node_index += 1
                                    index += 1
                                    input_buffers = simtask.get_input_buffer_sizes()
                                    output_buffers = simtask.get_output_buffer_sizes()

                                    print "BUFFER STATS LEVEL ITERATION"
                                    print input_buffers
                                    print output_buffers
                                    # Generate input buffers to node relationships for current level

                                    for buf_size in input_buffers:
                                        G.add_node(index, shape='square', label=str(buf_size))
                                        G.add_edge(index, task_index)
                                        buffer_dict[index] = buf_size
                                        input_buffer_level.append(index)
                                        index += 1

                                    # Generate node to output buffer relationships for current level

                                    for buf_size in output_buffers:
                                        G.add_node(index, shape='square', label=str(buf_size))
                                        G.add_edge(task_index, index)
                                        buffer_dict[index] = buf_size
                                        output_buffer_level.append(index)
                                        index += 1
                                print "BUFFER SIZES PREVIOUS LEVEL ITERATION"
                                print [buffer_dict[buffer_index] for buffer_index in bufs]
                                print "INPUT BUF SIZES LEVEL ITERATION"
                                input_buf_sizes = [buffer_dict[buffer_index] for buffer_index in input_buffer_level]

                                print input_buf_sizes

                                level_edges = 0
                                if 1 + int(density * len(input_buffer_level)) >= len(input_buffer_level):
                                    level_edges = len(input_buffer_level)
                                else:
                                    level_edges = np.random.randint(1 + int(density * len(input_buffer_level)),
                                                                    len(input_buffer_level))
                                edge_count = 0
                                if len(bufs) <= len(input_buffer_level):
                                    for buffer_index in bufs:
                                        pred_task = int(G.predecessors(buffer_index)[0])
                                        succ_task = -1
                                        source_node = -1
                                        target_node = -1
                                        required_buf_index = -1
                                        buffer_size = buffer_dict[buffer_index]
                                        probe_count = 0
                                        add_edge = False
                                        while True:
                                            required_buf_index = random.choice(input_buffer_level)
                                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(
                                                    required_buf_index) == 0:
                                                succ_task = int(G.successors(required_buf_index)[0])
                                                source_node = node_mapping[pred_task]
                                                target_node = node_mapping[succ_task]
                                                if (not dag.has_edge(source_node, target_node)):
                                                    add_edge = True
                                                    break
                                            probe_count += 1

                                            if probe_count > len(input_buffer_level):
                                                break

                                        if add_edge:
                                            dag.add_edge(source_node, target_node)
                                            G.add_edge(buffer_index, required_buf_index)
                                            edge_count += 1
                                        if edge_count > level_edges:
                                            break

                                    if edge_count < level_edges:
                                        for buffer_index in bufs:
                                            for required_buf_index in input_buffer_level:
                                                if buffer_dict[buffer_index] == buffer_dict[
                                                    required_buf_index] and G.in_degree(
                                                        required_buf_index) == 0:
                                                    succ_task = int(G.successors(required_buf_index)[0])
                                                    pred_task = int(G.predecessors(buffer_index)[0])
                                                    if (not dag.has_edge(source_node, target_node)):
                                                        dag.add_edge(source_node, target_node)
                                                        G.add_edge(buffer_index, required_buf_index)
                                                        edge_count += 1
                                                if edge_count > level_edges:
                                                    break
                                            if edge_count > level_edges:
                                                break







                                else:

                                    counter = 0
                                    for buffer_index in input_buffer_level:

                                        succ_task = int(G.successors(buffer_index)[0])
                                        pred_task = -1
                                        source_node = -1
                                        target_node = -1
                                        required_buf_index = -1
                                        buffer_size = buffer_dict[buffer_index]
                                        probe_count = 0
                                        add_edge = False

                                        while True:
                                            required_buf_index = random.choice(bufs)
                                            # print buffer_index
                                            # print required_buf_index
                                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(
                                                    buffer_index) == 0:
                                                print "match found"
                                                pred_task = int(G.predecessors(required_buf_index)[0])
                                                print pred_task
                                                source_node = node_mapping[pred_task]
                                                target_node = node_mapping[succ_task]

                                                if (not dag.has_edge(source_node, target_node)):
                                                    print "Edge does not exist"
                                                    add_edge = True
                                                    break
                                                else:
                                                    print "Edge exists"
                                            probe_count += 1
                                            if probe_count > len(bufs):
                                                break
                                        if add_edge:
                                            dag.add_edge(source_node, target_node)
                                            G.add_edge(required_buf_index, buffer_index)
                            print "OUTPUT BUFFER LEVEL"
                            print output_buffer_level
                            buffer_levels.append(output_buffer_level)

                        print levels
                        print buffer_levels

                        G_file = "buf_task_" + graph_file[:-5] + ".png"
                        G.layout(prog='dot')
                        print "Dumping buffer tasks graph"
                        G.draw(G_file)
                        filename = graph_file
                        dag_dump_file = "complete_dump_" + filename
                        dag_dump = open(dag_dump_file, 'w')
                        dag_dump.write(str(len(G.nodes())) + "\n")
                        for edge in G.edges():
                            dag_dump.write(str(edge[0]) + " " + str(edge[1]) + "\n")
                        dag_dump.write("\n")
                        for buffer_index in buffer_dict:
                            dag_dump.write(str(buffer_index) + " " + str(buffer_dict[buffer_index]) + "\n")
                        dag_dump.write("\n")
                        inv_node_map = {v: k for k, v in node_mapping.iteritems()}
                        for task_index in task_dict:
                            dag_dump.write(str(inv_node_map[task_index]) + " " + str(task_dict[task_index].name) + "\n")
                        # dag_dump.write(str(buffer_dict))
                        # dag_dump.write(str(task_dict))
                        dag_dump.close()
                        f = open(filename, "w")
                        f.write(str(n))
                        f.write("\n")
                        for i in range(0, n):
                            f.write(task_dict[i].name)
                            f.write("\n")
                        for e in dag.edges():
                            u, v = e
                            edge = str(u) + " " + str(v)
                            f.write(edge)
                            f.write("\n")
                        f.close()
                        simtaskdag = SimTaskDAG(task_dict, dag, dag_id, ex_map)
                        viz_graph = "names_" + graph_file[:-5] + ".png"
                        print "Dumping Graph"
                        self.dump_graph_names(simtaskdag, viz_graph)

                    level_nodes.append(index)

                    dag.add_node(node_index)


                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()
                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers

                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1

                remaining = level_sizes[i] - len(bufs)


                while(remaining > 0):
                    remaining -= 1
                    buffer_index = random.choice(bufs)
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool
                    key = random.choice(self.select_database_dimension(task_pool, dimension))
                    level_nodes.append(index)
                    dag.add_node(node_index)

                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()

                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers

                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1
                print "BUFFER SIZES PREVIOUS LEVEL ITERATION"
                print [buffer_dict[buffer_index] for buffer_index in bufs]
                print "INPUT BUF SIZES LEVEL ITERATION"
                input_buf_sizes = [buffer_dict[buffer_index] for buffer_index in input_buffer_level]
                print input_buf_sizes
                level_edges = 0
                if 1 + int(density * len(input_buffer_level)) >= len(input_buffer_level):
                    level_edges = len(input_buffer_level)
                else:
                    level_edges = np.random.randint(1 + int(density * len(input_buffer_level)), len(input_buffer_level))
                if len(bufs) <= len(input_buffer_level):
                    edge_count = 0
                    for buffer_index in bufs:

                        pred_task = int(G.predecessors(buffer_index)[0])
                        succ_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False
                        while True:
                            required_buf_index = random.choice(input_buffer_level)
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(required_buf_index) == 0:
                                succ_task = int(G.successors(required_buf_index)[0])
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]
                                if(not dag.has_edge(source_node, target_node)):
                                    add_edge = True
                                    print "Edge does not exist"
                                    break
                                else:
                                    print "Edge exists"
                            probe_count += 1
                            if probe_count > len(input_buffer_level):
                                break
                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(buffer_index, required_buf_index)
                            edge_count += 1
                        if edge_count > level_edges:
                            break

                    if edge_count < level_edges:
                        for buffer_index in bufs:
                            for required_buf_index in input_buffer_level:
                                if buffer_dict[buffer_index] == buffer_dict[required_buf_index] and G.in_degree(required_buf_index) == 0:
                                    succ_task = int(G.successors(required_buf_index)[0])
                                    pred_task = int(G.predecessors(buffer_index)[0])
                                    if (not dag.has_edge(source_node, target_node)):
                                        dag.add_edge(source_node, target_node)
                                        G.add_edge(buffer_index, required_buf_index)
                                        edge_count +=1
                                if edge_count > level_edges:
                                    break
                            if edge_count > level_edges:
                                break



                    # for buffer_index in input_buffer_level:
                    #     if G.in_degree(buffer_index) == 0:
                    #         for required_buffer_index in bufs:
                    #             if buffer_dict[buffer_index] == buffer_dict[required_buffer_index]:
                    #                 succ_task = int(G.successors(required_buf_index)[0])
                    #                 pred_task = int(G.predecessors(buffer_index)[0])
                    #                 if (not dag.has_edge(source_node, target_node)):
                    #                     dag.add_edge(source_node, target_node)
                    #                     G.add_edge(buffer_index, required_buf_index)
                    #                     edge_count +=1
                        if edge_count >level_edges:
                            break









                bufs.append(output_buffer_level)
            else:
                buf_counter = 0
                while buf_counter < level_sizes[i]:
                    buf_counter +=1
                    buffer_index = random.choice(bufs)
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool
                    key = random.choice(self.select_database_dimension(task_pool, dimension))
                    level_nodes.append(index)

                    dag.add_node(node_index)


                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()

                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers
                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1
                print "BUFFER SIZES PREVIOUS LEVEL ITERATION"
                print [buffer_dict[buffer_index] for buffer_index in bufs]
                print "INPUT BUF SIZES LEVEL ITERATION"
                input_buf_sizes = [buffer_dict[buffer_index] for buffer_index in input_buffer_level]

                print input_buf_sizes

                level_edges = 0
                if 1 + int(density * len(input_buffer_level)) >= len(input_buffer_level):
                    level_edges = len(input_buffer_level)
                else:
                    level_edges = np.random.randint(1 + int(density * len(input_buffer_level)), len(input_buffer_level))
                edge_count = 0
                if len(bufs) <= len(input_buffer_level):
                    for buffer_index in bufs:
                        pred_task = int(G.predecessors(buffer_index)[0])
                        succ_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False
                        while True:
                            required_buf_index = random.choice(input_buffer_level)
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(required_buf_index) == 0:
                                succ_task = int(G.successors(required_buf_index)[0])
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]
                                if (not dag.has_edge(source_node, target_node)):
                                    add_edge = True
                                    break
                            probe_count +=1

                            if probe_count > len(input_buffer_level):
                                break

                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(buffer_index, required_buf_index)
                            edge_count += 1
                        if edge_count > level_edges:
                            break

                    if edge_count < level_edges:
                        for buffer_index in bufs:
                            for required_buf_index in input_buffer_level:
                                if buffer_dict[buffer_index] == buffer_dict[required_buf_index] and G.in_degree(
                                        required_buf_index) == 0:
                                    succ_task = int(G.successors(required_buf_index)[0])
                                    pred_task = int(G.predecessors(buffer_index)[0])
                                    if (not dag.has_edge(source_node, target_node)):
                                        dag.add_edge(source_node, target_node)
                                        G.add_edge(buffer_index, required_buf_index)
                                        edge_count += 1
                                if edge_count > level_edges:
                                    break
                            if edge_count > level_edges:
                                break







                else:

                    counter = 0
                    for buffer_index in input_buffer_level:

                        succ_task = int(G.successors(buffer_index)[0])
                        pred_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False

                        while True:
                            required_buf_index = random.choice(bufs)
                            # print buffer_index
                            # print required_buf_index
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(buffer_index) == 0:
                                print "match found"
                                pred_task = int(G.predecessors(required_buf_index)[0])
                                print pred_task
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]

                                if (not dag.has_edge(source_node, target_node)):
                                    print "Edge does not exist"
                                    add_edge = True
                                    break
                                else:
                                    print "Edge exists"
                            probe_count += 1
                            if probe_count > len(bufs):

                                break
                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(required_buf_index, buffer_index)
            print "OUTPUT BUFFER LEVEL"
            print output_buffer_level
            buffer_levels.append(output_buffer_level)


        print levels
        print buffer_levels


        G_file = "buf_task_" + graph_file[:-5] + ".png"
        G.layout(prog='dot')
        print "Dumping buffer tasks graph"
        G.draw(G_file)
        filename = graph_file
        dag_dump_file = "complete_dump_"+filename
        dag_dump = open(dag_dump_file, 'w')
        dag_dump.write(str(len(G.nodes())) + "\n")
        for edge in G.edges():
            dag_dump.write(str(edge[0]) + " " + str(edge[1]) + "\n")
        dag_dump.write("\n")
        for buffer_index in buffer_dict:
            dag_dump.write(str(buffer_index) + " " + str(buffer_dict[buffer_index]) + "\n")
        dag_dump.write("\n")
        inv_node_map = {v: k for k, v in node_mapping.iteritems()}
        for task_index in task_dict:
            dag_dump.write(str(inv_node_map[task_index]) + " " +  str(task_dict[task_index].name) + "\n")
        # dag_dump.write(str(buffer_dict))
        # dag_dump.write(str(task_dict))
        dag_dump.close()
        f = open(filename, "w")
        f.write(str(n))
        f.write("\n")
        for i in range(0, n):
            f.write(task_dict[i].name)
            f.write("\n")
        for e in dag.edges():
            u, v = e
            edge = str(u) + " " + str(v)
            f.write(edge)
            f.write("\n")
        f.close()
        simtaskdag = SimTaskDAG(task_dict, dag, dag_id, ex_map)
        viz_graph = "names_"+graph_file[:-5] + ".png"
        print "Dumping Graph"
        self.dump_graph_names(simtaskdag, viz_graph)


    def create_clustered_dag_randomly(self, global_map, dag_id, ex_map, n, width, regular, density, workitem_range, graph_file, percent_GPU):

        # Create random task pool indexed by data set size and work dimension
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        ex_cpu, ex_gpu = ex_map
        rand_task_pool = {k: {} for k in [128 * 2 ** (r - 1) for r in range(1, 8)]}
        rand_bufsize_pool = {}
        rand_dev_bufsize_pool = {'cpu': {}, 'gpu': {}}
        for k in rand_task_pool.keys():
            rand_task_pool[k] = {1: [], 2: []}
        for key in global_map.keys():
            kernel_info = key.split("_")
            kernelName = kernel_info[0]
            worksize = kernel_info[1].strip("\n")
            if kernelName == 'uncoalesced' or kernelName == 'shared':
                kernelName = kernelName + "_copy"
                worksize = kernel_info[2].strip("\n")
            if kernelName == 'transpose':
                kernelName = kernelName + "_naive"
                worksize = kernel_info[2].strip("\n")
            dataset = int(worksize)
            dimension = obtain_kernel_dimension(key)
            rand_task_pool[dataset][dimension].append(kernelName)
            print kernel_info
            feat_dict = create_feat_dict(key, global_map)
            extime = 0.0
            device_class = ""
            if value_int(feat_dict['Class']) > 5:
                extime = ex_cpu[key]
                device_class = "cpu"
            else:
                extime = ex_gpu[key]
                device_class = "gpu"
            simtask = SimTask(key, 0, dag_id, feat_dict, extime)
            buf_sizes = simtask.get_input_buffer_sizes()
            for buf in buf_sizes:
                if buf not in rand_dev_bufsize_pool:
                    rand_bufsize_pool[buf] = []
                    rand_bufsize_pool[buf].append(key)
                else:
                    rand_bufsize_pool[buf].append(key)



        ex_cpu, ex_gpu = ex_map
        task_dict = dict()
        avg_tasks_per_level = exp(width * log(n))
        print avg_tasks_per_level
        level_sizes = []
        total_tasks = 0


        print "Generating Random DAG"

        # Get number of nodes per level

        while (True):
            temp = get_random_integer_around(avg_tasks_per_level, regular)
            print "Number of nodes in one level " + str(temp)
            if total_tasks + temp > n:
                temp = n - total_tasks
            level_sizes.append(temp)
            total_tasks += temp
            if (total_tasks >= n):
                break
        index = 0
        node_index = 0
        buffer_levels = []
        task_levels = []
        graph_levels = []
        buffer_dict = {}
        tasks = []
        buffers = []
        node_mapping = {}

        dag = nx.DiGraph()
        levels = []
        level_nodes = []

        for i in range(0, level_sizes[0]):
            dimension = random.choice([1, 2])
            simtask = self.randomly_select_SimTask(node_index, dag_id, ex_map, rand_task_pool,
                                                                 global_map, workitem_range, dimension)
            task_dict[node_index] = simtask
            node_mapping[index] = node_index
            G.add_node(index, shape='circle', label=simtask.name)
            task_index = index
            level_nodes.append(task_index)
            dag.add_node(node_index)
            node_index +=1
            index += 1
            input_buffers = simtask.get_input_buffer_sizes()
            output_buffers = simtask.get_output_buffer_sizes()
            print "BUFFER STATS"
            print input_buffers
            print output_buffers
            for buf_size in input_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(index, task_index)
                buffer_dict[index] = buf_size
                index+=1

            for buf_size in output_buffers:
                G.add_node(index, shape='square', label=str(buf_size))
                G.add_edge(task_index, index)
                buffer_dict[index] = buf_size
                buffers.append(index)
                index += 1


        buffer_levels.append(buffers)
        levels.append(level_nodes)

        print "BUFFER SIZES LEVEL 0"
        print [buffer_dict[buffer_index] for buffer_index in buffers]
        print buffer_levels

        for i in range(1, len(level_sizes)):
            level_nodes = []
            bufs = buffer_levels[-1]
            input_buffer_level = []
            output_buffer_level = []

            # Number of output buffers in previous level is less than or equal to number of nodes in current level
            print "level" + str(i)
            print bufs
            buf_sizes = [buffer_dict[buffer_index] for buffer_index in bufs]
            print buf_sizes
            if len(bufs) <= level_sizes[i]:
                for buffer_index in bufs:
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool
                    key = random.choice(task_pool)
                    level_nodes.append(index)

                    dag.add_node(node_index)


                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()
                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers

                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1

                remaining = level_sizes[i] - len(bufs)


                while(remaining > 0):
                    remaining -= 1
                    buffer_index = random.choice(bufs)
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool
                    key = random.choice(task_pool)
                    level_nodes.append(index)
                    dag.add_node(node_index)

                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()

                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers

                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1
                print "BUFFER SIZES PREVIOUS LEVEL ITERATION"
                print [buffer_dict[buffer_index] for buffer_index in bufs]
                print "INPUT BUF SIZES LEVEL ITERATION"
                input_buf_sizes = [buffer_dict[buffer_index] for buffer_index in input_buffer_level]
                print input_buf_sizes
                level_edges = 0
                if 1 + int(density * len(input_buffer_level)) >= len(input_buffer_level):
                    level_edges = len(input_buffer_level)
                else:
                    level_edges = np.random.randint(1 + int(density * len(input_buffer_level)), len(input_buffer_level))
                if len(bufs) <= len(input_buffer_level):
                    edge_count = 0
                    for buffer_index in bufs:

                        pred_task = int(G.predecessors(buffer_index)[0])
                        succ_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False
                        while True:
                            required_buf_index = random.choice(input_buffer_level)
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(required_buf_index) == 0:
                                succ_task = int(G.successors(required_buf_index)[0])
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]
                                if(not dag.has_edge(source_node, target_node)):
                                    add_edge = True
                                    print "Edge does not exist"
                                    break
                                else:
                                    print "Edge exists"
                            probe_count += 1
                            if probe_count > len(input_buffer_level):
                                break
                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(buffer_index, required_buf_index)
                            edge_count += 1
                        if edge_count > level_edges:
                            break

                    if edge_count < level_edges:
                        for buffer_index in bufs:
                            for required_buf_index in input_buffer_level:
                                if buffer_dict[buffer_index] == buffer_dict[required_buf_index] and G.in_degree(required_buf_index) == 0:
                                    succ_task = int(G.successors(required_buf_index)[0])
                                    pred_task = int(G.predecessors(buffer_index)[0])
                                    if (not dag.has_edge(source_node, target_node)):
                                        dag.add_edge(source_node, target_node)
                                        G.add_edge(buffer_index, required_buf_index)
                                        edge_count +=1
                                if edge_count > level_edges:
                                    break
                            if edge_count > level_edges:
                                break



                    # for buffer_index in input_buffer_level:
                    #     if G.in_degree(buffer_index) == 0:
                    #         for required_buffer_index in bufs:
                    #             if buffer_dict[buffer_index] == buffer_dict[required_buffer_index]:
                    #                 succ_task = int(G.successors(required_buf_index)[0])
                    #                 pred_task = int(G.predecessors(buffer_index)[0])
                    #                 if (not dag.has_edge(source_node, target_node)):
                    #                     dag.add_edge(source_node, target_node)
                    #                     G.add_edge(buffer_index, required_buf_index)
                    #                     edge_count +=1
                        if edge_count >level_edges:
                            break









                bufs.append(output_buffer_level)
            else:
                buf_counter = 0
                while buf_counter < level_sizes[i]:
                    buf_counter +=1
                    buffer_index = random.choice(bufs)
                    buf = buffer_dict[buffer_index]
                    task_pool = rand_bufsize_pool[buf]
                    print task_pool
                    key = random.choice(task_pool)
                    level_nodes.append(index)

                    dag.add_node(node_index)


                    feat_dict = create_feat_dict(key, global_map)
                    extime = 0.0
                    if value_int(feat_dict['Class']) > 5:
                        extime = ex_cpu[key]
                    else:
                        extime = ex_gpu[key]
                    simtask = SimTask(key, node_index, dag_id, feat_dict, extime)
                    task_dict[node_index] = simtask
                    G.add_node(index, shape='circle', label=simtask.name)
                    task_index = index
                    node_mapping[task_index] = node_index
                    node_index += 1
                    index += 1
                    input_buffers = simtask.get_input_buffer_sizes()
                    output_buffers = simtask.get_output_buffer_sizes()

                    print "BUFFER STATS LEVEL ITERATION"
                    print input_buffers
                    print output_buffers
                    # Generate input buffers to node relationships for current level

                    for buf_size in input_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(index, task_index)
                        buffer_dict[index] = buf_size
                        input_buffer_level.append(index)
                        index += 1

                    # Generate node to output buffer relationships for current level

                    for buf_size in output_buffers:
                        G.add_node(index, shape='square', label=str(buf_size))
                        G.add_edge(task_index, index)
                        buffer_dict[index] = buf_size
                        output_buffer_level.append(index)
                        index += 1
                print "BUFFER SIZES PREVIOUS LEVEL ITERATION"
                print [buffer_dict[buffer_index] for buffer_index in bufs]
                print "INPUT BUF SIZES LEVEL ITERATION"
                input_buf_sizes = [buffer_dict[buffer_index] for buffer_index in input_buffer_level]

                print input_buf_sizes

                level_edges = 0
                if 1 + int(density * len(input_buffer_level)) >= len(input_buffer_level):
                    level_edges = len(input_buffer_level)
                else:
                    level_edges = np.random.randint(1 + int(density * len(input_buffer_level)), len(input_buffer_level))
                edge_count = 0
                if len(bufs) <= len(input_buffer_level):
                    for buffer_index in bufs:
                        pred_task = int(G.predecessors(buffer_index)[0])
                        succ_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False
                        while True:
                            required_buf_index = random.choice(input_buffer_level)
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(required_buf_index) == 0:
                                succ_task = int(G.successors(required_buf_index)[0])
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]
                                if (not dag.has_edge(source_node, target_node)):
                                    add_edge = True
                                    break
                            probe_count +=1

                            if probe_count > len(input_buffer_level):
                                break

                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(buffer_index, required_buf_index)
                            edge_count += 1
                        if edge_count > level_edges:
                            break

                    if edge_count < level_edges:
                        for buffer_index in bufs:
                            for required_buf_index in input_buffer_level:
                                if buffer_dict[buffer_index] == buffer_dict[required_buf_index] and G.in_degree(
                                        required_buf_index) == 0:
                                    succ_task = int(G.successors(required_buf_index)[0])
                                    pred_task = int(G.predecessors(buffer_index)[0])
                                    if (not dag.has_edge(source_node, target_node)):
                                        dag.add_edge(source_node, target_node)
                                        G.add_edge(buffer_index, required_buf_index)
                                        edge_count += 1
                                if edge_count > level_edges:
                                    break
                            if edge_count > level_edges:
                                break







                else:

                    counter = 0
                    for buffer_index in input_buffer_level:

                        succ_task = int(G.successors(buffer_index)[0])
                        pred_task = -1
                        source_node = -1
                        target_node = -1
                        required_buf_index = -1
                        buffer_size = buffer_dict[buffer_index]
                        probe_count = 0
                        add_edge = False

                        while True:
                            required_buf_index = random.choice(bufs)
                            # print buffer_index
                            # print required_buf_index
                            if buffer_dict[required_buf_index] == buffer_size and G.in_degree(buffer_index) == 0:
                                print "match found"
                                pred_task = int(G.predecessors(required_buf_index)[0])
                                print pred_task
                                source_node = node_mapping[pred_task]
                                target_node = node_mapping[succ_task]

                                if (not dag.has_edge(source_node, target_node)):
                                    print "Edge does not exist"
                                    add_edge = True
                                    break
                                else:
                                    print "Edge exists"
                            probe_count += 1
                            if probe_count > len(bufs):

                                break
                        if add_edge:
                            dag.add_edge(source_node, target_node)
                            G.add_edge(required_buf_index, buffer_index)
            print "OUTPUT BUFFER LEVEL"
            print output_buffer_level
            buffer_levels.append(output_buffer_level)


        print levels
        print buffer_levels


        G_file = "buf_task_" + graph_file[:-5] + ".png"
        G.layout(prog='dot')
        print "Dumping buffer tasks graph"
        G.draw(G_file)
        filename = graph_file
        dag_dump_file = "complete_dump_"+filename
        dag_dump = open(dag_dump_file, 'w')
        dag_dump.write(str(len(G.nodes())) + "\n")
        for edge in G.edges():
            dag_dump.write(str(edge[0]) + " " + str(edge[1]) + "\n")
        dag_dump.write("\n")
        for buffer_index in buffer_dict:
            dag_dump.write(str(buffer_index) + " " + str(buffer_dict[buffer_index]) + "\n")
        dag_dump.write("\n")
        inv_node_map = {v: k for k, v in node_mapping.iteritems()}
        for task_index in task_dict:
            dag_dump.write(str(inv_node_map[task_index]) + " " +  str(task_dict[task_index].name) + "\n")
        # dag_dump.write(str(buffer_dict))
        # dag_dump.write(str(task_dict))
        dag_dump.close()
        f = open(filename, "w")
        f.write(str(n))
        f.write("\n")
        for i in range(0, n):
            f.write(task_dict[i].name)
            f.write("\n")
        for e in dag.edges():
            u, v = e
            edge = str(u) + " " + str(v)
            f.write(edge)
            f.write("\n")
        f.close()
        simtaskdag = SimTaskDAG(task_dict, dag, dag_id, ex_map)
        viz_graph = "names_"+graph_file[:-5] + ".png"
        print "Dumping Graph"
        self.dump_graph_names(simtaskdag, viz_graph)

    def create_OpenCLDAG_randomly(self, global_map, dag_id, ex_map, n, width, regular, density, workitem_range, graph_file):
        """
        Takes a set of parameters pertaining to the nature of the graph, a dictionary of execution time estimates and
        randomly generates a DAG of OpenCL Kernels

        Args:
            global_map ():
            dag_id ():
            ex_map ():
            n ():
            width ():
            regular ():
            density ():
            workitem_range ():

        """

        # Create task pool indexed according to workitem size and workdimension

        rand_task_pool ={k : {} for k in [128 * 2 ** (r-1) for r in range(1,8)]}
        for k in rand_task_pool.keys():
            rand_task_pool[k] = {1: [], 2: []}
        for key in global_map.keys():
            kernel_info = key.split("_")
            kernelName = kernel_info[0]
            worksize = kernel_info[1].strip("\n")
            if kernelName == 'uncoalesced' or kernelName == 'shared':
                kernelName = kernelName + "_copy"
                worksize = kernel_info[2].strip("\n")
            if kernelName == 'transpose':
                kernelName = kernelName + "_naive"
                worksize = kernel_info[2].strip("\n")
            dataset = int(worksize)
            dimension = obtain_kernel_dimension(key)
            rand_task_pool[dataset][dimension].append(kernelName)


        ex_cpu, ex_gpu = ex_map
        task_dict = dict()
        avg_tasks_per_level = exp(width * log(n))
        print avg_tasks_per_level
        level_sizes = []
        total_tasks = 0
        # Determine contents of levels and populates with node ids for DAG
        print "Generating Random DAG"
        while (True):
            temp = get_random_integer_around(avg_tasks_per_level, regular)
            print "Number of nodes in one level " + str(temp)
            if total_tasks + temp > n:
                temp = n - total_tasks
            level_sizes.append(temp)
            total_tasks += temp
            if (total_tasks >= n):
                break
        print level_sizes
        node_index = 0
        dag = nx.DiGraph()
        levels = []
        for level in level_sizes:
            level_nodes = []
            for i in range(0, level):
                dag.add_node(node_index)
                level_nodes.append(node_index)
                node_index += 1
            levels.append(level_nodes)
        print dag.nodes()
        print levels

        # Constrained SimTask node generation for the DAG

        # Initialize level 1
        for i in range(0, len(levels[0])):
            node_index = levels[0][i]
            dimension = random.choice([1, 2])
            task_dict[node_index] = self.randomly_select_SimTask(node_index, dag_id, ex_map, rand_task_pool, global_map, workitem_range, dimension)
            print "Level 0"
            print node_index
            print task_dict[node_index].name
            print task_dict[node_index].get_dimension()

        # First Pass Update levels
        for i in range(1, len(levels)):
            for j in range(0, len(levels[i])):
                task_node = levels[i][j]
                num_parents = min(1 + int(density * len(levels[i - 1])), len(levels[i - 1]))
                existing_parents = []
                for k in range(0, num_parents):
                    parent_index = np.random.randint(0, len(levels[i - 1]))
                    if parent_index not in existing_parents:
                        parent_node = levels[i - 1][parent_index]
                        print parent_node
                        dataset = task_dict[parent_node].get_dataset()
                        dimension = task_dict[parent_node].get_dimension()
                        if task_node not in task_dict.keys():
                            task_dict[task_node] = self.randomly_select_SimTask(task_node, dag_id, ex_map, rand_task_pool, global_map, [dataset, dataset], dimension)

                        if task_dict[task_node].get_dataset() == task_dict[parent_node].get_dataset():
                            if dag.in_degree(task_node) < task_dict[task_node].get_num_input_buffers():
                                dag.add_edge(parent_node, task_node, weight=0.0, time=0.0)


                        existing_parents.append(parent_index)

        # Second pass Update levels for disconnected nodes

        for i in range(0, len(levels)-1):
            for j in range(0, len(levels[i])):
                task_node = levels[i][j]
                if dag.out_degree(task_node) == 0:
                    for level_index in range(i+1, len(levels)-1):
                        for child_index in range(0, len(levels[level_index])):
                            child_node = levels[level_index][child_index]
                            if task_dict[child_node].get_num_input_buffers() > dag.in_degree(child_node):
                                flag = 0
                                dag.add_edge(task_node, child_node)
                                break





        # for i in range(0, len(levels)):
        #     for j in range(0, len(levels[i])):
        #         key = random.choice(global_map.keys())
        #         feat_dict = create_feat_dict(key, global_map)
        #         extime = 0.0
        #         if value_int(feat_dict['Class']) > 5:
        #             extime = ex_cpu[key]
        #         else:
        #             extime = ex_gpu[key]
        #         node_index = levels[i][j]
        #         task_dict[node_index] = SimTask(key, node_index, dag_id, feat_dict, extime)
        #         kernel_object = task_dict[node_index].Kernel_Object
        #         print j,
        #         print kernel_object.name
        #         print kernel_object.work_dimension
        #         print kernel_object.global_work_size
        #         print "Input Buffers: " + str(len(kernel_object.buffer_info['input']))
        #         print "Output Buffers: " + str(len(kernel_object.buffer_info['output']))


        # # Determine edges across levels and nodes based on density parameter and workdimension constraint
        # for i in range(1, len(levels)):
        #     for j in range(0, len(levels[i])):
        #         task_node = levels[i][j]
        #         num_parents = min(1 + int(density * len(levels[i - 1])), len(levels[i - 1]))
        #         existing_parents = []
        #         for k in range(0, num_parents):
        #             parent_index = np.random.randint(0, len(levels[i - 1]))
        #             if parent_index not in existing_parents:
        #                 parent_node = levels[i - 1][parent_index]
        #                 dag.add_edge(parent_node, task_node, weight=0.0, time=0.0)
        #                 existing_parents.append(parent_index)
        # print dag.edges()

        # Dump graph in file

        filename = "dag_" + str(n) + "_" + str(width) + "_" + str(regular) + "_" + str(density) + ".graph"
        f = open(filename, "w")
        f.write(str(n))
        f.write("\n")
        for i in range(0, n):
            f.write(task_dict[i].name)
            f.write("\n")
        for e in dag.edges():
            u, v = e
            edge = str(u) + " " + str(v)
            f.write(edge)
            f.write("\n")
        f.close()

        filename = graph_file
        f = open(filename, "w")
        f.write(str(n))
        f.write("\n")
        for i in range(0, n):
            f.write(task_dict[i].name)
            f.write("\n")
        for e in dag.edges():
            u, v = e
            edge = str(u) + " " + str(v)
            f.write(edge)
            f.write("\n")
        f.close()
        return SimTaskDAG(task_dict, dag, dag_id, ex_map)

    def visualize_graph(self, dag):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = dag.skeleton
        T = dag.G
        for node in S.nodes():
            Class = partition_class(dag.tasks[node])
            G.add_node(node, label=Class)
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v)
        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['fillcolor'] = color

        G.layout(prog='dot')
        file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)

    def dump_graph_ids(self, dag, file_name):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = dag.skeleton
        T = dag.G
        for node in S.nodes():

            G.add_node(node, label=str(node))
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v)


        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)


    def dump_graph_class(self, dag, file_name):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = dag.skeleton
        T = dag.G
        for node in S.nodes():
            Class = partition_class_absolute(dag.tasks[node])
            G.add_node(node, label=Class)
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v)
        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['fillcolor'] = color

        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)

    def dump_graph_DT_ECO(self, dag, file_name):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = dag.skeleton
        T = dag.G
        for node in S.nodes():
            Class = partition_class(dag.tasks[node])
            G.add_node(node, label=dag.tasks[node].ECO)
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v, label=S[u][v]['weight'])
        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['style'] = 'filled'
                    n.attr['fillcolor'] = color

        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)

    def dump_graph_start_finish_times(self, dag, file_name):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = dag.skeleton
        T = dag.G
        for node in S.nodes():
            name = dag.tasks[node].name
            rank = dag.tasks[node].rank
            
            device_type = dag.tasks[node].device_type
            device_id = str(dag.tasks[node].device_id)
            start_time = (dag.tasks[node].start_time)
            finish_time = (dag.tasks[node].finish_time)
            execution_time = finish_time - start_time
            label_string =  str(node) + " " + device_type + " " + str(start_time)
            # label_string = name + "\n" + "RANK: " + str(rank) +"\n" + str(start_time) + " " + str(
                # finish_time) + "\n" + device_type + device_id + ", " + str(execution_time)
            G.add_node(node, label=label_string)
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v, S[u][v]['time'])
        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['fillcolor'] = color

        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)

    def dump_graph_heatmap_exec(self, dag1_s, dag2_s, dag_filename):
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        dag1 = dag1_s.skeleton
        dag1 = dag2_s.skeleton
        T = dag2_s.G
        for node in dag1.nodes():
            name = dag1_s.tasks[node].name
            start_time1 = (dag1_s.tasks[node].start_time)
            finish_time1 = (dag1_s.tasks[node].finish_time)
            execution_time1 = finish_time1 - start_time1
            list_time_string = str(start_time1) + " " + str(finish_time1)
            start_time2 = (dag2_s.tasks[node].start_time)
            finish_time2 = (dag2_s.tasks[node].finish_time)
            execution_time2 = finish_time2 - start_time2
            contract_time_string = str(start_time2) + " " + str(finish_time2)
            speedup = execution_time1/execution_time2

            label_string = name + "\n" +str(speedup) + "\n" + list_time_string + "\n" + contract_time_string + "\n"
            time_diff = finish_time1 - finish_time2
            color = ''
            if time_diff < 0:
                color = 'red'
            else:
                color = 'green'

            # if speedup <=1 :
            #     color = 'red'
            # elif speedup <= 1.25:
            #     color = 'blue'
            # elif speedup <= 1.5:
            #     color = 'turquoise'
            # elif speedup <= 1.75:
            #     color = 'green'
            # else:
            #     color = 'darkgreen'
            G.add_node(node, label=label_string, fillcolor=color)

        for edge in dag1.edges():
            u, v = edge
            G.add_edge(u, v, dag1[u][v]['time'])



        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(dag_filename)

    def dump_graph_heatmap_info(self, dag1_s, dag2_s, dag_filename):
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        dag1 = dag1_s.skeleton
        dag1 = dag2_s.skeleton
        T = dag2_s.G
        for node in dag1.nodes():
            name = dag1_s.tasks[node].name
            start_time1 = (dag1_s.tasks[node].start_time)
            finish_time1 = (dag1_s.tasks[node].finish_time)
            execution_time1 = finish_time1 - start_time1
            list_time_string = str(start_time1) + " " + str(finish_time1)
            start_time2 = (dag2_s.tasks[node].start_time)
            finish_time2 = (dag2_s.tasks[node].finish_time)
            execution_time2 = finish_time2 - start_time2
            contract_time_string = str(start_time2) + " " + str(finish_time2)
            speedup = execution_time1/execution_time2
            rank = dag1_s.tasks[node].rank
            step1 = dag1_s.tasks[node].dispatch_step
            step2 = dag2_s.tasks[node].dispatch_step
            ex1 = finish_time1 - start_time1
            ex2 = finish_time2 - start_time2
            fsp = finish_time2/finish_time1
            label_string = name + "\nEXSP: " + str(speedup) + "\n" + "L: " + str(ex1) + " C: " + str(ex2) + "\nFSP: " + str(fsp) + "\n"+ "\nL: " + list_time_string + "\nC: " + contract_time_string + "\nRANK " + str(rank) + "\n" + "L: " + str(step1) + " C: " + str(step2)

            time_diff = finish_time1 - finish_time2

            color = ''
            if time_diff <= 0:
                color = 'red'
            else:
                color = 'green'

            # if speedup <=1 :
            #     color = 'red'
            # elif speedup <= 1.25:
            #     color = 'blue'
            # elif speedup <= 1.5:
            #     color = 'turquoise'
            # elif speedup <= 1.75:
            #     color = 'green'
            # else:
            #     color = 'darkgreen'
            G.add_node(node, label=label_string, penwidth=5.75, style='dashed', color=color, fillcolor='grey')

        for edge in dag1.edges():
            u, v = edge
            G.add_edge(u, v, dag1[u][v]['time'])


        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['style'] = 'filled'
                    n.attr['fillcolor'] = color




        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(dag_filename)












    def dump_graph_heatmap_holistic_info(self, dag1_s, dag2_s, dag_filename):
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        dag1 = dag1_s.skeleton
        dag1 = dag2_s.skeleton
        T = dag2_s.G
        for node in dag1.nodes():
            name = dag1_s.tasks[node].name
            device_type1 = dag1_s.tasks[node].device_type
            device_id1 = str(dag1_s.tasks[node].device_id)
            device_type2 = dag2_s.tasks[node].device_type
            device_id2 = str(dag2_s.tasks[node].device_id)
            start_time1 = (dag1_s.tasks[node].start_time)
            finish_time1 = (dag1_s.tasks[node].finish_time)
            execution_time1 = finish_time1 - start_time1
            list_time_string = str(start_time1) + " " + str(finish_time1)
            start_time2 = (dag2_s.tasks[node].start_time)
            finish_time2 = (dag2_s.tasks[node].finish_time)
            execution_time2 = finish_time2 - start_time2
            contract_time_string = str(start_time2) + " " + str(finish_time2)
            speedup = execution_time1/execution_time2
            rank = dag1_s.tasks[node].rank
            step1 = dag1_s.tasks[node].dispatch_step
            step2 = dag2_s.tasks[node].dispatch_step
            ex1 = finish_time1 - start_time1
            ex2 = finish_time2 - start_time2
            fsp = finish_time2/finish_time1
            # label_string = name + "\nEXSP: " + str(speedup) + "\n" + "L: " + str(ex1) + " C: " + str(ex2) + "\nFSP: " + str(fsp) + "\n"+ "\nL: " + list_time_string + "\nC: " + contract_time_string + "\nRANK " + str(rank) + "\n" + "L: " + str(step1) + " C: " + str(step2)
            label_string = name + "1.0\n" + device_type1 + device_id1 + " --> " + device_type2 + device_id2
            time_diff = finish_time1 - finish_time2

            color = ''
            if time_diff <= 0:
                color = 'red'
            else:
                color = 'green'

            # if speedup <=1 :
            #     color = 'red'
            # elif speedup <= 1.25:
            #     color = 'blue'
            # elif speedup <= 1.5:
            #     color = 'turquoise'
            # elif speedup <= 1.75:
            #     color = 'green'
            # else:
            #     color = 'darkgreen'
            G.add_node(node, label=label_string, penwidth=5.75, style='dashed', color=color, fillcolor='grey')

        for edge in dag1.edges():
            u, v = edge
            G.add_edge(u, v, dag1[u][v]['time'])


        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                holistic_speedup = 1.0
                ex1 = 0.0
                ex2 = 0.0
                for k in kid:
                    ex1_t= float(dag1_s.tasks[k].finish_time) - float(dag1_s.tasks[k].start_time)
                    ex2_t = float(dag2_s.tasks[k].finish_time) - float(dag2_s.tasks[k].start_time)
                    ex1 += ex1_t
                    ex2 += ex2_t
                holistic_speedup = ex1/ex2

                nodes = node.get_kernel_ids()
                for k in nodes:
                    name = dag2_s.tasks[k].name
                    n = G.get_node(k)
                    device_type1 = dag1_s.tasks[k].device_type
                    device_id1 = str(dag1_s.tasks[k].device_id)
                    device_type2 = dag2_s.tasks[k].device_type
                    device_id2 = str(dag2_s.tasks[k ].device_id)
                    n.attr['shape'] = 'square'
                    n.attr['style'] = 'filled'
                    n.attr['fillcolor'] = color
                    n.attr['label'] = name + " " + str(holistic_speedup) + "\n" + device_type1 + device_id1 + " --> " + device_type2 + device_id2





        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(dag_filename)

    def dump_graph_heatmap(self, dag1_s, dag2_s, dag_filename):
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        dag1 = dag1_s.skeleton
        dag1 = dag2_s.skeleton

        for node in dag1.nodes():
            name = dag1_s.tasks[node].name
            device_type1 = dag1_s.tasks[node].device_type
            device_id1 = str(dag1_s.tasks[node].device_id)
            device_type2 = dag2_s.tasks[node].device_type
            device_id2 = str(dag2_s.tasks[node].device_id)
            start_time1 = (dag1_s.tasks[node].start_time)
            finish_time1 = (dag1_s.tasks[node].finish_time)
            execution_time1 = finish_time1 - start_time1
            list_time_string = str(start_time1) + " " + str(finish_time1)
            start_time2 = (dag2_s.tasks[node].start_time)
            finish_time2 = (dag2_s.tasks[node].finish_time)
            execution_time2 = finish_time2 - start_time2
            contract_time_string = str(start_time2) + " " + str(finish_time2)
            speedup = execution_time1/execution_time2

            # label_string = name + "\n" +str(speedup) + "\n" + list_time_string + "\n" + contract_time_string + "\n"
            label_string = name + "\n" + str(speedup) + "\n" + device_type1 + device_id1 + "-->" + device_type2+device_id2 + "\n" + str(execution_time1) + "-->" + str(execution_time2)
            color = ''
            if speedup <=1 :
                color = 'red'
            elif speedup <= 1.5:
                color = 'blue'
            elif speedup <= 3:
                color = 'turquoise'
            elif speedup <= 6:
                color = 'green'
            else:
                color = 'darkgreen'
            G.add_node(node, label=label_string, penwidth=5.75, color=color, fillcolor='grey')

        for edge in dag1.edges():
            u, v = edge
            G.add_edge(u, v, dag1[u][v]['time'])

        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(dag_filename)

    def dump_graph_component_classes(self, dag, file_name):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = dag.skeleton
        T = dag.G
        for node in S.nodes():
            Class = partition_class(dag.tasks[node])
            G.add_node(node, label=str(dag.tasks[node].device_type) + str(dag.tasks[node].device_id) + " " + str(dag.tasks[node].execution_time))
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v, label=S[u][v]['time'])
        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = 'red'
                if partition_class_value(node.get_first_kernel().Class) == "gpu":
                    color = 'green'
                else:
                    color = 'blue'
                # color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['fillcolor'] = color

        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)

    def dump_graph_execution_times(self, dag, file_name):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = dag.skeleton
        T = dag.G
        for node in S.nodes():
            Class = partition_class(dag.tasks[node])
            G.add_node(node, label=dag.tasks[node].execution_time)
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v, label=S[u][v]['time'])
        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['fillcolor'] = color

        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)

    def dump_graph_names(self, dag, file_name):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = dag.skeleton
        T = dag.G
        for node in S.nodes():
            Class = partition_class(dag.tasks[node])
            G.add_node(node, label=dag.tasks[node].name)
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v)
        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                color = "#" + str(format(r(), '02x')) + str(format(r(), '02x')) + str(format(r(), '02x'))
                kid = node.get_kernel_ids()
                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['fillcolor'] = color

        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)
