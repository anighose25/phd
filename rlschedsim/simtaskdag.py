from simtask import *

class SimTaskDAG(object):
    """
    Class to handle all operations on Simulation DAG
    """

    def __init__(self, task_dict, dag, dag_id, ex_map, ml_classifier=None,name=None,deadline=False,job_id=-1):
        self.starting_time = 0.0
        self.finishing_time = 0.0
        self.ex_map = ex_map
        self.dag_id = dag_id
        self.tasks = task_dict
        self.task_components = dict()
        self.skeleton = dag
        self.finished_tasks = list()
        self.free_task_components = list()
        self.processing_tasks = list()
        self.ranks = dict()
        self.task_component_mappings = dict()
        self.task_component_id_map = dict()
        self.num_nodes = nx.number_of_nodes(dag)
        self.accuracy = 0.0
        self.job_id = job_id
        self.period=0.0
        self.rate=0.0
        self.rates = []
        self.release=0.0
        self.deadline=0.0
        self.tolerance=0.0
        self.wcet=0.0
        self.phase=0.0
        self.fused_kernel_timings={'cpu':{},'gpu': {}}
        self.adas=False
        # print "TOTAL_NODES ", self.skeleton.nodes()
        self.available_device_lookahead = {}
        self.currently_executing = {}
        # for edge in self.skeleton.edges():
        #     global buffer_times
        #     u, v = edge
        #     k_source = self.tasks[u].Kernel_Object
        #     k_target = self.tasks[v].Kernel_Object
        #     size_output_buf = 0
        #     time_output_buf = 0
        #     for buf in k_source.buffer_info['output']:
        #         buf_size = float(buf['size']) * get_sizeof(buf['type'])
        #         size_output_buf += buf_size
        #         time_output_buf += float(buffer_times[str(int(buf_size))])
        #     size_input_buf = 0
        #     time_input_buf = 0
        #     for buf in k_target.buffer_info['input']:
        #         buf_size = float(buf['size']) * get_sizeof(buf['type'])
        #         size_input_buf += buf_size
        #         time_input_buf += float(buffer_times[str(int(buf_size))])





        #     # if self.tasks[v].name == "bicg1_128" and self.tasks[u].name == "bicg1_128":
        #     #     print "GRAPH_EDGE_WEIGHT_TARGET ","bicg1_128", size_output_buf, size_input_buf


        #     self.skeleton[u][v]['weight'] = min(size_output_buf, size_input_buf)

        #     self.skeleton[u][v]['time'] = min(time_output_buf, time_input_buf)
            # print self.skeleton[u][v]['weight']
            # print self.skeleton[u][v]['time']

        # mapping = lambda s: SimTaskComponent(self.tasks[s])
        self.G = nx.relabel_nodes(self.skeleton, lambda s: SimTaskComponent(self.tasks[s]), copy=True)
        for task_component in self.G.nodes():
            for kid in task_component.get_kernel_ids():
                self.task_components[kid] = task_component
            if not self.get_task_component_parents(task_component):
                self.free_task_components.append(task_component)
        for task_component in self.G.nodes():
            self.task_component_id_map[task_component.id] = task_component

        self.name = None

        if name:
            self.name = name

        if deadline:
            self.calculate_wcet()
            self.get_available_rates()
            self.deadline=self.release+self.wcet
            self.adas=True

        if ml_classifier:
            ml_classifier.train_classifier("RandomForest")
            
            target_class_map = ml_classifier.predict_partition_classes()
            for node in self.skeleton.nodes():
                simtask = self.tasks[node]
                # print "Changing class of ", simtask.name, "from ", simtask.Class ,"to",
                if target_class_map[simtask.name] == "CPU":
                    simtask.Class = "ZERO"
                    self.task_components[node].Class = "ZERO"
                    # print simtask.Class, self.task_components[node].Class
                else:
                    simtask.Class = "TEN"
                    self.task_components[node].Class = "TEN"
                    # print simtask.Class,self.task_components[node].Class
            self.accuracy = ml_classifier.accuracy
        
    
    def modify_partition_classes(self,accuracy_file):
        # print "Modifying ",accuracy_file
        target_class_map = {}
        # file_name = open("GraphsNewAccuracy/accuracy_" + self.name + ".stats" ,'r').readlines()
        file_name = open(accuracy_file,'r')
        for line in file_name:
            k, p = line.strip("\n").split(":")
            target_class_map[k] = p

        for node in self.skeleton.nodes():
            simtask = self.tasks[node]
            # print "Changing class of ", simtask.name, "from ", simtask.Class ,"to",
            if target_class_map[simtask.name] == "CPU":
                simtask.Class = "ZERO"
                self.task_components[node].Class = "ZERO"
                # print simtask.Class, self.task_components[node].Class
            else:
                simtask.Class = "TEN"
                self.task_components[node].Class = "TEN"
                # print simtask.Class,self.task_components[node].Class
        
    
    
    def get_max_dt(self):
        dt = 0.0
        for edge in self.skeleton.edges():
            u,v = edge
            dt = max(dt, self.skeleton[u][v]['weight'])
        return dt

    def get_unique_tasks(self):
        unique_tasks = []
        for key in self.tasks.keys():
            unique_tasks.append(self.tasks[key].name) 
        return list(set(unique_tasks))

    
    def get_num_unique_tasks(self):
        return len(self.get_unique_tasks())
   
    @staticmethod

  

    def make_levels(G):
        # G = self.skeleton
        node_level = dict()
        levels = dict()
        for node in nx.algorithms.topological_sort(G):
            pred = G.predecessors(node)
            if not pred:
                node_level[node] = 0
            else:
                node_level[node] = min(map(lambda x: node_level[x], pred)) + 1
            try:
                levels[node_level[node]].append(node)
            except KeyError:
                levels[node_level[node]] = [node]
        return map(lambda x: x[1], sorted(levels.items(), key=lambda x: x[0]))

    def print_device_preferences(self):
        for t in self.tasks:
            print self.tasks[t].name, self.tasks[t].Class
    
    def print_information(self, rank_name):
        print self.skeleton.nodes()
        print self.skeleton.edges()
        for t in self.tasks:
            print str(t) + " " + self.tasks[t].name
        for node in self.G.nodes():
            print node.get_kernel_ids()
            # print node.rank_values[rank_name]
            # print "Ex Time: " + str(node.projected_ex_time)
        for edge in self.G.edges():
            u, v = edge
            print u.get_kernel_ids(),
            print "--->",
            print v.get_kernel_ids()

    def print_cc_information(self, cc):
        i = 1
        for c in cc:
            print "### Component " + str(i) + "###"
            for node in c.nodes():
                print node.get_kernel_ids(),
            print ""
            for edge in c.edges():
                u, v = edge
                print u.get_kernel_ids(),
                print "--->",
                print v.get_kernel_ids()
            i = i + 1

    def calculate_wcet(self):
        wcet= 0.0
        cpu_time = 0.0
        gpu_time = 0.0
        for node in self.skeleton.nodes():
            task=self.tasks[node].name
            ex_cpu,ex_gpu=self.ex_map
            cpu_time +=ex_cpu[task]
            gpu_time +=ex_gpu[task]

            if ex_cpu[task] > ex_gpu[task]:
                # print "CPU",task,ex_cpu[task]
                wcet += ex_cpu[task]
            else:
                # print "GPU",task,ex_gpu[task]
                wcet += ex_gpu[task]
        self.wcet=wcet
        # print cpu_time, gpu_time
        # print "CALCULATED WCET", wcet
        return cpu_time,gpu_time
        

    
    def get_available_rates(self):
        periods = []
        for r in [0.75,1,1.5]:
            periods.append(6*self.wcet*r)
        rates=[1/p for p in periods]
        self.rates=rates

    def assign_device_lookahead(self, dev_lookahead):
        self.available_device_lookahead = dev_lookahead

    def assign_currently_executing(self, current_execution_status):
        self.currently_executing = current_execution_status

    def print_ranks(self):
        for i in self.skeleton.nodes():
            print  str(i) + " --> " + str(self.ranks[i])
    
    def print_rank_info(self):
        for i in self.skeleton.nodes():
            print "DAG", self.dag_id, "WCET", self.wcet, "DAG RELEASE", self.release, "DAG DEADLINE", self.deadline, "TASK ", str(i) + " LOCAL DEADLINE --> " + str(self.tasks[i].rank)

    def print_task_info(self):
        for i in self.skeleton.nodes():
            print "DAG", self.dag_id, "TASK", self.tasks[i].id, self.tasks[i].start_time, self.tasks[i].finish_time

    def get_source_nodes(self):
        source_nodes = []
        for i in self.skeleton.nodes():
            if (self.skeleton.in_degree(i) == 0):
                source_nodes.append(i)
        return source_nodes

    def get_sink_nodes(self):
        sink_nodes = []
        for i in self.skeleton.nodes():
            if (self.skeleton.out_degree(i) == 0):
                sink_nodes.append(i)
        return sink_nodes

    def add_dummy_node_source(self):
        source_nodes = self.get_source_nodes()
        n = self.num_nodes
        feat_dict = {'Float16': 0.0, 'Float32': 0.0, 'Int16': 0.0, 'Int32': 0.0, 'DataTransfer': 0.0, 'Barrier': 0.0,
                     'ComputePerDataTransfer': 0.0, 'TotalBranches': 0.0, 'TotalMemory': 0.0, 'NumberOfWorkItems': 1.0,
                     'ComputeToMemoryRatio': 0.0, 'Class': 0.0}
        self.tasks[n] = SimTask("dummy", n, self.dag_id, feat_dict, 0)
        self.skeleton.add_node(n)
        for s in source_nodes:
            self.skeleton.add_edge(n, s)

    def remove_dummy_node_source(self):
        n = self.num_nodes
        self.skeleton.remove_node(n)
        del self.tasks[n]

    def remove_dummy_node_exit(self):
        n = self.num_nodes + 1
        self.skeleton.remove_node(n)
        del self.tasks[n]

    def add_dummy_node_exit(self):
        sink_nodes = self.get_sink_nodes()
        n = self.num_nodes + 1
        feat_dict = {'Float16': 0.0, 'Float32': 0.0, 'Int16': 0.0, 'Int32': 0.0, 'DataTransfer': 0.0, 'Barrier': 0.0,
                     'ComputePerDataTransfer': 0.0, 'TotalBranches': 0.0, 'TotalMemory': 0.0, 'NumberOfWorkItems': 1.0,
                     'ComputeToMemoryRatio': 0.0, 'Class': 0.0}
        self.tasks[n] = SimTask("dummy", n, self.dag_id, feat_dict, 0)
        self.skeleton.add_node(n)
        for s in sink_nodes:
            self.skeleton.add_edge(s, n)

    def is_processed(self):
        # print "Finished " + str(len(self.finished_tasks))
        # print "Number of nodes: " + str(len(self.skeleton.nodes()))
                
        rev_top_list = list(nx.topological_sort(self.skeleton))
        rev_top_list.reverse()
        finish_timestamp = 0.0
        if len(self.finished_tasks) == len(self.skeleton.nodes()):
            for node in rev_top_list:
                if self.skeleton.out_degree(node) == 0:
                    finish_timestamp = max(finish_timestamp,self.tasks[node].finish_time)
            self.finishing_time = finish_timestamp    
            return True
        
        return False
        # return len(self.finished_tasks) == len(self.skeleton.nodes())

    def reset_node_ranks(self, rank_name):
        # print "Resetting Node Ranks"
        for node in self.skeleton.nodes():
            # self.tasks[node].rank_values[rank_name] = 0.0
            self.tasks[node].rank = 0.0
            # print "RANK After Reset", self.tasks[node].rank_values[rank_name]
            # import time
            # time.sleep(2)
        for task_component in self.G.nodes():
            if task_component.is_supertask():
                raise Exception("Cannot reset rank of super task")
            else:
                # task_component.rank_values[rank_name] = 0.0
                task_component.rank_name = 0.0


    def update_rank_values(self, rank_name):
        for node in self.skeleton.nodes():
            self.tasks[node].rank_values[rank_name] = self.tasks[node].rank

    def task_component_data_transfer_size(self, task_component):
        tdt = 0
        k_set = task_component.get_kernels()
        kernel_ids = map(lambda k: k.id, list(k_set))
        subgraph = self.get_skeleton_subgraph(kernel_ids)
        for r, s in subgraph.edges():
            tdt += subgraph[r][s]['weight']
        return tdt

    def task_component_data_transfer_time(self, task_component):
        tdt = 0
        k_set = task_component.get_kernels()
        kernel_ids = map(lambda k: k.id, list(k_set))
        subgraph = self.get_skeleton_subgraph(kernel_ids)
        for r, s in subgraph.edges():
            tdt += subgraph[r][s]['time']
        return tdt

#Implementation only for linear DAGs

    def get_allowable_fused_timing_and_depth(self,task_component,dev,threshold):
        task_id = task_component.get_first_kernel().id
        remaining_depth = len(self.tasks.keys()) - task_id 
        ex_time = float("inf")
        depth = 0
        task_ids = [task_id]
        for d in range(1,remaining_depth):
            tid = task_id + d
            if self.task_components[tid].get_first_kernel().rank < threshold:
                task_ids.append(tid)
                depth +=1
        ex_time = 0.0
       
        if depth == 0:
            ex_time = task_component.get_component_time(self.ex_map,dev)
        else:
            ex_time = self.fused_kernel_timings[dev][tuple(sorted(tuple(task_ids)))]
        
        return ex_time,depth          

#Implementation only for linear DAGs

    def get_min_fused_timing_and_depth(self,task_component,dev):
        task_id = task_component.get_first_kernel().id
        remaining_depth = len(self.tasks.keys()) - task_id 
        ex_time = float("inf")
        depth = 0
        device = ""
        current_speedup = 1.0
        max_speedup = 0.0
        for d in range(0,remaining_depth):
            task_ids = []
            
            for tid in range(task_id,task_id + d+1):
                task_ids.append(tid)
            
            if len(task_ids) == 1:
                current_depth = 0
                current_speedup = 1.0
            else:
                fused_time = self.fused_kernel_timings[dev][tuple(sorted(tuple(task_ids)))]
                cumulative_time = 0.0
                for t in task_ids:
                    cumulative_time += self.task_components[t].get_component_time(self.ex_map,dev)
                current_depth = d
                current_speedup = cumulative_time/fused_time
            print current_depth, current_speedup
            if current_speedup > max_speedup:
                max_speedup = current_speedup
                depth = current_depth    
                device = dev
                if depth == 0:
                    ex_time = task_component.get_component_time(self.ex_map,dev)
                else:
                    ex_time = self.fused_kernel_timings[dev][tuple(sorted(tuple(task_ids)))]
        return ex_time,depth                

    def reduce_adas_execution_time(self,task,task_component):
        ex_cpu,ex_gpu=self.ex_map
        if task_component.is_supertask():
            
            
            task_ids=task_component.get_kernel_ids()
            
            if partition_class_value(task_component.Class) is "gpu":
                t_gpu=task_component.get_component_time(self.ex_map,"gpu")
                t_fused=self.fused_kernel_timings['gpu'][tuple(sorted(tuple(task_ids)))]
                factor=ex_gpu[task.name]/float(t_gpu)
                diff=(t_gpu-t_fused)*factor
                # print t_gpu,t_fused,factor,diff
                task.projected_ex_time=ex_gpu[task.name]-diff
                task.execution_time = task.projected_ex_time
            else:
                t_cpu=task_component.get_component_time(self.ex_map,"cpu")
                t_fused=self.fused_kernel_timings['cpu'][tuple(sorted(tuple(task_ids)))]
                factor=ex_cpu[task.name]/float(t_cpu)
                diff=(t_cpu-t_fused)*factor
                # print t_gpu,t_fused,factor,diff
                task.projected_ex_time=ex_cpu[task.name]-diff
                task.execution_time = task.projected_ex_time


            # print "REDUCTION",task.id, task.dag_id, task.projected_ex_time
        else:
            if partition_class_value(task_component.Class) is "cpu":
                task.projected_ex_time=ex_cpu[task.name]
                task.execution_time = task.projected_ex_time
            else:
                task.projected_ex_time=ex_gpu[task.name]
                task.execution_time = task.projected_ex_time
            # print "NO REDUCTION", task.id, task.dag_id, task.projected_ex_time

    def reduce_execution_time(self, task, task_component):

        ex_cpu, ex_gpu = self.ex_map
        # print "EX_TIME_"+str(task.id),ex_cpu[task.name],ex_gpu[task.name]
        predecessors = self.skeleton.predecessors(task.id)
        successors = self.skeleton.successors(task.id)
        ex_time = 0.0
        if partition_class_value(task_component.Class) is "gpu":
            ex_time = ex_gpu[task.name]
        else:
            ex_time = ex_cpu[task.name]

        '''
        if partition_class_value(task_component.Class) is "gpu":

            if value_int(task.Class) > 5:
                ex_time = task.execution_time
            else:
                ex_time = ex_gpu[task.name]
            # ex_time = ex_gpu[task.name]

        else:
            if value_int(task.Class) < 5:
                ex_time = task.execution_time
            else:
                ex_time = ex_cpu[task.name]
        '''
            # ex_time = ex_gpu[task.name]
        dt_time = 0.0
        task_ids = task_component.get_kernel_ids()
        # print "LOOKAHEAD_ESTIMATE_GPU_" +task.name , task.id, task_ids, predecessors, successors
        dt_size = 0.0
        pred_dt_size = 0.0
        succ_dt_size = 0.0
        pred_dt_time = 0.0
        succ_dt_time = 0.0
        for pred in predecessors:
            if pred in task_ids:
                pred_dt_time += self.skeleton[pred][task.id]['time']
                pred_dt_size += self.skeleton[pred][task.id]['weight']
        # print "PRED_LOOKAHEAD_ESTIMATE_GPU_" + task.name, task.id, pred_dt_size
        k_source = self.tasks[task.id].Kernel_Object
        max_dt_size = 0.0
        max_dt_time = 0.0
        for buf in k_source.buffer_info['output']:
            buf_size = float(buf['size']) * get_sizeof(buf['type'])
            max_dt_size += buf_size
            max_dt_time += float(buffer_times[str(int(buf_size))])


        for succ in successors:
            if succ in task_ids:
                succ_dt_time += self.skeleton[task.id][succ]['time']
                succ_dt_size += self.skeleton[task.id][succ]['weight']
                if succ_dt_size >= max_dt_size:
                    succ_dt_size = max_dt_size
                    succ_dt_time = max_dt_time
                    break


        # print "SUCC_LOOKAHEAD_ESTIMATE_GPU_" + task.name, task.id, succ_dt_size
        dt_size = pred_dt_size + succ_dt_size
        dt_time = pred_dt_time + succ_dt_time
        # print "execution time",
        # print task.execution_time
        # print "data transfer time",
        # print dt_time
        # print "Reducing execution time of ",
        # print task.name,
        # print " to ",

        if partition_class_value(task_component.Class) is "gpu":
            task.projected_ex_time = ex_time - dt_time
        else:
            task.projected_ex_time = ex_time


        if partition_class_value(task_component.Class) is "gpu":
            task.estimated_ex_time = task.ECO/GPU_FLOPS + task.DT/BW - dt_size/BW
            # print "EXEC_LOOKAHEAD_ESTIMATE_GPU_" + task.name, task.estimated_ex_time, task.projected_ex_time
            # print "LOOKAHEAD_ESTIMATE_GPU_"+ task.name," Breakup ", task.ECO/GPU_FLOPS, task.DT ,dt_size
        else:
            task.estimated_ex_time = task.ECO/CPU_FLOPS
            # print "EXEC_LOOKAHEAD_ESTIMATE_CPU", task.name,task.estimated_ex_time, task.projected_ex_time


        # print task.projected_ex_time

    def calculate_projected_execution_time(self, task_component):
        global regression_model
        # global buffer_times
        k_set = task_component.get_kernels()
        total_execution_time = sum(map(lambda k: k.execution_time, list(k_set)))
        # print "total execution time " + str(total_execution_time)
        # total_data_transfer = self.task_component_data_transfer_size(task_component)
        # print "total data transfer " + str(total_data_transfer)
        # total_transfer_execution_time = regression_model.predict(total_data_transfer)[0][0]
        total_transfer_execution_time = self.task_component_data_transfer_time(task_component)
        # print "total_transfer_execution_time " + str(total_transfer_execution_time)
        task_component.projected_ex_time = total_execution_time - 2 * total_transfer_execution_time

    # Deprecated

    def task_data_transfer_size(self, task_r, task_s):
        tdt = 0
        k_r = task_r.get_kernels()
        k_s = task_s.get_kernels()
        kernel_ids = map(lambda k: k.id, list(k_r) + list(k_s))
        subgraph = self.get_skeleton_subgraph(kernel_ids)
        for r, s in subgraph.edges():
            tdt += subgraph[r][s]['weight']
        return tdt

    def get_task_parents(self, task):
        task_id = task.id
        t_parents = []
        for t in self.skeleton.predecessors(task_id):
            t_parents.append(self.tasks[t])
        return t_parents

    def get_task_children(self, task):
        task_id = task.id
        t_children = []
        for t in self.skeleton.successors(task_id):
            t_children.append(self.tasks[t])
        return t_children

    def get_remaining_tasks(self):
        for id in self.skeleton.nodes():
            if self.tasks[id].is_finished == False:
                print id,
        print "."



    def are_parents_finished(self, task):
        t_parents = self.get_task_parents(task)
        flag = True
        # print "Checking parents of " + str(task.id) + " of dag " + str(self.dag_id) + ": ",
        for t in t_parents:
            # print t.id,
            # print " ",
            # print t.is_finished,
            # print " ",
            flag = flag and t.is_finished
        return flag

    def are_task_component_parents_finished(self, task_component):
        t_parents = self.get_task_component_parents(task_component)
        flag = True
        for t in t_parents:
            flag = flag and t.is_finished
        return flag

    def get_kernel_parent_ids(self, kid):
        """
        Should return a list of kernel ids that are predecessors to given kernel.
        """
        return self.skeleton.predecessors(kid)

    def get_kernel_children_ids(self, kid):
        """
        Should return a list of kernel ids that are successors to given kernel.
        """
        return self.skeleton.successors(kid)

    def get_skeleton_subgraph(self, kernel_ids):
        return self.skeleton.subgraph(kernel_ids)

    def get_task_component_parents(self, task_component):
        return self.G.predecessors(task_component)

    def get_task_component_children(self, task_component):

        # print [(k.get_kernel_ids(),k) for k in self.G.nodes()]
        # print task_component.get_kernel_ids(), task_component
        return self.G.successors(task_component)

    def update_dependencies(self, task_component):
        """
        Updates task dependencies. Call this whenever a task is modified. Adds or remove edges to task dag based on
        skeleton kernel dag for the given task.
        :param task:
        :return:
        """
        p, c = set(self.get_task_component_parents(task_component)), set(
            self.get_task_component_children(task_component))
        pt, ct = set(), set()
        for kid in task_component.get_kernel_ids():
            for pkid in self.get_kernel_parent_ids(kid):
                pt.add(self.task_components[pkid])
            for ckid in self.get_kernel_children_ids(kid):
                ct.add(self.task_components[ckid])
        pt -= set([task_component])
        ct -= set([task_component])
        for t in pt - p:
            self.G.add_edge(t, task_component)
        for t in ct - c:
            self.G.add_edge(task_component, t)
        for t in p - pt:
            self.G.remove_edge(t, task_component)
        for t in c - ct:
            self.G.remove_edge(task_component, t)

    def merge_task_components(self, t1, t2):
        dependencies = set().union(*[set(self.get_kernel_parent_ids(kid)) for kid in t2.get_kernel_ids()])
        # print set(t1.get_kernel_ids()), dependencies
        if set(t1.get_kernel_ids()) >= dependencies:
            t1.add_kernels_from_task(t2)
        else:
            raise Exception('Some dependent kernels are not part of this task.')
        for kid in t2.get_kernel_ids():
            self.task_components[kid] = t1
        self.update_dependencies(t1)
        self.G.remove_node(t2)
        self.task_component_id_map.pop(t2.id)

    def merge_independent_task_components(self, t1, t2):
        # print t2.get_kernel_ids(), t2
        # for node in self.G.nodes():
        #     print node.get_kernel_ids(),node
        t1.add_kernels_from_task(t2)
        for kid in t2.get_kernel_ids():
            self.task_components[kid] = t1
        self.update_dependencies(t1)
        # print "Removing ",t2.get_kernel_ids(), t2
        self.G.remove_node(t2)
        self.task_component_id_map.pop(t2.id)

    def split_kernel_from_task_component(self, kernel, task_component):
        """
        Remove the given kernel from the given task and create a new task from that kernel, update task
        dependencies accordingly. Returns the newly created task.
        """
        task_component.remove_kernel(kernel)
        t = SimTaskComponent(kernel)
        self.G.add_node(t)
        self.task_components[kernel.id] = t
        self.update_dependencies(task_component)
        self.update_dependencies(t)
        return t

    def merge_task_list(self, t):
        # print "Tasks to be merged: ",
        # for task_component in t:
        #     print task_component.get_kernel_names(),
        t1 = t[0]
        for t2 in t[1:]:
            # print "T1 ",
            # print t1.get_kernel_names()
            # print "T2 ",
            # print t2.get_kernel_names()
            self.merge_independent_task_components(t1, t2)
        self.calculate_projected_execution_time(t1)
        return t1

    def get_connected_components_two_level(self, tasks):
        successors = set().union(*[set(self.get_task_component_children(task)) for task in tasks])
        subgraph_nodes = tasks + list(successors)
        print "Component Ids ",
        for t in subgraph_nodes:
            print t.get_kernel_ids(),
        print "\n"
        g = self.G.subgraph(subgraph_nodes)
        # print g.nodes()
        gid = nx.relabel_nodes(g, lambda s: s.id, copy=True)
        # cc = list(nx.connected_component_subgraphs(g))
        cc = []
        for component in nx.connected_components(gid.to_undirected()):
            cc.append(self.G.subgraph([self.task_component_id_map[c] for c in list(component)]))
        return cc

    def get_connected_components_k_level(self, tasks, k):
        subgraph_nodes = tasks
        for i in range(0, k):
            successors = set().union(*[set(self.get_task_component_children(task)) for task in tasks])
            subgraph_nodes = subgraph_nodes + list(successors)
            del tasks[:]
            tasks = list(successors)
        g = self.G.subgraph(subgraph_nodes)
        print g.nodes()
        gid = nx.relabel_nodes(g, lambda s: s.id, copy=True)
        # cc = list(nx.connected_component_subgraphs(g))
        cc = []
        for component in nx.connected_components(gid.to_undirected()):
            cc.append(self.G.subgraph([self.task_component_id_map[c] for c in list(component)]))
        return cc


    def get_next_parent(self, task, first_parent):
        parents = self.get_task_component_parents(task)
        # print "TC_GEN: Parents ",
        # print parents
        sorted(parents, key=lambda x: x.get_first_kernel().rank)
        for parent in parents:
            if parent.get_first_kernel().in_frontier and not parent.is_supertask() and parent is not first_parent:
                return parent



    def construct_component_dl(self, task, component, width, depth):

        if task is None:
            return

        if not task.is_supertask() and task not in component:
            print "TC_GEN : adding task of dag " + str(task.dag_id),
            print task.get_kernel_names(), task.get_kernel_ids()
            component.append(task)

        else:
            return

        for succ in self.get_task_component_children(task):

            t = succ.get_first_kernel()
            print "TC_GEN : investigate parents of successor " + str(t.id)
            if width > 0:
                if len(self.get_task_parents(t)) > 1:
                    parent = self.get_next_parent(succ, task)
                    print parent

                    self.construct_component_dl(parent, component, width-1, depth)
            if depth > 0:
                self.construct_component_dl(succ, component, width, depth-1)

    def construct_component_peek_var(self, task, component, width, depth):
        p_values = []
        component.append(task)
        blacklist_parents = []
        p_values.append(partition_class_absolute(task))
        L1 = [task]
        L2 = []
        level = 0
        d = depth
        while d > 0:

            T = []

            if level % 2 == 0:
                T = L1
            else:
                T = L2
            # print "Before Loop", [k.get_kernel_ids() for k in L1]
            for t in T:
                # print "Successors of ", t.get_kernel_ids()
                # print self.get_task_component_children(t)

                H = []
                mean = np.mean(p_values)
                for succ in self.get_task_component_children(t):
                    heapq.heappush(H, (abs(partition_class_absolute(succ) - mean), succ))
                # print H
                while (len(H) > 0):
                    priority, succ = heapq.heappop(H)

                    flag = 1
                    for parent in self.get_task_component_parents(succ):
                        if parent is not t and parent not in component:
                            if parent.get_first_kernel().in_frontier and not parent.is_supertask() and parent not in blacklist_parents:
                                # component.append(parent)
                                # print "TC_GEN : parent is ok to add ", parent.get_kernel_ids(), parent.dag_id
                                # print parent.get_kernel_names(), parent.get_kernel_ids()
                                pass
                            else:
                                flag = 0
                                # print "TC_GEN : parent is not ok to add (potential deadlock) successor shouldn't be added (parent,successor)", parent.get_kernel_ids(), succ.get_kernel_ids(), parent.dag_id

                    if flag:
                        for parent in self.get_task_component_parents(succ):
                            if parent not in component:
                                p_values.append(partition_class_absolute(parent))
                        p_values.append(partition_class_absolute(succ))
                        if np.std(p_values) >= 1.5:
                            blacklist_parents.append(succ)
                            flag = 0

                    if flag:
                        for parent in self.get_task_component_parents(succ):
                            if parent not in component:
                                component.append(parent)

                        if level % 2 == 0:
                            if succ not in L2:
                                L2.append(succ)
                        else:
                            if succ not in L1:
                                L1.append(succ)
                        if succ not in component:
                            component.append(succ)

            if level % 2 == 0:
                del L1[:]
                # print "After Loop", [k.get_kernel_ids() for k in L2]
            else:
                del L2[:]
                # print "After Loop", [k.get_kernel_ids() for k in L1]
            # print "Loop Component ",[k.get_kernel_ids() for k in component]
            level += 1
            d -= 1

    def construct_component_peek(self, task, component, width, depth):
        p_values = []
        component.append(task)
        L1 = [task]
        L2 = []
        level = 0
        d = depth
        while d > 0:

            T = []

            if level%2 == 0:
                T = L1
            else:
                T = L2
            print "Before Loop", [k.get_kernel_ids() for k in L1]
            for t in T:
                print "Successors of ", t.get_kernel_ids()
                print self.get_task_component_children(t)
                for succ in self.get_task_component_children(t):
                    flag = 1
                    for parent in self.get_task_component_parents(succ):
                        if parent is not t and parent not in component:
                            if parent.get_first_kernel().in_frontier and not parent.is_supertask():
                                # component.append(parent)
                                # print "TC_GEN : parent is ok to add ", parent.get_kernel_ids(), parent.dag_id
                                # print parent.get_kernel_names(), parent.get_kernel_ids()
                                pass
                            else:
                                flag = 0
                                # print "TC_GEN : parent is not ok to add (potential deadlock) successor shouldn't be added (parent,successor)", parent.get_kernel_ids(), succ.get_kernel_ids(), parent.dag_id

                    if flag:
                        for parent in self.get_task_component_parents(succ):
                            if parent not in component:
                                p_values.append(partition_class_absolute(parent))
                        p_values.append(partition_class_absolute(succ))
                        if np.std(p_values) >=1.5:
                            return

                        for parent in self.get_task_component_parents(succ):
                            if parent not in component:
                                component.append(parent)

                        if level%2 == 0:
                            if succ not in L2:
                                L2.append(succ)
                        else:
                            if succ not in L1:
                                L1.append(succ)
                        if succ not in component:
                            component.append(succ)

            if level%2 == 0:
                del L1[:]
                print "After Loop", [k.get_kernel_ids() for k in L2]
            else:
                del L2[:]
                print "After Loop", [k.get_kernel_ids() for k in L1]
            print "Loop Component ",[k.get_kernel_ids() for k in component]
            level +=1
            d -= 1




    def construct_component(self, task, component, width, depth):

        if task is None:
            return

        if not task.is_supertask() and task not in component:
            # print "TC_GEN : adding task of dag " + str(task.dag_id),
            # print task.get_kernel_names(), task.get_kernel_ids()
            p = []
            '''
            for t in component:
                p.append(partition_class_absolute(t))
            print "Dispersion: ", p, np.std(p)
            '''
            component.append(task)

        else:
            return
        # print "Successors of ", task.get_kernel_ids(), self.get_task_component_children(task)
        for succ in self.get_task_component_children(task):
            flag = 1
            for parent in self.get_task_component_parents(succ):
                if parent is not task and parent not in component:
                    if parent.get_first_kernel().in_frontier and not parent.is_supertask():
                        # component.append(parent)
                        # print "TC_GEN : parent is ok to add ", parent.get_kernel_ids(), parent.dag_id
                        # print parent.get_kernel_names(), parent.get_kernel_ids()
                        pass
                    else:
                        flag = 0
                        # print "TC_GEN : parent is not ok to add (potential deadlock) successor shouldn't be added (parent,successor)", parent.get_kernel_ids(), succ.get_kernel_ids(), parent.dag_id

            if flag:
                for parent in self.get_task_component_parents(succ):
                    if parent not in component:
                        component.append(parent)
                        # print "TC_GEN : adding parent task of dag ",  parent.dag_id, parent.get_kernel_ids(), parent.get_kernel_names()
                    # print parent.get_kernel_names(), parent.get_kernel_ids()
            if depth > 0 and flag:
                self.construct_component(succ, component, width, depth-1)

    def construct_component_var(self, task, component, dt_max, percent, depth):

        if task is None:
            return

        if not task.is_supertask() and task not in component:
            # print "VAR_TC_GEN : adding task of dag " + str(task.dag_id),
            # print task.get_kernel_names(), task.get_kernel_ids()
            component.append(task)

        else:
            return
        depth_flag = 0
        for succ in self.get_task_component_children(task):
            flag = 1
            s = succ.get_kernel_ids()[0]

            for succ_prime in self.get_task_component_children(succ):
                s_prime = succ_prime.get_kernel_ids()[0]
                dt1 = self.skeleton[s][s_prime]['weight']
                # print "VAR_TC_GEN (s,s') ", dt_max, dt1, self.tasks[s].name, self.tasks[s].id, self.tasks[s_prime].name, self.tasks[s_prime].id
                if dt1 >= percent * dt_max:
                    for succ_prime_prime in self.get_task_component_children(succ_prime):
                        s_prime_prime = succ_prime_prime.get_kernel_ids()[0]
                        dt2 = self.skeleton[s_prime][s_prime_prime]['weight']
                        if dt2 >= dt1:
                            #flag = 0
                            depth += 1
                            depth_flag = 1
                            # print "VAR_TC_GEN (s,s'') ", dt_max, dt1, dt2, self.tasks[s_prime].name, self.tasks[s_prime].id, self.tasks[s_prime_prime].name, self.tasks[s_prime_prime].id
                            break

            for parent in self.get_task_component_parents(succ):
                if parent is not task and parent not in component:
                    if parent.get_first_kernel().in_frontier and not parent.is_supertask():
                        # component.append(parent)
                        # print "TC_GEN : parent is ok to add ", parent.get_kernel_ids(), parent.dag_id
                        # print parent.get_kernel_names(), parent.get_kernel_ids()
                        pass
                    else:
                        flag = 0
                        # print "TC_GEN : parent is not ok to add (potential deadlock) successor shouldn't be added (parent,successor)", parent.get_kernel_ids(), succ.get_kernel_ids(), parent.dag_id

            if flag:
                for parent in self.get_task_component_parents(succ):
                    if parent not in component:
                        component.append(parent)
                    # print "TC_GEN : adding parent task of dag ", parent.get_kernel_ids(), parent.dag_id
                    # print parent.get_kernel_names(), parent.get_kernel_ids()
            if depth > 0 and flag:
                self.construct_component_var(succ, component, dt_max, percent, depth - 1)
        if depth_flag:
            depth -= 1

    def construct_component_set(self, task, component, width, depth):
        print "adding task of dag " + str(task.dag_id),
        print task.get_kernel_names()
        if task not in component:
            component.append(task)
        else:
            return

        for succ in self.get_task_component_children(task):
            if succ not in component:
                t = succ.get_first_kernel()
                if len(self.get_task_parents(t)) > 1:
                    parents = self.get_task_component_parents(succ)
                    immediate_parent = next((p for p in parents if p.get_first_kernel().in_frontier == True and p != task), None)
                    if immediate_parent is not None:
                        print immediate_parent.get_kernel_names()
                    if width > 0 and immediate_parent is not None:
                        print "adding width based parent"
                        self.construct_component(immediate_parent, component, width - 1, depth)

                if depth > 0:
                    print "adding depth based child"

                    self.construct_component(succ, component, width, depth - 1)


    def construct_component_gpu(self, task, component, width, depth):
        print "adding task of dag " + str(task.dag_id),
        print task.get_kernel_names()
        if task in component:
            return

        if task.get_first_kernel().Class == "gpu":
            component.append(task)
            task.contract = False
        else:
            task.contract = False

        for succ in self.get_task_component_children(task):
            if succ not in component:
                t = succ.get_first_kernel()
                if len(self.get_task_parents(t)) > 1:
                    parents = self.get_task_component_parents(succ)
                    immediate_parent = next((p for p in parents if p.get_first_kernel().in_frontier == True), None)
                    if width > 0 and immediate_parent is not None:
                        self.construct_component(immediate_parent, component, width - 1, depth)

                if depth > 0:
                    self.construct_component(succ, component, width, depth - 1)




    def kernel_fusion(self,task,depth,frontier):
        subgraph_nodes = []
        if depth != -1:
            self.construct_component(task, subgraph_nodes, 0, depth)
        
        if len(subgraph_nodes) > 1:
            for node in subgraph_nodes:
                if node.get_first_kernel().in_frontier and node is not task:
                    # print "Removing ", node.get_first_kernel().id, node.get_first_kernel().name
                    node.get_first_kernel().in_frontier=False
                    frontier.queue.remove(node)
            # subgraph_nodes[0].Class="TEN"
            return self.merge_task_list(subgraph_nodes)
        else:
            task.get_first_kernel().in_frontier = False
            return task


    

    def initialize_task_component_ranks(self, rank_name):
        for task_component in self.G.nodes():
            if task_component.is_supertask():
                raise Exception("Cannot initialize rank of super task")
            else:
                task_component.rank_values[rank_name] = list(task_component.get_kernels())[0].rank_values[rank_name]
                task_component.rank_name = rank_name

    def get_component_device_affinity(self, task_components):
        cpu = 0
        gpu = 0
        for task_component in task_components:
            classes = task_component.get_kernel_classes()
            for Class in classes:
                if value_int(Class) < 5:
                    cpu += 1
                else:
                    gpu += 1
        return cpu, gpu
