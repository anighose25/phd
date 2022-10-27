import networkx as nx

class Ranker(object):
    """
    Class to compute different rank functions for a list of Simulation DAGs
    """

    def __init__(self, dags, ex_map):
        self.dags = dags
        self.ex_map = ex_map
        # print "Inside Ranker Constructor", self.dags
        # for dag in self.dags:
        #     print dag.dag_id, dag.job_id
        # import time
        # time.sleep(1)


    def oct(self, dag, nCPU=6, mGPU=6):
        rev_top_list = list(nx.topological_sort(dag.skeleton))
        rev_top_list.reverse()
        proc = []
        octable = {}
        for i in range(0, nCPU):
            p = "cpu"+str(i)
            proc.append(p)
            octable[p]={}
        for i in range(0, mGPU):
            p = "gpu"+str(i)
            proc.append(p)
            octable[p]={}
        for p in proc:
            for t in range(0, len(dag.tasks)):
                octable[p][t] = 0.0

        for p in proc:
            for t in rev_top_list:
                if dag.skeleton.out_degree(t) == 0:
                    octable[p][t] = 0.0
                else:
                    tmax = 0.0
                    for succ in dag.skeleton.successors(t):
                        tmin = float("inf")
                        for succ_proc in proc:
                            if proc == succ_proc:
                                if "cpu" in succ_proc:
                                    tmin = min(tmin, octable[succ_proc][succ] + dag.tasks[succ].ECO / CPU_FLOPS)
                                else:
                                    tmin = min(tmin, octable[succ_proc][succ] + dag.tasks[succ].ECO / GPU_FLOPS)
                            else:
                                if "cpu" in succ_proc:
                                    if "cpu" in proc:
                                        tmin = min(tmin, octable[succ_proc][succ] + dag.tasks[succ].ECO / CPU_FLOPS)
                                    else:
                                        tmin = min(tmin, octable[succ_proc][succ] + dag.tasks[succ].ECO / CPU_FLOPS + dag.tasks[t].DT / BW)
                                else:
                                    if "cpu" in proc:
                                        tmin = min(tmin, octable[succ_proc][succ] + dag.tasks[succ].ECO / GPU_FLOPS + dag.tasks[succ].DT /BW)
                                    else:
                                        tmin = min(tmin, octable[succ_proc][succ] + dag.tasks[succ].ECO / GPU_FLOPS + dag.tasks[t].DT/BW + dag.tasks[succ].DT/BW)

                        tmax = max(tmax, tmin)
                    octable[p][t] = tmax



            for task in dag.skeleton.nodes():
                trank = 0.0
                for p in proc:
                    trank += octable[p][t]

                dag.tasks[task].rank = trank/len(proc)










    def tlevel(self,dag):
        top_list = list(nx.topological_sort(dag.skeleton))
        for node in top_list:
            if dag.skeleton.in_degree(node) == 0:
                task = dag.tasks[node]
                if value_int(task.Class) < 5:
                    task.rank = task.ECO / CPU_FLOPS
                else:
                    task.rank = task.ECO / GPU_FLOPS + task.DT / BW

        for node in top_list:
            max = 0.0
            task = dag.tasks[node]
            if value_int(task.Class) < 5:
                task.rank = task.ECO / CPU_FLOPS
            else:
                task.rank = task.ECO / GPU_FLOPS + task.DT / BW
            max = task.rank
            for parent in dag.get_task_parents(task):
                if parent.rank + task.rank > max:
                    max = parent.rank + task.rank
            task.rank = max


    def tlevel_exec(self,dag):
        ex_cpu, ex_gpu = self.ex_map
        top_list = list(nx.topological_sort(dag.skeleton))
        for node in top_list:
            if dag.skeleton.in_degree(node) == 0:
                task = dag.tasks[node]
                if value_int(task.Class) < 5:
                    task.rank = ex_cpu[task.name]
                else:
                    task.rank = ex_gpu[task.name]

        for node in top_list:
            max = 0.0
            task = dag.tasks[node]
            if value_int(task.Class) < 5:
                task.rank = ex_cpu[task.name]
            else:
                task.rank = ex_gpu[task.name]
            max = task.rank
            for parent in dag.get_task_parents(task):
                if parent.rank + task.rank > max:
                    max = parent.rank + task.rank
            task.rank = max

    def blevel(self,dag):
        rev_top_list = list(nx.topological_sort(dag.skeleton))
        rev_top_list.reverse()
        for node in rev_top_list:
            if dag.skeleton.out_degree(node) == 0:
                task = dag.tasks[node]
                # print " ECO0 of ", node, task.ECO
                if value_int(task.Class) < 5:
                    task.rank = task.ECO / CPU_FLOPS
                else:
                    task.rank = task.ECO / GPU_FLOPS + task.DT / BW
                # print "RANKCompute", node, task.rank

        for node in rev_top_list:
            max = 0.0
            task = dag.tasks[node]
            # print "ECOn of ",node, task.ECO
            if value_int(task.Class) < 5:
                task.rank = task.ECO / CPU_FLOPS
            else:
                task.rank = task.ECO / GPU_FLOPS + task.DT / BW
            max = task.rank
            for child in dag.get_task_children(task):
                if child.rank + task.rank > max:
                    max = child.rank + task.rank
            task.rank = max
            # print "RANKCompute", node, task.rank
    
    
    def blevel_wcet(self,dag):

        def wcet(task,dag):
            exec_cpu,exec_gpu=dag.ex_map
            if exec_cpu[task.name]>exec_gpu[task.name]:
                return exec_cpu[task.name]
            else: 
                return exec_gpu[task.name]

        rev_top_list = list(nx.topological_sort(dag.skeleton))
        rev_top_list.reverse()
        for node in rev_top_list:
            if dag.skeleton.out_degree(node) == 0:
                task = dag.tasks[node]
                task.rank = wcet(task,dag)

        for node in rev_top_list:
            max = 0.0
            task = dag.tasks[node]
            # print "ECOn of ",node, task.ECO
            task.rank = wcet(task,dag)
            max = task.rank
            for child in dag.get_task_children(task):
                if child.rank + task.rank > max:
                    max = child.rank + task.rank
            task.rank = max
            # print "RANKCompute", node, task.rank
            # import time
            # time.sleep(1)
    
    def local_deadline(self,dag):
        
        def wcet(task,dag):
            exec_cpu,exec_gpu=dag.ex_map
            if exec_cpu[task.name]>ex_gpu[task.name]:
                return exec_cpu[task.name]
            else: 
                return exec_gpu[task.name]

        rev_top_list = list(nx.topological_sort(dag.skeleton))
        rev_top_list.reverse()
        ex_cpu,ex_gpu=dag.ex_map
        for node in rev_top_list:
            if dag.skeleton.out_degree(node) == 0:
                task = dag.tasks[node]
                task.rank=wcet(task,dag)
        
                
        for node in rev_top_list:
            
            max = 0.0
            if dag.skeleton.out_degree(node) != 0:
                task = dag.tasks[node]
                               
                task.rank = wcet(task,dag)
                max = task.rank
                for child in dag.get_task_children(task):
                    if child.rank + task.rank > max:
                        max = child.rank + task.rank
                task.rank = max
            # print "RANKCompute", node, task.rank

        # Calculate local deadlines

        for node in rev_top_list:
            task=dag.tasks[node]
            if dag.skeleton.out_degree(node) == 0:
                task.rank=dag.deadline
            else:
                task.rank=dag.deadline-task.rank+wcet(task,dag)
        

    
    def blevel_exec(self,dag):
        ex_cpu, ex_gpu = self.ex_map
        rev_top_list = list(nx.topological_sort(dag.skeleton))
        rev_top_list.reverse()
        for node in rev_top_list:
            if dag.skeleton.out_degree(node) == 0:
                task = dag.tasks[node]
                print " ECO0 of ", node, task.ECO
                if value_int(task.Class) < 5:
                    task.rank = ex_cpu[task.name]
                else:
                    task.rank = ex_gpu[task.name]
                print "RANKCompute", node, task.rank

        for node in rev_top_list:
            max = 0.0
            task = dag.tasks[node]
            print "ECOn of ",node, task.ECO
            if value_int(task.Class) < 5:
                task.rank = ex_cpu[task.name]
            else:
                task.rank = ex_gpu[task.name]
            max = task.rank
            for child in dag.get_task_children(task):
                if child.rank + task.rank > max:
                    max = child.rank + task.rank
            task.rank = max
            print "RANKCompute", node, task.rank

    def compute_upward_rank_ECO_DT(self, dag, node):
        max_node = 0
        temp = 0
        rank = 0
        task = dag.tasks[node]
        # If node is dummy node set rank to 0 else set rank to ECO_DT measure
        # print nx.number_of_nodes(dag.skeleton)
        if node is (nx.number_of_nodes(dag.skeleton) - 1):
            rank = 0
        else:
            if value_int(task.Class) < 5:
                rank = task.ECO / CPU_FLOPS
            else:
                rank = task.ECO / GPU_FLOPS + task.DT / BW

                # If node has no successors return rank value
        if (len(dag.skeleton.successors(node)) == 0):
            task.rank = rank
            dag.ranks[node] = rank
            return rank

            # Compute recursively maximum among successors
        else:
            for succ in dag.skeleton.successors(node):
                temp = self.compute_upward_rank_ECO_DT(dag, succ)
                if (max_node < rank + temp):
                    max_node = rank + temp
            rank = max_node
            task.rank = rank
            dag.ranks[node] = rank
            return rank

    def compute_downward_rank_ECO_DT(self, dag, node):
        max_node = 0
        temp = 0
        rank = 0
        task = dag.tasks[node]
        # If node is dummy node set rank to 0 else set rank to ECO_DT measure
        # print nx.number_of_nodes(dag.skeleton)
        if node is dag.num_nodes + 1:
            rank = 0
        else:
            if value_int(task.Class) < 5:
                rank = task.ECO / CPU_FLOPS
            else:
                rank = task.ECO / GPU_FLOPS + task.DT / BW

                # If node has no predecessors return rank value
        if (len(dag.skeleton.predecessors(node)) == 0):
            task.rank = rank
            dag.ranks[node] = rank
            return rank

            # Compute recursively maximum among predecessors
        else:
            for pred in dag.skeleton.predecessors(node):
                temp = self.compute_downward_rank_ECO_DT(dag, pred)
                if (max_node < rank + temp):
                    max_node = rank + temp
            rank = max_node
            task.rank = rank
            dag.ranks[node] = rank
            return rank

    def compute_upward_rank_EXTime(self, dag, node):
        ex_cpu, ex_gpu = self.ex_map
        max_node = 0
        temp = 0
        rank = 0
        task = dag.tasks[node]
        # If node is dummy node set rank to 0 else set rank to ExTime measure
        if node is dag.num_nodes:
            rank = 0
        else:
            if value_int(task.Class) < 5:
                rank = ex_cpu[task.name]
            else:
                rank = ex_gpu[task.name]

                # If node has no successors return rank value
        if (len(dag.skeleton.successors(node)) == 0):
            task.rank = rank
            dag.ranks[node] = rank
            return rank

            # Compute recursively maximum among successors
        else:
            for succ in dag.skeleton.successors(node):
                temp = self.compute_upward_rank_EXTime(dag, succ)
                if (max_node < rank + temp):
                    max_node = rank + temp
            rank = max_node
            task.rank = rank
            dag.ranks[node] = rank
            return rank

    def compute_downward_rank_EXTime(self, dag, node):
        ex_cpu, ex_gpu = self.ex_map
        max_node = 0
        temp = 0
        rank = 0
        task = dag.tasks[node]
        # If node is dummy node set rank to 0 else set rank to ExTime measure
        if node is dag.num_nodes + 1:
            rank = 0
        else:
            if value_int(task.Class) < 5:
                rank = ex_cpu[task.name]
            else:
                rank = ex_gpu[task.name]

                # If node has no predecessors return rank value
        if (len(dag.skeleton.predecessors(node)) == 0):
            task.rank = rank
            dag.ranks[node] = rank
            return rank

            # Compute recursively maximum among predecessors
        else:
            for pred in dag.skeleton.predecessors(node):
                temp = self.compute_downward_rank_EXTime(dag, pred)
                if (max_node < rank + temp):
                    max_node = rank + temp
            rank = max_node
            task.rank = rank
            dag.ranks[node] = rank
            return rank

    def compute_percentage_remaining_ECO(self, dag):
        for node in dag.skeleton.nodes():
            ECO_Sum = 0.0
            for succ in dag.skeleton.descendants(node):
                ECO_Sum = ECO_Sum + dag.tasks[succ].ECO
            dag.tasks[node].rank = ECO_Sum

    def compute_percentage_remaining_DT(self, dag):
        for node in dag.skeleton.nodes():
            DT_Sum = 0.0
            for succ in dag.skeleton.descendants(node):
                DT_Sum = DT_Sum + dag.tasks[succ].DT
            dag.tasks[node].rank = DT_Sum

    def compute_DT_next_level(self, dag):
        for node in dag.skeleton.nodes():
            DT_Sum = 0.0
            for succ in dag.skeleton.successors(node):
                DT_Sum += dag.skeleton[node][succ]['weight']
            dag.tasks[node].rank = DT_Sum

    def reset_ranks(self, rank_name):
        for dag in self.dags:
            dag.reset_node_ranks(rank_name)

    def compute_rank_dags(self, compute_rank, nCPU, mGPU):
        # print "Computing ranks of dags", compute_rank
        # import time
        # time.sleep(1)
        if compute_rank is "upward_rank_ECO_DT":
            for dag in self.dags:
                print dag.dag_id
                print nx.is_directed_acyclic_graph(dag.skeleton)
                dag.add_dummy_node_source()
                self.compute_upward_rank_ECO_DT(dag, dag.num_nodes)
                dag.remove_dummy_node_source()
                dag.update_rank_values(compute_rank)
        if compute_rank is "downward_rank_ECO_DT":
            for dag in self.dags:
                dag.add_dummy_node_exit()
                self.compute_downward_rank_ECO_DT(dag, dag.num_nodes + 1)
                dag.remove_dummy_node_exit()
                dag.update_rank_values(compute_rank)
        if compute_rank is "upward_rank_EXTime":
            for dag in self.dags:
                dag.add_dummy_node_source()
                self.compute_upward_rank_EXTime(dag, dag.num_nodes)
                dag.remove_dummy_node_source()
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "downward_rank_EXTime":
            for dag in self.dags:
                dag.add_dummy_node_exit()
                self.compute_downward_rank_ECO_DT(dag, dag.num_nodes + 1)
                dag.remove_dummy_node_exit()
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "percentage_ECO_remaining":
            for dag in self.dags:
                self.compute_percentage_remaining_ECO(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "percentage_DT_remaining":
            for dag in self.dags:
                self.compute_percentage_remaining_DT(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "DT_next_level":
            for dag in self.dags:
                self.compute_DT_next_level(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "blevel":
            for dag in self.dags:
                self.blevel(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "tlevel":
            for dag in self.dags:
                self.tlevel(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "blevel_exec":
            for dag in self.dags:
                self.blevel_exec(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "tlevel_exec":
            for dag in self.dags:
                self.tlevel_exec(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        if compute_rank is "oct":
            for dag in self.dags:
                self.oct(dag, nCPU, mGPU)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        
        if compute_rank is "local_deadline":
            for dag in self.dags:
                self.local_deadline(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)
        
        if compute_rank is "blevel_wcet":
            for dag in self.dags:
                # print "blevel_wcet", dag.dag_id, dag.job_id
                
                # time.sleep(1)
                self.blevel_wcet(dag)
                dag.update_rank_values(compute_rank)
                dag.initialize_task_component_ranks(compute_rank)

