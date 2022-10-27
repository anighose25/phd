from core import *
class SimTask(object):
    """
    Class to handle all operations perfomed on simulation tasks.
    """

    def __init__(self, key="", uid=-1, dag_id=-1, feature_dict=dict(), extime=-1,job_id=-1):
        self.id = uid
        self.dispatch_step = 0
        self.dag_id = dag_id
        self.job_id = job_id
        self.name = key
        self.execution_time = extime
        self.projected_ex_time = extime
        self.estimated_ex_time = 0.0
        if any(feature_dict):
            self.Float = float(feature_dict['Float16'])
            self.Float4 = float(feature_dict['Float32'])
            self.Float4 = float(feature_dict['Float32'])
            self.Int = float(feature_dict['Int16'])
            self.Int4 = float(feature_dict['Int32'])
            # self.DT = float(feature_dict['DataTransfer'])
            self.DT = 0.0
            self.Barrier = float(feature_dict['Barrier'])
            self.ComputePerDT = float(feature_dict['ComputePerDataTransfer'])
            self.Branches = float(feature_dict['TotalBranches'])
            self.Memory = float(feature_dict['TotalMemory'])
            self.WorkItems = float(feature_dict['NumberOfWorkItems'])
            self.Memory = float(feature_dict['TotalMemory'])
            self.CMRatio = float(feature_dict['ComputeToMemoryRatio'])
            # self.ECO = self.WorkItems*((self.Int + self.Int4 + self.Float + self.Float4) * self.DT/ self.WorkItems)
            # self.ECO = ((self.Int + self.Int4 + self.Float + self.Float4)) * self.WorkItems

        self.Class = "TEN"
        self.rank = 0
        self.start_time = 0
        self.finish_time = 0
        self.device_type = ""
        self.device_id = 0
        self.is_finished = False
        self.in_frontier = False
        self.contract = True
        # if key is not "dummy" and key is not "":
        #     self.Kernel_Object = obtain_kernel_info(key)
        #     for buf in self.Kernel_Object.buffer_info['output']:
        #         buf_size = float(buf['size']) * get_sizeof(buf['type'])
        #         self.DT += buf_size
        #         # print "OUTPUT_DT_CALCULATION_"+key, buf_size
        #     for buf in self.Kernel_Object.buffer_info['input']:
        #         buf_size = float(buf['size']) * get_sizeof(buf['type'])
        #         self.DT += buf_size
        #         # print "INPUT_DT_CALCULATION_" + key, buf_size
        #     for buf in self.Kernel_Object.buffer_info['io']:
        #         buf_size = float(buf['size']) * get_sizeof(buf['type'])
        #         self.DT += buf_size
        #         # print "IO_DT_CALCULATION_" + key, buf_size
        #     self.ECO = self.WorkItems * ((self.Int + self.Int4 + self.Float + self.Float4) * self.DT / self.WorkItems)
        self.levels = dict()
        self.rank_values = {'DT_next_level': 0.0, 'upward_rank_ECO_DT': 0.0, 'downward_rank_ECO_DT': 0.0,
                            'upward_rank_EXTime': 0.0, 'downward_rank_EXTime': 0.0, 'percentage_ECO_remaining': 0.0,
                            'percentage_DT_remaining': 0.0, 'blevel': 0.0, 'tlevel': 0.0, 'oct': 0.0, 
                            'local_deadline': 0.0}
        self.latest_start_time = 0.0
        # self.feature_vector = []
        # if any(feature_dict):
        #     for feat in feature_dict.keys():
        #         if feat!= "Class":
        #             self.feature_vector.append(float(feature_dict[feat]))
        self.feature_vector = 0.0
        if any(feature_dict):
            for feat in feature_dict.keys():
                if feat != "Class":
                    self.feature_vector += (float(feature_dict[feat]) * float(feature_dict[feat]))

    def __cmp__(self, other):
        return cmp(self.rank, other.rank)

    def get_dataset(self):
        return self.Kernel_Object.dataset

    def get_dimension(self):
        return self.Kernel_Object.work_dimension

    def get_num_input_buffers(self):
        return len(self.Kernel_Object.buffer_info['input'])

    def get_num_output_buffers(self):
        return len(self.Kernel_Object.buffer_info['output'])

    def get_input_buffer_sizes(self):
        input_buf_sizes = []
        for buf in self.Kernel_Object.buffer_info['input']:
            input_buf_sizes.append(buf['size'])
        return input_buf_sizes


    def get_output_buffer_sizes(self):
        output_buf_sizes = []
        for buf in self.Kernel_Object.buffer_info['output']:
            output_buf_sizes.append(buf['size'])
        return output_buf_sizes



    def get_output_buffer_dimension(self):
        dataset = float(self.Kernel_Object.dataset)
        output_buf_size = float(self.Kernel_Object.buffer_info['output']['size'])
        if output_buf_size/dataset > 1:
            return 2
        else:
            return 1




class SimTaskComponent(object):
    """
    Class to handle all operations perfomed on simulation task components.
    """

    def __init__(self, kernel):
        self.id = generate_unique_id()
        self.kernels = set()
        self.sorted_kernels = list()
        self.kernels.add(kernel)
        self.is_horizontal_component = False
        self.is_vertical_component = False
        self.rank_values = {'DT_next_level': 0.0, 'upward_rank_ECO_DT': 0.0, 'downward_rank_ECO_DT': 0.0,
                            'upward_rank_EXTime': 0.0, 'downward_rank_EXTime': 0.0, 'percentage_ECO_remaining': 0.0,
                            'percentage_DT_remaining': 0.0, 'blevel': 0.0, 'tlevel': 0.0, 'oct': 0.0}
        self.rank_name = ""
        self.projected_ex_time = kernel.execution_time
        self.start_time = 0
        self.finish_time = 0
        self.to_be_scheduled = False
        self.future_device_type = ""
        self.future_device_id = -1
        self.device_type = ""
        self.device_id = -1
        self.is_finished = False
        self.dag_id = kernel.dag_id
        self.Class = kernel.Class
        self.contract = True
        self.local_frontier = collections.deque()
        self.number_of_tasks = 0
        self.rank=0
        self.dispersion = 0.0
        self.is_dispatched = False



    # def __gt__(self, other):
    #     return (self.rank_value[self.rank_name] < other.rank_value[other.rank_name])

    def __gt__ (self, other):
        if self is None or other is None:
            return False
        return self.get_first_kernel().rank > other.get_first_kernel().rank

    def is_supertask(self):
        """
        Returns a boolean value indicating whether task component has more than one kernel
        :return:
        :rtype:
        """
        return len(self.get_kernels()) > 1


    def get_component_deadline(self,dag):
        
        #reverse topologically sorted list of kernels 
        
        kernels = self.get_kernels_sorted(dag)[::-1]
        
        subgraph = dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))
        min_rank = float('inf')
        for k in kernels:
            task_id = k.id
            if subgraph.out_degree(task_id) == 0:
                min_rank = min(min_rank,k.rank)
            else:
                break
        return min_rank
                
    def get_component_wcet_rank(self):
        max_wcet = 0.0
        for k in self.kernels:
            max_wcet = max(max_wcet,k.rank_values['blevel_wcet'])
        return max_wcet



    def get_first_kernel(self):
        """
        Returns the first kernel (SimTask object)in the task component (indegree zero)
        :return:
        :rtype:
        """
        return list(self.kernels)[0]

    def get_kernels(self):
        """
        Returns all kernels (SimTask objects)
        :return:
        :rtype:
        """
        return self.kernels

    def get_component_time(self,ex_map,device):
        ex_cpu,ex_gpu = ex_map
        t=0.0
        for k in self.kernels:
            if device=="cpu":
                t+=ex_cpu[k.name]
            else:
                t+=ex_gpu[k.name]
        return t

    def get_kernel_ids(self):
        """
        Return list of ids pertaining to each kernel (SimTask Object)
        :return:
        :rtype:
        """
        return map(lambda k: k.id, self.get_kernels())

    def get_kernel_classes(self):
        """
        Returns list of Partition Classes pertaining to each kernel (SimTask Object)
        :return:
        :rtype:
        """
        return map(lambda k: k.Class, self.get_kernels())

    def get_kernel_pvalues(self):
        """
        Returns list of Partition Classes pertaining to each kernel (SimTask Object)
        :return:
        :rtype:
        """
        return map(lambda k: partition_class_absolute(k), self.get_kernels())

    def get_kernel_names(self):
        """
        Returns list of names pertaining to each kernel (SimTask Object)
        :return:
        :rtype:
        """
        return map(lambda k: k.name, self.get_kernels())

    def get_kernels_sorted(self, dag):
        """
        Returns list of kernels (SimTask objects) in topologically sorted order
        :param dag:
        :type dag:
        :return:
        :rtype:
        """
        return map(lambda kid: dag.tasks[kid],
                   nx.algorithms.topological_sort(dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))))

    def get_free_kernels(self, dag):
        subgraph = dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))
        free_task_ids = []
        for i in subgraph.nodes():
            if subgraph.in_degree(i) == 0:
                free_task_ids.append(i)
        free_tasks = []
        for task_id in free_task_ids:
            free_tasks.append(dag.tasks[task_id])
        return free_tasks
    
    def get_free_kernel_ids(self, dag):
        subgraph = dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))
        free_task_ids = []
        for i in subgraph.nodes():
            if subgraph.in_degree(i) == 0:
                free_task_ids.append(i)
        return free_task_ids


    def top(self, dag):
        subgraph = dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))
        free_task_ids = []
        for i in subgraph.nodes():
            if subgraph.in_degree(i) == 0:
                free_task_ids.append(i)
        free_tasks = []
        for task_id in free_task_ids:
            free_tasks.append(dag.tasks[task_id])
        return free_tasks

    def btm(self, dag):
        subgraph = dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))
        free_task_ids = []
        for i in subgraph.nodes():
            if subgraph.out_degree(i) == 0:
                free_task_ids.append(i)
        free_tasks = []
        for task_id in free_task_ids:
            free_tasks.append(dag.tasks[task_id])
        return free_tasks

    def TL(self, dag):
        subgraph = dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))
        max_start_time = 0.0
        free_task_ids = []
        for i in subgraph.nodes():
            if subgraph.in_degree(i) == 0:
                free_task_ids.append(i)
        for task_id in free_tasks:
            max_start_time = max(max_start_time, dag.tasks[task_id].rank_values['tlevel'])
        return max_start_time

    def BL(self, dag):
        subgraph = dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))
        max_start_time = 0.0
        free_task_ids = []
        for i in subgraph.nodes():
            if subgraph.out_degree(i) == 0:
                free_task_ids.append(i)
        for task_id in free_tasks:
            max_start_time = max(max_start_time, dag.tasks[task_id].rank_values['blevel'])
        return max_start_time

    def LV(self, dag):
        return self.LV(dag) + self.BL(dag)


    def get_kernel_ids_sorted(self, dag):
        """
        Returns list of kernel ids (SimTask objects) in topologically sorted order
        :param dag:
        :type dag:
        :return:
        :rtype:
        """
        return map(lambda k: k.id, self.get_kernels_sorted(dag))

    def add_kernels_from_task(self, task):
        """
        Merges a child task into itself.
        """
        self.kernels.update(task.get_kernels())

    def remove_kernel(self, kernel):
        """
        Removes given kernel from this task.
        """
        if kernel in self.kernels:
            self.kernels.remove(kernel)
        else:
            raise Exception("Given kernel is not a subset of this task")