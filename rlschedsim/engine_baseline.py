from dagcreator import *
from ranker import *
#################################################################################################


class ScheduleEngine(object):
    """
    Class for backbone of scheduling algorithms
    """

    def __init__(self, nCPU, mGPU, rank_name, ex_map,online=False, global_map=None,adas=False, num_jobs=0):
        self.avg_makespan=0.0
        self.frontier = Q.PriorityQueue()
        self.nCPU = nCPU
        self.mGPU = mGPU
        self.num_CPU_devices = nCPU
        self.num_GPU_devices = mGPU
        self.num_jobs = num_jobs
        self.dispatch_step = 0
        self.device_history = {"gpu": [], "cpu": []}
        self.ready_queue = {"gpu": collections.deque(), "cpu": collections.deque()}
        self.dispatch_queue = {"gpu": {}, "cpu": {}}
        for i in range(0,nCPU):
            self.dispatch_queue['cpu'][i] = collections.deque()
        for i in range(0,mGPU):
            self.dispatch_queue['gpu'][i] = collections.deque()
        
        self.available_device_lookahead = {"gpu": [], "cpu": [], "nD": [nCPU, mGPU]}
        self.time_stamp = 0.0
        self.prev_time_stamp = 0.0
        self.estimate_time_stamp = 0.0
        self.estimate_prev_time_stamp = 0.0
        self.currently_executing = {"gpu": [], "cpu": []}
        self.waiting_tasks = {"gpu": {}, "cpu": {}}
        self.makespan = 0.0
        self.rank_name = rank_name
        self.ex_map = ex_map
        self.online=False
        self.adas=False
        self.arriving_dags = []
        self.dag_arrival_time_stamps = []
        self.dag_arrival_timestamp_map = {}
        if online:
            self.online=True
            # self.arrival_time_stamps = self.random_arrival_times()
            # self.create_random_dags_online()
        for i in range(0, nCPU):
            self.waiting_tasks['cpu'][i] = None
        for i in range(0, mGPU):
            self.waiting_tasks['gpu'][i] = None
        if adas:
            self.adas=True
        self.dags = []
        self.global_dags = []
        self.log = {}
        self.global_map = None
        if global_map:
            self.global_map = global_map
        self.dispatch_history = []
        self.task_set_history = []
    



    # Operations for online scheduling

    def random_arrival_times(self,lambda_rate=1.5, time_horizon=10):
        intervals = [random.expovariate(lambda_rate) for i in range(1000)]
        timestamps = [0.0]
        timestamp = 0.0
        for t in intervals:
            timestamp += t*1000000
            timestamps.append(timestamp)
        return timestamps[:time_horizon]
    
    def random_arrival_times_limit(self,lambda_rate=1.5, time_limit=10):
        intervals = [random.expovariate(lambda_rate) for i in range(1000)]
        timestamps = [0.0]
        timestamp = 0.0
        time_limit *=1000000
        for t in intervals:
            timestamp += t*1000000
            if timestamp>time_limit:
                break
            timestamps.append(timestamp)
        self.arrival_time_stamps = timestamps

    def create_mm_dags_online(self,filename,weights):
        n=100
        work_item_array = []
        work_items=[1024,2048,4096,8192]
        index_elements = []
       
        for i in range(n):
            index_elements.append(i)
            work_item_array.append(0)
       
        for w,W in zip(weights,work_items):
            num_samples = int(w*100)
            current_list = random.sample(index_elements,num_samples)
            for i in current_list:
                work_item_array[i]=W
            index_elements = list(set(index_elements)-set(current_list))
       
        # print work_item_array
       
        file_name_1024 = "graphs_1024.txt"
        file_name_2048 = "graphs_2048.txt"
        file_name_4096 = "graphs_4096.txt"
        file_name_8192 = "graphs_8192.txt"
       
        dagpool_1024=[]
        dagpool_2048=[]
        dagpool_4096=[]
        dagpool_8192=[]
        
        for graph in open(file_name_1024).readlines():
            dagpool_1024.append(graph.split(":")[0])
        for graph in open(file_name_2048).readlines():
            dagpool_2048.append(graph.split(":")[0])
        for graph in open(file_name_4096).readlines():
            dagpool_4096.append(graph.split(":")[0])
        for graph in open(file_name_8192).readlines():
            dagpool_8192.append(graph.split(":")[0])
        
        dag_names = []
        
        for W in work_item_array:
            if W ==1024:
                dag_name = random.choice(dagpool_1024)
            elif W == 2048:
                dag_name = random.choice(dagpool_2048)
            elif W == 4096:
                dag_name = random.choice(dagpool_4096)
            elif W == 8192:
                dag_name = random.choice(dagpool_8192)

            dag_names.append(dag_name)
        
        # print len(dag_names)
        # print len(self.arrival_time_stamps)
        d = ','.join(dag_names) + "\n"
        ts =','.join(map(str,self.arrival_time_stamps)) + "\n"
        
        configuration_file = open(filename,'w')
        configuration_file.write(d)
        configuration_file.write(ts)
        configuration_file.close()



    def select_rates(self, dags):
        for dag in dags:
            dag.rate=random.choice(dag.rates)
            dag.period=1/dag.rate
            dag.period=int(dag.period)-int(dag.period)%5 

    def get_dag_prt(self, rank_name):

        dag_prt = {}
        # print "Number of JOBS", self.num_jobs
        for i in range(0, self.num_jobs):
            # print i 
            dag_prt[i] = 0.0
        
        for tc in self.frontier.queue:
            job_id = self.dags[tc.dag_id].job_id
            current_value = tc.rank_values[rank_name]
            
            if job_id in dag_prt.keys():
                dag_prt[job_id] = max(dag_prt[job_id],current_value)    
            else:
                dag_prt[job_id] = current_value    
        
        prt_values = []
        # print dag_prt
        for i in sorted(dag_prt.keys()):
            prt_values.append(dag_prt[i])
        return prt_values
    
    def get_dispatch_history(self, file_name):
        f = open(file_name, 'w')
        for line in self.dispatch_history:
            f.write(line)
        f.close()
    
    def get_task_set_history(self, file_name):
        f = open(file_name, 'w')
        for line in self.task_set_history:
            f.write(line)
        f.close()
            
    def generate_periodic_schedule(self, dags, periods=None,dag_history="",episode=-1):
        import fractions
        from math import ceil
        def _lcm(a,b): return abs(a * b) / fractions.gcd(a,b) if a and b else 0
        def lcm(a):
            return reduce(_lcm, a)
        rates = []
        if periods is None:
            self.select_rates(dags)
            rates = [d.rate for d in dags]
            periods = [d.period for d in dags]
        else:
            for p,d in zip(periods,dags):
                d.period=p
                d.rate=1/float(d.period)
                rates.append(d.rate)
        # print rates
        wcets=[d.wcet for d in dags]
        print wcets
        print periods
        hyperperiod=lcm(periods)
        print hyperperiod
        DC=DAGCreator()
        arrival_instances = []
        dag_arrivals = []
        
        for d in dags:
            i=0.0
            s=0.0
            # print int(d.period/hyperperiod)
            for i in xrange(0,int(ceil(hyperperiod/d.period))):
                release=d.phase+d.period*i
                task_arrival_pair = (d,release)
                arrival_instances.append(task_arrival_pair)
                # dag_arrivals.append((dag,release))
 

        arrival_instances = sorted(arrival_instances,key=lambda x: x[1])
        dag_arrivals = sorted(dag_arrivals,key=lambda x: x[1])
        dag_history_file = None
        if episode >=0:
            dag_history_file = open(dag_history+"/dag_history_"+str(episode)+".stats",'w')
        # print arrival_instances
        dag_id = 0
        for d, release in arrival_instances:
            
            job_id = d.dag_id
            input_file = "ADAS_Graphs/"+d.name
            dag=DC.create_adas_job_from_file_with_fused_times(input_file,dag_id,self.global_map,self.ex_map,job_id,release,d.period)
            # print dag.job_id, dag.release,dag.deadline,dag.wcet,dag.period
            if episode >= 0:
                print >>dag_history_file, str(dag_id) + " " + str(dag.job_id) + " " +str(dag.release) +" " + str(dag.deadline)
            self.dag_arrival_time_stamps.append(release)
            self.dag_arrival_timestamp_map[dag]=release
            self.arriving_dags.append(dag)
            dag_id +=1
        r = Ranker(self.arriving_dags, self.ex_map)
        r.compute_rank_dags("blevel_wcet", self.num_CPU_devices, self.num_GPU_devices)
        r.reset_ranks("blevel_wcet")
        r.compute_rank_dags(self.rank_name, self.num_CPU_devices, self.num_GPU_devices)
        # for d in self.arriving_dags[0:10]:
        #     d.print_rank_info()

        for d in self.arriving_dags:
            self.global_dags.append(d) 
                          
        a_time = self.dag_arrival_time_stamps[0]
        while a_time == 0.0:
            
            dag = self.arriving_dags.pop(0)
            self.dags.append(dag)
            self.dag_arrival_time_stamps.pop(0)

            if len(self.dag_arrival_time_stamps) > 0:
                a_time = self.dag_arrival_time_stamps[0]
            else:
                break
        
        if episode >= 0 :
            dag_history_file.close()
        
        return rates, periods



    
    def create_random_dags_online(self, rank, filename=None,work_item=None):
        # file_name = "NewStats/dump_graphs_new_"+str(rank)+".stats"
        file_name = ""
        if work_item:
            file_name = "graphs_"+str(work_item)+".txt"
        else:
            file_name = "graphs_8192.txt"
        graphs = open(file_name, 'r').readlines()
        dagpool = []
        for graph in graphs:
            dagpool.append(graph.split(":")[0])
                
        DC = DAGCreator()
        starting_dag_id = len(self.dags)
        dag_names = []
        for i in range(len(self.arrival_time_stamps)):
            # print "Creating dag ", starting_dag_id + i
            dag_name = random.choice(dagpool)
            dag_names.append(dag_name)
            dag = DC.create_dag_from_file(dag_name, starting_dag_id + i, self.global_map, self.ex_map)
            # DC.dump_graph_ids(dag, "list_dag1_ids.png")
            self.dag_arrival_timestamp_map[dag]=self.arrival_time_stamps[i]
            # dagname_ts_map[dag_name]=self.arrival_time_stamps[i]
            self.arriving_dags.append(dag)
            r = Ranker(self.arriving_dags, self.ex_map)
            r.compute_rank_dags(self.rank_name, self.num_CPU_devices, self.num_GPU_devices)

        if filename:
            d = ','.join(dag_names) + "\n"
            ts =','.join(map(str,self.arrival_time_stamps)) + "\n"
            configuration_file = open(filename,'w')
            configuration_file.write(d)
            configuration_file.write(ts)
            configuration_file.close()

        dag = self.arriving_dags.pop(0)
        self.dags.append(dag)
        self.arrival_time_stamps.pop(0)
        '''
        for task_component in dag.free_task_components:
            task_component.get_first_kernel().in_frontier = True
            print "OnlineAddFrontier ",task_component.get_kernel_ids()
            self.frontier.put(task_component)
        '''

    def release_dags(self, timestamp):
        dags = []
        
        if len(self.dag_arrival_time_stamps) <=0:
            return
        t = self.dag_arrival_time_stamps[0] # WAS THERE IN ORIGINAL FRAMEWORK
        # t = self.dag_arrival_time_stamps.pop(0) # WAS NOT THERE IN ORIGINAL FRAMEWORK
        
        #if t > timestamp:  WAS THERE IN ORIGINAL FRAMEWORK
        
        # if t < timestamp:
        #     dag = self.arriving_dags.pop(0)
        #     self.dags.append(dag)
        #     dag.starting_time = self.dag_arrival_time_stamps.pop(0)
        #     for task_component in dag.free_task_components:
        #         task_component.get_first_kernel().in_frontier = True
        #         self.frontier.put(task_component)
        #     self.time_stamp = t
        # else:               # WAS NOT THERE IN ORIGINAL FRAMEWORK
            # self.dag_arrival_time_stamps.insert(0,t) # WAS NOT THERE IN ORIGINAL FRAMEWORK
        
        while t <= timestamp and len(self.dag_arrival_time_stamps) > 0:
            # print t,timestamp
            dag = self.arriving_dags.pop(0)
            self.dags.append(dag)
            dag.starting_time = self.dag_arrival_time_stamps.pop(0)
            if len(self.dag_arrival_time_stamps) > 0:
                t = self.dag_arrival_time_stamps[0]
            for task_component in dag.free_task_components:
                task_component.get_first_kernel().in_frontier = True
                self.frontier.put(task_component)
        # print "OnlineRelease"    

    
    
    
    def online_setup(self, rate, horizon, cluster_rank, filename=None, work_item=None):
        self.random_arrival_times_limit(rate,horizon)
        self.create_random_dags_online(cluster_rank,filename,work_item=None)
    
    def generate_mm_configurations(self,rate,limit,weights,filename):
        self.arrival_time_stamps = self.random_arrival_times(rate,limit)
        self.create_mm_dags_online(filename,weights)
        

    def calculate_avg_makespan(self):
        for dag in self.dags:
            # print dag.job_id,dag.starting_time,dag.finishing_time,dag.deadline
            self.avg_makespan += dag.finishing_time-dag.starting_time
        self.avg_makespan = self.avg_makespan/len(self.dags)
    
    def calculate_avg_lateness(self):
        lateness = 0.0
        for dag in self.dags:
            # print "\n", dag.job_id,"RELEASE", dag.release, "START TIME", dag.starting_time, "FINISHING TIME ", dag.finishing_time,"DEADLINE", dag.deadline, "WCET", dag.wcet
            # dag.print_task_info()
            lateness += dag.finishing_time-dag.deadline
        lateness = lateness/len(self.dags)
        
        return lateness
    
    def calculate_max_lateness(self):
        max_lateness = 0.0
        for dag in self.dags:
            # print "\n", dag.job_id,"RELEASE", dag.release, "START TIME", dag.starting_time, "FINISHING TIME ", dag.finishing_time,"DEADLINE", dag.deadline, "WCET", dag.wcet
            # dag.print_task_info()
            max_lateness = max(max_lateness,dag.finishing_time-dag.deadline)
              
        return max_lateness

    def calculate_percent_lateness(self):
        lateness = 0.0
        makespan = 0.0
        percent_lateness = 0.0
        
        for dag in self.dags:
            # print "\n", dag.job_id,"RELEASE", dag.release, "START TIME", dag.starting_time, "FINISHING TIME ", dag.finishing_time,"DEADLINE", dag.deadline, "WCET", dag.wcet
            # dag.print_task_info()
            lateness = dag.finishing_time-dag.deadline
            makespan =dag.finishing_time-dag.starting_time
            percent_lateness += lateness/makespan
        return percent_lateness/len(self.dags)

    

    def calculate_avg_tardiness(self):
        tardiness = 0.0
        counter = 0
        for dag in self.dags:
            # print "\n", dag.job_id,"RELEASE", dag.release, "START TIME", dag.starting_time, "FINISHING TIME ", dag.finishing_time,"DEADLINE", dag.deadline, "WCET", dag.wcet
            # dag.print_task_info()
            t = dag.finishing_time-dag.deadline
            if t > 0:
                tardiness += dag.finishing_time-dag.deadline
                counter +=1
        if counter > 0:
            tardiness = tardiness/counter
        print "TARDINESS", tardiness
        return tardiness,counter
        
   
   
    def dump_online_makespans(self,name):
        filename = open(name,'w')
        for dag in self.dags:
            makespan = dag.finishing_time - dag.starting_time
            line = dag.name+ ":" + str(makespan)+"\n"
            # print line
            filename.write(line)
        filename.close()

    
    def online_setup_configuration(self, online_configuration_file):
        online_configuration = open(online_configuration_file,'r').readlines()
        arriving_dags = online_configuration[0].strip("\n").split(",")
        arriving_timestamps = map(float, online_configuration[1].strip("\n").split(","))
        self.arrival_time_stamps = arriving_timestamps
        DC = DAGCreator()
        starting_dag_id = len(self.dags)
        for i in range(len(self.arrival_time_stamps)):
            dag_name = arriving_dags[i]
            dag = DC.create_dag_from_file(dag_name, starting_dag_id + i, self.global_map, self.ex_map)
            dag.modify_partition_classes("GraphsNewAccuracy/accuracy_"+dag_name[10:][:-6]+".stats")
            # print "modified"
            self.dag_arrival_timestamp_map[dag]=self.arrival_time_stamps[i]
            self.arriving_dags.append(dag)
            r = Ranker(self.arriving_dags, self.ex_map)
            r.compute_rank_dags(self.rank_name, self.num_CPU_devices, self.num_GPU_devices)
        dag = self.arriving_dags.pop(0)
        self.dags.append(dag)
        dag.starting_time = self.arrival_time_stamps.pop(0)

        

   
   

        



        
        

    # Operations for Task Sets

    #
    # def dispatch_task_set(self, cpu, gpu, task_component, dag):    
    #     # Update start time of task with current time stamp of system
    #     print "time Stamp: " + str(self.time_stamp)
    #     task_component.start_time = self.time_stamp
    #     # Update currently executing list with task, dag and device id dispatched
    #
    #     task_component.sorted_list = list(task_component.get_kernels_sorted(dag))
    #     task = task_component.sorted_list[0]
    #     print "Dispatching " + str(task.id) + " of " + str(dag.dag_id)
    #     task.start_time = self.time_stamp
    #     del task_component.sorted_list[0]
    #     dag.reduce_execution_time(task, task_component)
    #     if (cpu == -1 and gpu != -1):
    #         self.currently_executing['gpu'].append((task, dag, gpu))
    #         self.mGPU -= 1
    #         heapq.heappush(self.available_device_lookahead['gpu'], task.projected_ex_time)
    #     if (cpu != -1 and gpu == -1):
    #         self.currently_executing['cpu'].append((task, dag, cpu))
    #         self.nCPU -= 1
    #         heapq.heappush(self.available_device_lookahead['cpu'], task.projected_ex_time)

    def dispatch_task_set(self, cpu, gpu, task_component, dag):
        # print "time Stamp: " + str(self.time_stamp)
        task_component.is_dispatched = True
        task_component.start_time = self.time_stamp
        task_list = task_component.get_free_kernels(dag)
        total_task_list_ids = task_component.get_kernel_ids()
        task_component.number_of_tasks = len(total_task_list_ids)
        # print "TASK_COMPONENT size",
        # print task_component.number_of_tasks,
        # print total_task_list_ids
        for t in task_list:
            task_component.local_frontier.append(t)

        task = task_component.local_frontier.popleft()
        if self.adas:
            # print "REDUCING TIME"
            dag.reduce_adas_execution_time(task, task_component)
        else:
            dag.reduce_execution_time(task, task_component)
        # print "\n DISPATCHING " , total_task_list_ids , "of DAG " + str(dag.dag_id) , "to CPU", cpu, "GPU", gpu
        task.start_time = self.time_stamp

        task.in_frontier = False
        task.dispatch_step = self.dispatch_step
        self.dispatch_step += 1
        # print "TASK_DISPATCHED_GLOBAL ", task.id, task_component.number_of_tasks
        if (cpu == -1 and gpu != -1):
            self.currently_executing['gpu'].append((task, dag, gpu))
            if not task_component.to_be_scheduled:
                self.mGPU -= 1
            heapq.heappush(self.available_device_lookahead['gpu'], task.projected_ex_time)
            # heapq.heappush(self.available_device_lookahead['gpu'], task.estimated_ex_time)
        if (cpu != -1 and gpu == -1):
            self.currently_executing['cpu'].append((task, dag, cpu))
            if not task_component.to_be_scheduled:
                self.nCPU -= 1
            heapq.heappush(self.available_device_lookahead['cpu'], task.projected_ex_time)
            # heapq.heappush(self.available_device_lookahead['cpu'], task.estimated_ex_time)
        if task_component.to_be_scheduled == True:
            task_component.to_be_scheduled = False
        self.log_dispatch_history(cpu,gpu,task_component,dag)
        return self.time_stamp

    
    def log_dispatch_history(self,cpu,gpu,task_component,dag):
        fused_kernels = task_component.get_kernel_ids_sorted(dag)
        task_list = task_component.get_free_kernel_ids(dag)
        kernel_id_str = fused_kernels[0]
        fused_kernel_ids_str = '-'.join(map(str,fused_kernels))
        parent_ids = []
        dag_id_str = dag.job_id
        for t in task_list:
            parents = dag.get_kernel_parent_ids(t)
            parent_ids.extend(parents)
        parent_ids_str = "-1"
        if len(parent_ids)>0:
            parent_ids = list(set(parent_ids))
            parent_ids_str = "-".join(map(str,parent_ids)) 
        else:
            parent_ids_str = "-1"
        platform = "-1"
        device_id = "-1"
        task_ids=task_component.get_kernel_ids()
        expected_time = 0.0
        ex_cpu, ex_gpu = self.ex_map
        if (cpu == -1 and gpu != -1):
            platform = "0"
            device_id = gpu
            if len(task_ids)>1:
                expected_time=dag.fused_kernel_timings['gpu'][tuple(sorted(tuple(task_ids)))]
            else:
                expected_time = ex_gpu[dag.tasks[task_ids[0]].name]
        elif (cpu != -1 and gpu == -1):
            platform = "1"
            device_id = cpu
            if len(task_ids)>1:
                expected_time=dag.fused_kernel_timings['cpu'][tuple(sorted(tuple(task_ids)))]
            else:
                expected_time = ex_cpu[dag.tasks[task_ids[0]].name]
        
        terminal_task = 0
        if len(dag.get_task_component_children(task_component)) == 0:
            terminal_task = 1



        # print str(dag_id_str)+","+str(kernel_id_str)+","+str(fused_kernel_ids_str)+","+str(parent_ids_str)+","+str(platform)+","+str(device_id)
        deadline_str = task_component.get_component_deadline(dag)
        start_time_str = task_component.start_time
        dag_id = str(dag.dag_id)
        history = str(dag_id_str)+","+str(kernel_id_str)+","+str(fused_kernel_ids_str)+","+str(parent_ids_str)+","+str(platform)+","+str(device_id)+","+str(expected_time)+","+str(deadline_str)+","+str(terminal_task)+","+str(dag_id)+"\n"
        
        self.dispatch_history.append(history)
    
    def log_task_set_history(self,cpu,gpu,task_component,dag,finish_time):
        fused_kernels = task_component.get_kernel_ids_sorted(dag)
        task_list = task_component.get_free_kernels(dag)
        kernel_id_str = fused_kernels[0]
        fused_kernel_ids_str = '-'.join(map(str,fused_kernels))
        job_id_str = dag.job_id
        dag_id_str = dag.dag_id
        start_time = task_list[0].start_time
        for t in task_list:
            start_time = min(start_time, t.start_time)

        platform = ""
        device_id = "-1"
        task_ids=task_component.get_kernel_ids()
        expected_time = 0.0
        ex_cpu, ex_gpu = self.ex_map
        if (cpu == -1 and gpu != -1):
            platform = "gpu"
            device_id = gpu
            
        elif (cpu != -1 and gpu == -1):
            platform = "cpu"
            device_id = cpu
            
        
        terminal_task = 0
        if len(dag.get_task_component_children(task_component)) == 0:
            terminal_task = 1



        # print str(dag_id_str)+","+str(kernel_id_str)+","+str(fused_kernel_ids_str)+","+str(parent_ids_str)+","+str(platform)+","+str(device_id)
        
        start_time_str = task_component.start_time
        history = str(dag_id_str)+","+str(job_id_str)+","+str(fused_kernel_ids_str)+","+str(platform)+","+str(device_id)+","+str(start_time)+","+str(finish_time)+","+str(terminal_task)+"\n"
        
        self.task_set_history.append(history)

    def suspend(self, dev_type, device_id, task_component):
        # print "SUSPEND"
        self.waiting_tasks[dev_type][device_id] = task_component


    def resume(self, task_component, task):
        # print "RESUME"
        dev_type = ""
        device_id = ""
        for device in ['cpu', 'gpu']:
            for dev_id in range(0, len(self.waiting_tasks[device])):
                if self.waiting_tasks[device][dev_id] == task_component:
                    dev_type = device
                    device_id = dev_id
                    self.waiting_tasks[device][dev_id] = None
        task_component.local_frontier.append(task)
        next_task = task_component.local_frontier.popleft()
        dag = self.dags[task_component.dag_id]
        if self.adas:
            dag.reduce_adas_execution_time(next_task, task_component)
        else:
            dag.reduce_execution_time(next_task, task_component)
        next_task.start_time = self.time_stamp
        # print "SCHEDULE: next task ",
        # print str(next_task.id) + " of " + str(dag.dag_id),
        # print " of task component id",
        # print task_component.id,
        # print " ",
        # print next_task.start_time,
        # print " ",
        # print next_task.projected_ex_time
        if dev_type == "gpu":
            self.currently_executing['gpu'].append((next_task, dag, device_id))
            # heapq.heappush(self.available_device_lookahead['gpu'], next_task.projected_ex_time)
            heapq.heappush(self.available_device_lookahead['gpu'], next_task.estimated_ex_time)
        else:
            self.currently_executing['cpu'].append((next_task, dag, device_id)) 
            heapq.heappush(self.available_device_lookahead['cpu'], next_task.projected_ex_time)
            heapq.heappush(self.available_device_lookahead['cpu'], next_task.estimated_ex_time)



        # dev_type = ""
        # device_id = ""
        # for device in ['cpu', 'gpu']:
        #     for dev_id in range(0, len(self.waiting_tasks[device])):
        #         if self.waiting_tasks[device][dev_id] == task_component:
        #             dev_type = device
        #             device_id = dev_id
        #             self.waiting_tasks[device][dev_id] = None
        #
        # dag = self.dags[task_component.dag_id]
        #
        # print "RESUME " + str(dev_type) + str(device_id)
        # if dev_type == "gpu":
        #     if len(task_component.sorted_list) == 0:
        #         print "Task Component Finished"
        #         self.ready_queue['gpu'].append(device_id)
        #         self.mGPU += 1
        #     else:
        #         next_task = task_component.sorted_list[0]
        #         if not dag.are_parents_finished(next_task):
        #             print "SUSPENDING GPU " + str(device_id)
        #             self.suspend(dev_type, device_id, task_component)
        #             return
        #         del task_component.sorted_list[0]
        #         dag.reduce_execution_time(next_task, task_component)
        #         next_task.start_time = self.time_stamp
        #         print "next task ",
        #         print str(next_task.id) + " of " + str(dag.dag_id),
        #         print " of task component id",
        #         print task_component.id,
        #         print " ",
        #         print next_task.start_time,
        #         print " ",
        #         print next_task.projected_ex_time
        #         self.currently_executing['gpu'].append((next_task, dag, device_id))
        #         heapq.heappush(self.available_device_lookahead['gpu'], next_task.projected_ex_time)
        #
        # else:
        #     if len(task_component.sorted_list) == 0:
        #         print "Task Component Finished"
        #         self.ready_queue['cpu'].append(device_id)
        #         self.nCPU += 1
        #     else:
        #         next_task = task_component.sorted_list[0]
        #         if not dag.are_parents_finished(next_task):
        #             print "SUSPENDING CPU " + str(device_id)
        #             self.suspend(dev_type, device_id, task_component)
        #             return
        #         del task_component.sorted_list[0]
        #         dag.reduce_execution_time(next_task, task_component)
        #         next_task.start_time = self.time_stamp
        #         print "next task ",
        #         print str(next_task.id) + " of " + str(dag.dag_id),
        #         print " of task component id",
        #         print task_component.id,
        #         print " ",
        #         print next_task.start_time,
        #         print " ",
        #         print next_task.projected_ex_time
        #         self.currently_executing['cpu'].append((next_task, dag, device_id))
        #         heapq.heappush(self.available_device_lookahead['cpu'], next_task.projected_ex_time)

    def get_waiting_tasks(self):
        gpu_tasks = self.waiting_tasks['gpu']
        cpu_tasks = self.waiting_tasks['cpu']

        for device in gpu_tasks:
            if gpu_tasks[device] is not None:
                pass
                # print "SCHEDULE: Suspended Status GPU",device, gpu_tasks[device].get_kernel_ids(),gpu_tasks[device].dag_id

        for device in cpu_tasks:
            if cpu_tasks[device] is not None:
                pass
                # print "SCHEDULE: Suspended Status CPU",device, cpu_tasks[device].get_kernel_ids(),cpu_tasks[device].dag_id

    def update_finished_task_set(self, executing_task, dev_type):

        # Getting task, task component, dag and device ids

        task, dag, device_id = executing_task
        # print "Current Task Component ID ",

        task_component = dag.task_components[task.id]
        # print task_component.id
        
        # Bookkeeping status of task parameters and frontier and finished status
        if not self.adas:
            task.finish_time = self.time_stamp
        else:
            task.finish_time = task.start_time + task.execution_time
        finish_time = task.finish_time
        task.is_finished = True
        # print "TASK_FINISHED ", task.id
        # print "SCHEDULE_Updating finished tasks for task", task.id,"with local deadline", task.rank, "of", task_component.get_kernel_ids(), "of DAG", dag.dag_id, "with deadline", dag.deadline, "START AND FINISH", task.start_time, task.finish_time
        heapq.heappop(self.available_device_lookahead[dev_type])
        dag.finished_tasks.append(task)

        # Checking whether task component has finished

        task_component.number_of_tasks -= 1
        cpu = -1
        gpu = -1
        if task_component.number_of_tasks == 0:
            # print "SCHEDULE: TASK COMPONENT finished",
            # print task_component
            if dev_type == "gpu":
                gpu = device_id
                # print "FINISH GPU: Task Component Finished"
                if len(self.dispatch_queue['gpu'][device_id]) > 0:
                    tc = self.dispatch_queue['gpu'][device_id].popleft()
                    self.mGPU += 1
                    self.dispatch_task_set(-1,device_id,tc,self.dags[tc.dag_id])
                else:
                    self.ready_queue['gpu'].append(device_id)
                    self.mGPU += 1
            else:
                cpu = device_id
                if len(self.dispatch_queue['cpu'][device_id]) > 0:
                    tc = self.dispatch_queue['cpu'][device_id].popleft()
                    self.nCPU += 1
                    # print "Popped from CPU queue", tc
                    self.dispatch_task_set(device_id,-1,tc,self.dags[tc.dag_id])
                # print "FINISH CPU: Task Component Finished"
                else:
                    self.ready_queue['cpu'].append(device_id)
                    self.nCPU += 1
            self.log_task_set_history(cpu,gpu,task_component,dag,finish_time)
            task_component.is_finished=True

        # Updating device history

        dev_history = (device_id, task.name, task.dag_id, task.start_time, task.finish_time, task.finish_time - task.start_time)

        if dev_type == "gpu":
            self.device_history['gpu'].append(dev_history)
            # print "Updating GPU Device History " + str(device_id)
        else:
            self.device_history['cpu'].append(dev_history)
            # print "Updating CPU Device History " + str(device_id)

        # Successor augmentation if all of its parents are finished
        # 1) Successor is not a supertask and does not belong to current component ---> Add to frontier
        # 2) Successor belongs to current compoent --> Add to local frontier
        # 3) Successor belongs to a different task component ---> If component is suspended, resume
        # 4) Successor beleongs to a different task component and is supposed to be scheduled ---> dispatch directly to specified device

        # print "SCHEDULE: Investigating successors of task" + str(task.id) + " of dag" + str(dag.dag_id)

        for i in dag.skeleton.successors(task.id):
            succ_task = dag.tasks[i]

            # All parents of successor task has finished
            # print "TASK_DISPATCHED_INVESTIGATE ", succ_task.id, task.id
            if dag.are_parents_finished(succ_task):
                # print "."
                # print "SCHEDULE_TASK_DISPATCHED_PARENTS_FINISHED (Successor, Current Task)", succ_task.id, task.id
                succ_task_component = dag.task_components[succ_task.id]
                # print "SUSPENDED_TASKS", self.waiting_tasks
                self.get_waiting_tasks()
                # Case 2

                if succ_task_component == task_component:
                    # print "."
                    # print "SCHEDULE: Update local frontier TASK_DISPATCHED_LOCAL_FRONTIER (Successor, Current Task)", succ_task.id, task.id,
                    task_component.local_frontier.append(succ_task)

                # Case 4
                elif succ_task_component.to_be_scheduled:
                    cpu = -1
                    gpu = -1
                    if succ_task_component.future_device_type == "cpu":
                        cpu = succ_task_component.future_device_id
                    else:
                        gpu = succ_task_component.future_device_id
                    self.dispatch_task_set(cpu, gpu, succ_task_component, self.dags[succ_task_component.dag_id])


                # Case 3

                elif succ_task_component in self.waiting_tasks['gpu'].values() or succ_task_component in self.waiting_tasks['cpu'].values():
                    # print "RESUME SUSPENDED TASK"
                    self.resume(succ_task_component, succ_task)

                # Case 1

                elif not succ_task_component.is_supertask():
                    # print "."
                    # print "TASK_DISPATCHED_GLOBAL_FRONTIER ", succ_task.id
                    # print succ_task.name
                    dag.task_components[succ_task.id].get_first_kernel().in_frontier = True
                    # print "UPDATE-FRONTIER", dag.task_components[succ_task.id].get_kernel_names()
                    self.frontier.put(dag.task_components[succ_task.id])
        # print "."
        # print "REMAINING ",
        # dag.get_remaining_tasks()
        # Remove task from local frontier and add to currently executing

        if task_component.is_supertask():
            # print "SCHEDULE_SUSPEND_LOCAL_FRONTIER_SIZE of ", task_component.get_kernel_ids(), [task.id for task in task_component.local_frontier]
            if len(task_component.local_frontier) > 0:
                next_task = task_component.local_frontier.popleft()

                next_task.dispatch_step = self.dispatch_step
                self.dispatch_step += 1
                if self.adas:
                    dag.reduce_adas_execution_time(next_task, task_component)
                else:
                    dag.reduce_execution_time(next_task, task_component)
                
                next_task.start_time = self.time_stamp
                # task_component.number_of_tasks -= 1
                # print "SCHEDULE: next task ",
                # print str(next_task.id) + " of " + str(dag.dag_id),
                # print " of task component id",
                # print task_component.id,
                # print " ",
                # print next_task.start_time,
                # print " ",
                # print next_task.projected_ex_time
                # print "TASK_DISPATCHED_LOCAL ", next_task.id
                if dev_type == "gpu":
                    self.currently_executing['gpu'].append((next_task, dag, device_id))
                    # heapq.heappush(self.available_device_lookahead['gpu'], next_task.projected_ex_time)
                    heapq.heappush(self.available_device_lookahead['gpu'], next_task.estimated_ex_time)
                else:
                    self.currently_executing['cpu'].append((next_task, dag, device_id))
                    # heapq.heappush(self.available_device_lookahead['cpu'], next_task.projected_ex_time)
                    heapq.heappush(self.available_device_lookahead['cpu'], next_task.estimated_ex_time)
            elif task_component.number_of_tasks > 0:
                # print "SCHEDULE_SUSPEND_TOTAL_SIZE", task_component.number_of_tasks
                # print "SCHEDULE_SUSPEND_TASK", task_component.get_kernel_ids(), task_component, task_component.dag_id
                self.suspend(dev_type, device_id, task_component)









    
    def update_execution_pool_of_task_sets(self):
        min_execution_time_CPU = 1e10
        min_execution_time_GPU = 1e10
        flag = False
        if (len(self.currently_executing['cpu']) > 0) or (len(self.currently_executing['gpu']) > 0):
            flag = True
        if flag is False:
            print "EMPTY ",
            print self.ready_queue,
            # print self.available_device_lookahead,
            # print self.waiting_tasks,
            # print self.nCPU,self.mGPU


        if (len(self.currently_executing['cpu']) > 0):
            first_task_CPU, dag_CPU, dev_CPU = self.currently_executing['cpu'][0]
            min_execution_time_CPU = first_task_CPU.projected_ex_time
            flag = flag and True
        if (len(self.currently_executing['gpu']) > 0):
            first_task_GPU, dag_GPU, dev_GPU = self.currently_executing['gpu'][0]
            min_execution_time_GPU = first_task_GPU.projected_ex_time
        cpu_index = 0
        gpu_index = 0
        self.prev_time_stamp = self.time_stamp
        self.estimate_prev_time_stamp = self.estimate_time_stamp
        # Obtain minimum execution time for cpu devices
        for i in range(0, len(self.currently_executing['cpu'])):
            task, dag, device_id = self.currently_executing['cpu'][i]
            # if device_id not in self.waiting_tasks['cpu']:
            # print "PROJECTED CPU execution pool update check for task ",
            # print task.name,
            # print " ",
            # print task.start_time,
            # print " + ",
            # print task.projected_ex_time,
            # print " = ",
            # print task.start_time + task.projected_ex_time
            if (min_execution_time_CPU > task.projected_ex_time):
                min_execution_time_CPU = task.projected_ex_time
                cpu_index = i
                # print "PROJECTED task with minimum finishing time on CPU: ",
                # print task.name
        # Obtain minimum execution time for gpu devices
        for i in range(0, len(self.currently_executing['gpu'])):
            task, dag, device_id = self.currently_executing['gpu'][i]
            # if device_id not in self.waiting_tasks['gpu']:

            # print "PROJECTED GPU CPU execution pool update check for task ",
            # print task.name,
            # print " ",
            # print task.start_time,
            # print " + ",
            # print task.projected_ex_time,
            # print " = ",
            # print task.start_time + task.projected_ex_time
            if (
                min_execution_time_GPU > task.projected_ex_time):  # removed task.start_time from equation, since the decision of finishing a task is dependent on current execution time
                min_execution_time_GPU = task.projected_ex_time
                gpu_index = i
                # print "PROJECTED task with minimum finishing time on GPU: ",
                # print task.name

        # print "PROJECTED CPU min time: " + str(min_execution_time_CPU)
        # print "PROJECTED GPU min time " + str(min_execution_time_GPU)

        # print len(self.currently_executing['cpu'])
        # print len(self.currently_executing['gpu'])
        cpu_task = SimTask(), None, 0
        gpu_task = SimTask(), None, 0

        if (min_execution_time_GPU < min_execution_time_CPU):
            executing_task = self.currently_executing['gpu'][gpu_index]
            task, dag, device_id = executing_task
            task.device_id = device_id
            task.device_type = "GPU"
            self.currently_executing['gpu'].pop(gpu_index)
            # Update system time stamp and finishing time of task
            execution_time = task.projected_ex_time
            # print "projected_ex_time ",
            # print task.projected_ex_time
            # print "PROJECTED Updating time stamp of " + str(task.name) + " " + str(self.time_stamp) + " to ",
            # print "GPU_OLD_TIME_STAMP: ", self.time_stamp, task.id
            self.time_stamp = self.time_stamp + execution_time
            # print "GPU_NEW_TIME_STAMP: ", self.time_stamp, task.id
            # self.time_stamp = min_execution_time_GPU
            # print(self.time_stamp)
            gpu_task = task, dag, device_id
            # self.update_finished_task_set(executing_task, "gpu")
        else:
            executing_task = self.currently_executing['cpu'][cpu_index]
            task, dag, device_id = executing_task
            task.device_id = device_id
            task.device_type = "CPU"
            self.currently_executing['cpu'].pop(cpu_index)
            # Update system time stamp and finishing time of task
            execution_time = task.projected_ex_time
            # print "projected_ex_time ",
            # print task.projected_ex_time
            # print "PROJECTED Updating time stamp " + str(task.name) + " " + str(self.time_stamp) + " to ",
            # print "CPU_OLD_TIME_STAMP: ",self.time_stamp, task.id
            self.time_stamp = self.time_stamp + execution_time
            # print "CPU_NEW_TIME_STAMP: ",self.time_stamp, task.id
            self.estimate_time_stamp += task.estimated_ex_time
            # self.time_stamp = min_execution_time_CPU
            # print(self.time_stamp)
            cpu_task = task, dag, device_id
            # self.update_finished_task_set(executing_task, "cpu")

        # Diminish projected execution time as per elapsed time of scheduling engine

        # print "PROJECTED Time stamp: " + str(self.time_stamp)
        # print "PROJECTED Previous Time Stamp: " + str(self.prev_time_stamp)
        elapsed = self.time_stamp - self.prev_time_stamp

        estimated_elapsed = self.estimate_time_stamp - self.estimate_prev_time_stamp
        # print "ESTIMATED_ELAPSED due to finishing of",task.name,elapsed, estimated_elapsed
        # print "Elapsed: " + str(elapsed)
        for i in range(0, len(self.currently_executing['cpu'])):
            task, dag, device_id = self.currently_executing['cpu'][i]
            # print "PROJECTED Diminished time for CPU device ",
            # print i,
            # print "and task: ",
            # print task.name,
            # print " ",
            # print " ",
            # print task.projected_ex_time,
            # print " - ",
            # print elapsed,
            # print " = ",
            task.projected_ex_time = task.projected_ex_time - elapsed
            task.estimated_ex_time -= estimated_elapsed
            # print task.projected_ex_time
        # Diminish projected execution time as per elapsed time of scheduling engine
        for i in range(0, len(self.currently_executing['gpu'])):
            task, dag, device_id = self.currently_executing['gpu'][i]
            # print "PROJECTED Diminished time for GPU device ",
            # print i,
            # print "and task: ",
            # print task.name,
            # print " ",
            # print task.projected_ex_time,
            # print " - ",
            # print elapsed,
            # print " = ",
            task.projected_ex_time = task.projected_ex_time - elapsed
            task.estimated_ex_time = task.estimated_ex_time - estimated_elapsed
            # print task.projected_ex_time
            # print "for elapsed time ",
            # print elapsed

        # Diminished execution time of device waiting times

        # print "AVAILABLE_DEVICE_LOOKAHEAD_BEFORE ",self.available_device_lookahead
        for i in range(0, len(self.available_device_lookahead['gpu'])):
            # self.available_device_lookahead['gpu'][i] -= elapsed
            self.available_device_lookahead['gpu'][i] -= estimated_elapsed

        for i in range(0, len(self.available_device_lookahead['cpu'])):
            self.available_device_lookahead['cpu'][i] -= estimated_elapsed

        # print "AVAILABLE_DEVICE_LOOKAHEAD_AFTER ", self.available_device_lookahead

        # Update finishing of task set only after update of current execution pool execution times for each task

        
        
        ex_cpu, ex_gpu = self.ex_map

        if (min_execution_time_GPU < min_execution_time_CPU):
            gpu_t, x, y = gpu_task
            # print "UPDATE_FINISHED_TASK_SET", gpu_t.id ,"GPU" ,y
            self.update_finished_task_set(gpu_task, "gpu")
            # print lineno(), "UPDATING ", gpu_t.name, gpu_t.feature_vector,ex_gpu[gpu_t.name]
            if gpu_t.feature_vector not in self.log:
                
                self.log[gpu_t.feature_vector] = {}
                self.log[gpu_t.feature_vector]["cpu"] = - 1.0
                self.log[gpu_t.feature_vector]["gpu"] = ex_gpu[gpu_t.name]
            else:
                self.log[gpu_t.feature_vector]["gpu"] = ex_gpu[gpu_t.name]


        else:
            cpu_t, x, y = cpu_task
            # print "UPDATE_FINISHED_TASK_SET", cpu_t.id ,"CPU" ,y
            self.update_finished_task_set(cpu_task, "cpu")
            # print lineno(), "UPDATING ", cpu_t.name, cpu_t.feature_vector,ex_cpu[cpu_t.name]
            if cpu_t.feature_vector not in self.log:
                
                self.log[cpu_t.feature_vector] = {}
                self.log[cpu_t.feature_vector]["gpu"] = - 1.0
                self.log[cpu_t.feature_vector]["cpu"] = ex_cpu[cpu_t.name]
            else:
                self.log[cpu_t.feature_vector]["cpu"] = ex_cpu[cpu_t.name]
        self.makespan = self.time_stamp
        
        if self.online:
            self.release_dags(self.time_stamp)

        return cpu_task,gpu_task

    
    def get_device_history(self):
        print "CPU History"
        for cpu_history in self.device_history['cpu']:
            print cpu_history
        print "GPU History"
        for gpu_history in self.device_history['gpu']:
            print gpu_history

    def get_device_utilization(self):
        devs = {'cpu': [], 'gpu': []}
        dev_utilization = {'cpu': [], 'gpu': []}

        for i in range(0, self.num_CPU_devices):
            d = []
            devs['cpu'].append(d)
        for i in range(0, self.num_GPU_devices):
            d = []
            devs['gpu'].append(d)
        for cpu_history in self.device_history['cpu']:
            device_id, task_name, task_dag_id, task_start_time, task_finish_time, task_execution_time = cpu_history
            time_stats = task_start_time, task_finish_time
            devs['cpu'][device_id].append(time_stats)
        for gpu_history in self.device_history['gpu']:
            device_id, task_name, task_dag_id, task_start_time, task_finish_time, task_execution_time = gpu_history
            time_stats = task_start_time, task_finish_time
            devs['gpu'][device_id].append(time_stats)
        for cpu_device_stats in devs['cpu']:
            sorted(cpu_device_stats, key=lambda x: x[1])
        for gpu_device_stats in devs['gpu']:
            sorted(gpu_device_stats, key=lambda x: x[1])
        # for cpu_device_stats in devs['cpu']:
        #     print cpu_device_stats
        # for gpu_device_stats in devs['gpu']:
        #     print gpu_device_stats
        for dtype in ['cpu', 'gpu']:
            for device_id in range(0, len(devs[dtype])):
                idle_time = 0.0
                if len(devs[dtype][device_id]) > 0:
                    start, final = devs[dtype][device_id][0]
                    for i in range(1, len(devs[dtype][device_id])):
                        s1, f1 = devs[dtype][device_id][i - 1]
                        s2, f2 = devs[dtype][device_id][i]
                        diff = s2 - f1
                        # print diff
                        idle_time += diff
                        final = f2
                    dev_utilization[dtype].append((start, final, idle_time))

        print devs
        for dtype in ['cpu', 'gpu']:
            for device_id in range(0, len(devs[dtype])):
                if len(devs[dtype][device_id]) > 0:
                    start, final, idle_time = dev_utilization[dtype][device_id]
                    utilization = (1.0 - idle_time / (final - start)) * 100
                    print dtype + str(device_id) + " --> Tasks dispatched: " + str(
                        len(devs[dtype][device_id])) + " start: " + str(start) + " finish: " + str(
                        final) + " idle time: " + str(idle_time) + " utilization: " + str(utilization) + "%"

    def dump_schedule_engine_stats(self, filename):
        f = open(filename, "w")
        original_stdout = sys.stdout
        sys.stdout = f
        print "Schedule Makespan: " + str(self.makespan)
        self.get_device_history()
        self.get_device_utilization()
        sys.stdout = original_stdout
        f.close()

    def get_device_history_of_task_components(self):
        print "CPU History"
        for cpu_history in self.device_history['cpu']:
            print cpu_history
        print "GPU History"
        for gpu_history in self.device_history['gpu']:
            print gpu_history

    def plot_gantt_chart(self, title='Gantt Chart', bar_width=0.2, showgrid_x=True, showgrid_y=True, height=600,
                         width=900, ):
        # devs = {"gpu": ['GPU 1', 'GPU 2'], "cpu": ['CPU 1', 'CPU 2']}
        devs = {"gpu": [], "cpu": []}
        for i in range(0, self.mGPU):
            dev_string = "GPU " + str(i)
            devs['gpu'].append(dev_string)
        for i in range(0, self.nCPU):
            dev_string = "CPU " + str(i)
            devs['cpu'].append(dev_string)
        # print devs['gpu'][0]
        xtasks = []
        for dtype in self.device_history:
            for devices in self.device_history[dtype]:
                dev, task_name, task_dag_id, task_start_time, task_finish_time, task_execution_time = devices
                # dev = int(dev)
                print '{} {} {} {}'.format(dev, task_name, task_start_time, task_finish_time)
                # print dtype
                print devs[dtype][dev]
                device = devs[dtype][dev]
                print device
                xtasks.append(dict(Task=device, Start=datetime.datetime.fromtimestamp(task_start_time),
                                   Finish=datetime.datetime.fromtimestamp(task_finish_time),
                                   Name='{} {}'.format(task_name, task_dag_id)))
                fig = ff.create_gantt(xtasks, index_col='Name', show_colorbar=True, group_tasks=True, title=title,
                                      bar_width=bar_width, showgrid_x=showgrid_x, showgrid_y=showgrid_y,
                                      height=height, width=width, colors=AC)
                layout = go.Layout(legend=dict(orientation="h"))

        return fig

def plot_gantt_chart_from_file(device_history, filename, nD, title='Gantt Chart', bar_width=0.2, showgrid_x=True, showgrid_y=True, height=600,
                     width=900):
    # devs = {"gpu": ['GPU 1', 'GPU 2'], "cpu": ['CPU 1', 'CPU 2']}
    import random
    import colorsys
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def save_png(fig, filename):
        fig.savefig(filename)
        print "GANTT chart is saved at %s" % filename

    def get_N_HexCol(N=5):

        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
            hex_out.append("".join(map(lambda x: chr(x).encode('hex'), rgb)))
        return hex_out

    def get_N_random_HexColor(N=5):
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        'rgb(31, 119, 180)'
        indexs = random.sample(range(0, 77), N)
        for i in indexs:
            r = int(ALL_COLORS[i][4:-1].split(",")[0])
            g = int(ALL_COLORS[i][4:-1].split(",")[1])
            b = int(ALL_COLORS[i][4:-1].split(",")[2])
            hex_out.append('#%02x%02x%02x' % (r, g, b))
        return hex_out

    devs = {"gpu": [], "cpu": []}
    for i in range(0, nD):
        dev_string = "GPU " + str(i)
        devs['gpu'].append(dev_string)
    for i in range(0, nD):
        dev_string = "CPU " + str(i)
        devs['cpu'].append(dev_string)
    # print devs['gpu'][0]
    xtasks = []
    xmax = 0
    device_info_list = []
    dev_time = {}
    for dtype in device_history:
        for devices in device_history[dtype]:
            dev, task_name, task_dag_id, task_start_time, task_finish_time, task_execution_time = devices
            kn = dtype + "_" + dev
            task_label = task_name + "(" + task_dag_id + ")"
            kernel_times = [task_label, task_start_time, task_finish_time]
            device_info_list.append(devices)
            if kn not in dev_time:
                dev_time[kn] = []
            if kn in dev_time:
                dev_time[kn].append(kernel_times)
            xmax = max(xmax, task_finish_time)

    colourMap = {}
    # colors = get_N_HexCol(len(device_info_list))
    colors = get_N_random_HexColor(len(device_info_list))

    c = 0


    for k in device_info_list:
        colourMap[k[2]] = colors[c]
        c = c + 1

    # legend_patches = []
    # for kn in colourMap:
    #     patch_color = "#" + colourMap[kn]
    #     legend_patches.append(patches.Patch(color=patch_color, label=str(k[2])))

    fig, ax = plt.subplots(figsize=(20, 10))
    device = 0

    for dev in dev_time:
        for k in dev_time[dev]:
            kname = k[0]
            # patch_color = "#" + colourMap[kname]
            patch_color = colourMap[kname]
            start = k[1]
            finish = k[2]
            y = 5 + device * 5
            x = start
            height = 5
            width = finish - start
            # print kname.split(",")[-1] + " : " + str(x) + "," + str(y) + "," + str(width) + "," + str(height)
            ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=patch_color, edgecolor="#000000",
                                           label=kname.split(",")[-1]))
        device = device + 1
    plt.legend(loc=1)
    ax.autoscale(True)
    x_length = float(get_max(device_info_list))
    ax.set_xlim(0, 1.2 * x_length)
    ax.set_ylim(0, len(dev_time) * 10, True, True)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = ""
    i = 1
    for dev in dev_time:
        labels[i] = (dev)
        i = i + 1

    y_ticks = np.arange(2.5, 2.5 + 5 * (1 + len(dev_time)), 5)

    plt.yticks(y_ticks.tolist(), labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('time ( in second )')
    ax.set_ylabel('devices')
    ax.set_yticklabels(labels)

    save_png(fig, filename)




def plot_gantt_chart_graph(device_history, filename):
    """
    Plots Gantt Chart and Saves as png.
    :param device_history: Dictionary Structure containing timestamps of every kernel on every device
    :type device_history: dict
    :param filename: Name of file where the gantt chart is saved. The plot is saved in gantt_charts folder.
    :type filename: String
    """
    import random
    import colorsys
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def save_png(fig, filename):
        fig.savefig(filename)
        print "GANTT chart is saved at %s" % filename

    def get_N_HexCol(N=5):

        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
            hex_out.append("".join(map(lambda x: chr(x).encode('hex'), rgb)))
        return hex_out

    def get_N_random_HexColor(N=5):
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        'rgb(31, 119, 180)'
        indexs = random.sample(range(0, 77), N)
        for i in indexs:
            r = int(ALL_COLORS[i][4:-1].split(",")[0])
            g = int(ALL_COLORS[i][4:-1].split(",")[1])
            b = int(ALL_COLORS[i][4:-1].split(",")[2])
            hex_out.append('#%02x%02x%02x' % (r, g, b))
        return hex_out

    def list_from_file(file):
        device_info_list = []
        dev_data = open(file, "r")
        for line in dev_data:
            if "HOST_EVENT" in line:
                d_list = line.split(" ")[1:]
                device_info_list.append(d_list)
        return device_info_list

    def list_from_dev_history(dev_history):
        device_info_list = []
        for his in dev_history:
            device_info_list.append(his.split(" ")[1:])
        return device_info_list

    def get_min(device_info_list):
        g_min = Decimal('Infinity')
        for item in device_info_list:
            n = Decimal(min(item[3:], key=lambda x: Decimal(x)))
            if g_min > n:
                g_min = n
        return g_min

    def get_max(device_info_list):
        g_max = -1
        for item in device_info_list:
            x = Decimal(max(item[3:], key=lambda x: Decimal(x)))
            if g_max < x:
                g_max = x
        return g_max

    def normalise_timestamp(device_info_list):
        min_t = get_min(device_info_list)
        for item in device_info_list:
            for i in range(len(item) - 3):
                item[i + 3] = Decimal(item[i + 3]) - min_t
        return device_info_list

    device_info_list = normalise_timestamp(list_from_dev_history(device_history))

    colourMap = {}
    # colors = get_N_HexCol(len(device_info_list))
    colors = get_N_random_HexColor(len(device_info_list))

    c = 0
    dev_time = {}
    for k in device_info_list:
        kn = k[0] + "_" + k[1]

        kernel_times = [k[2], k[3], k[-1]]
        if kn not in dev_time:
            dev_time[kn] = []
        if kn in dev_time:
            dev_time[kn].append(kernel_times)

    for k in device_info_list:
        colourMap[k[2]] = colors[c]
        c = c + 1

    # legend_patches = []
    # for kn in colourMap:
    #     patch_color = "#" + colourMap[kn]
    #     legend_patches.append(patches.Patch(color=patch_color, label=str(k[2])))

    fig, ax = plt.subplots(figsize=(20, 10))
    device = 0
    #print dev_time
    for dev in dev_time:
        for k in dev_time[dev]:
            kname = k[0]
            # patch_color = "#" + colourMap[kname]
            patch_color = colourMap[kname]
            start = k[1]
            finish = k[2]
            y = 5 + device * 5
            x = start
            height = 5
            width = finish - start
            # print kname.split(",")[-1] + " : " + str(x) + "," + str(y) + "," + str(width) + "," + str(height)
            ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=patch_color, edgecolor="#000000",
                                           label=kname.split(",")[-1]))
        device = device + 1
    plt.legend(loc=1)
    ax.autoscale(True)
    x_length = float(get_max(device_info_list))
    ax.set_xlim(0, 1.2 * x_length)
    ax.set_ylim(0, len(dev_time) * 10, True, True)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = ""
    i = 1
    for dev in dev_time:
        labels[i] = (dev)
        i = i + 1

    y_ticks = np.arange(2.5, 2.5 + 5 * (1 + len(dev_time)), 5)

    plt.yticks(y_ticks.tolist(), labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('time ( in second )')
    ax.set_ylabel('devices')
    ax.set_yticklabels(labels)

    save_png(fig, filename)


