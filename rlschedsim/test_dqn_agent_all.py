from engine import *
from rlnetworks_all import *
from argparse import RawTextHelpFormatter


# Number of training updates done uptil now
steps_done = 0

import argparse

def parse_arg(args=None):
    parser = argparse.ArgumentParser(
        description='RL DQN Agent Testing Module', formatter_class=RawTextHelpFormatter)
    
    parser.add_argument('-p','--period',
                        help='Period Configuration File Name',
                        default='period_tests.stats')
    parser.add_argument('-c', '--config',
                        help='Configuration File Name',
                        default='test.pth')    
    parser.add_argument('-m', '--model',
                        help='Model File Name',
                        default='test.pth')
    parser.add_argument('-t', '--trace_f',
                        help='Trace Folder Name',
                        default='sample_trace')
    parser.add_argument('-s', '--stats',
                        help='Statistics Filename',
                        default='sample_statistics.stats')
    parser.add_argument('-mo', '--mode',
                        help="""Type of RL Agent\n
1 -> Prioritized Replay = No \t Double Q Learning = No \t Dueling = No\n
2 -> Prioritized Replay = Yes \t Double Q Learning = No \t Dueling = No\n
3 -> Prioritized Replay = No \t Double Q Learning = Yes \t Dueling = No\n
4 -> Prioritized Replay = Yes \t Double Q Learning = Yes \t Dueling = No\n
5 -> Prioritized Replay = No \t Double Q Learning = No \t Dueling = Yes\n
6 -> Prioritized Replay = Yes \t Double Q Learning = No \t Dueling = Yes\n
7 -> Prioritized Replay = No \t Double Q Learning = Yes \t Dueling = Yes\n
8 -> Prioritized Replay = Yes \t Double Q Learning = Yes \t Dueling = Yes\n""",
                        default='1')
    parser.add_argument('-a', '--activation',
                        help="""Type of Activation Function\n
1 -> ReLU\n
2 -> Sigmoid\n""",
                        default='1')
    parser.add_argument('-dnn', '--dnn_type',
                        help="""Which DNN graph file to use\n
1 -> Normal\n
2 -> Reduced\n""",
                        default='2')
    parser.add_argument('-l', '--loss',
                        help="""Which loss function to use\n
1 -> Huber\n
2 -> Cross Entropy\n""",
                        default='1')
    parser.add_argument('-lr', '--rate',
                        help='Learning Rate',
                        default='0.001')
    
    return parser.parse_args(args)



args = parse_arg(sys.argv[1:])
dnn_t = args.dnn_type
l_type = args.loss
lr = args.rate

DAGTransitionMap = {}
# num_states=10
# num_actions=10
# cfg = {('L',16),('L',num_actions)}
# replay_size = 1000
# BATCH_SIZE = 8
# GAMMA = 1
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200
cfg, params = parse_configuration(args.config)

d1 = False
d2 = False
per = False

typ = args.mode
if typ == '1':
    per = False
    d1 = False
    d2 = False
elif typ == '2':
    per = True
    d1 = False
    d2 = False
elif typ == '3':
    per = False
    d1 = True
    d2 = False
elif typ == '4':
    per = True
    d1 = True
    d2 = False
elif typ == '5':
    per = False
    d1 = False
    d2 = True
elif typ == '6':
    per = True
    d1 = False
    d2 = True
elif typ == '7':
    per = False
    d1 = True
    d2 = True
elif typ == '8':
    per = True
    d1 = True
    d2 = True

num_states,num_actions,replay_size,BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY = tuple(params)
environment_params = (BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY)

print "Prioritized Replay ->",per
print "Double DQN ->",d1
print "Dueling DQN ->",d2
act = args.activation
DQNAgentObject = DQNAgent(cfg,num_states,num_actions,replay_size,per,d2,d1,1,act,l_type,lr,environment_params)

avg_time = 0
count_for_time = 0

################################################################################################################

############# Auxillary Functions Required for getting state and action vectors ###############################

def get_finished_task_dag_pair(cpu_task,gpu_task):
    task_cpu, dag_cpu, device_cpu = cpu_task
    task_gpu, dag_gpu, device_gpu = gpu_task
    
    if dag_cpu is not None:
        return (task_cpu,dag_cpu)
    else:
        return (task_gpu,dag_gpu)

def edf_task_dag_pair(SE):
    
    task_index_in_queue=0
    if len(SE.frontier.queue) == 0:
        return -1,-1
    min=SE.frontier.queue[0].get_first_kernel().rank
    for i in range(0, len(SE.frontier.queue)):
        dispatching_task_component = SE.frontier.queue[i]
        task_rank = SE.frontier.queue[i].get_first_kernel().rank
        if task_rank <= min:
            task_object = SE.frontier.queue[i]
            min = task_rank
            task_index_in_queue = i
    #####################################################
    
    dispatching_task_component = SE.frontier.queue[task_index_in_queue]
    # print "SELECTED TC_GEN of ", dispatching_task_component.get_kernel_ids()
    dag = SE.dags[dispatching_task_component.dag_id]

    return (dispatching_task_component.id, dag.job_id)


def get_state_vector(SE):
    
    state=[0.0, 0.0]
    ready_time = {'cpu':0.0, 'gpu':0.0}
    ex_cpu, ex_gpu = SE.ex_map
    for key in ['cpu', 'gpu']:
        if len(SE.currently_executing[key]) > 0:
            task, dag, device_id = SE.currently_executing[key][0]
            tc = dag.task_components[task.id]
            task_ids = tc.get_kernel_ids()
            if tc.is_supertask():
                ready_time[key] =dag.fused_kernel_timings[key][tuple(sorted(tuple(task_ids)))]
            else:
                ready_time[key] = tc.get_first_kernel().execution_time
            
        else:
            ready_time[key] = 0.0
    
    state[0] += ready_time['cpu']
    state[1] += ready_time['gpu']

    pending_time = {'cpu':0.0, 'gpu':0.0}
    
    for key in ['cpu', 'gpu']:
        
        for tc in SE.dispatch_queue[key]:
            task_ids = tc.get_kernel_ids()
            dag = SE.dags[tc.dag_id]
            if tc.is_supertask():
                pending_time[key] +=dag.fused_kernel_timings[key][tuple(sorted(tuple(task_ids)))]
            else:
                pending_time[key] += tc.get_first_kernel().execution_time                
    
    state[0] += pending_time['cpu']
    state[1] += pending_time['gpu']
    
    prt = SE.get_dag_prt("blevel_wcet")
    # print "REMAINING", prt
    state.extend(prt)
    dr = SE.get_dag_deadline_requirements()
    state.extend(dr)
    # print "GET STATE VECTOR", state
    return state

    

def is_component_finished(c,g):
    t,d=get_finished_task_dag_pair(c,g)
    if d.task_components[t.id].is_finished == True:
        return True
    else:
        return False

def get_transition(c,g,SE,TransitionMap,rates):
    t,d=get_finished_task_dag_pair(c,g)
    edf_t,edf_d=edf_task_dag_pair(SE)
    # print TransitionMap
    tc=d.task_components[t.id]
    state,action=TransitionMap[tc]
    next_state=get_state_vector(SE)
    # print "REWARD", t.finish_time, t.rank
    reward = 0
    
    reward=-(t.finish_time-t.rank)
    rewards.append(reward)
    reward = Tensor([reward])
    
    # print state
    state = FloatTensor(state)
    next_state = FloatTensor(next_state)
    return state,action,next_state, reward

def update_rewards(c,g):
    t,d=get_finished_task_dag_pair(c,g)
    reward = 0
    if d.is_processed():
        reward_flag = (d.finishing_time-(d.deadline+d.tolerance))
        # print "Finishing Time", d.finishing_time, "deadline", d.deadline, "tolerance",d.tolerance
        # import time
        # time.sleep(2)
        if reward_flag > 0:
            reward = -1
        else:
            reward = 1
        for transition in DAGTransitionMap[d.dag_id]:
            transition.reward = reward
        
        DQNAgentObject.rewards_history.append(reward)

def parse_action(a):
    action=a.cpu().numpy()[0][0]
    print "action", action
    global num_actions
    temp = num_actions
    if action < int(temp/2):
        return 'cpu', action
    else:
        return 'gpu', action - int(temp/2) 

################################################################################################################

################### Primary Scheduling Function ############################################################

def schedule_workflows(SE, dags,periods=None,episode=-1):


    # print "Schedule Engine for Task Sets Initiated"
    TransitionMap = {}
    rates, period_values=SE.generate_periodic_schedule(dags, periods,episode=episode)
    SE.adas=True
    
    # print "Setting up device queues"

    for i in range(0, SE.nCPU):
        SE.ready_queue['cpu'].append(i)
    for i in range(0, SE.mGPU):
        SE.ready_queue['gpu'].append(i)
    
    
    # Initialize frontier
    
        
    for dag in SE.dags:
        for task_component in dag.free_task_components:
            task_component.get_first_kernel().in_frontier = True
            SE.frontier.put(task_component)
    
       
  
    
    # Scheduling iteration behaviour
    
    # print "Primary Scheduling Iteration Starts here with " + str(SE.nCPU) + " CPU and " + str(
    #     SE.mGPU) + "GPU devices"

    for i in range(0,len(SE.global_dags)):
        DAGTransitionMap[i]=[]
    
    while (not all_processed(SE.global_dags)):
        
        # DEBUG print statistics
        # print "DAG Pool", len(SE.dags),
        # print [dag.dag_id for dag in SE.dags]
        # print "CURRENT FRONTIER TC_GEN",
        # print [tc.get_kernel_ids() for tc in SE.frontier.queue]
        # print "Iteration Stats: nCPU " + str(SE.nCPU),
        # print " mGPU " + str(SE.mGPU),
        # print " Frontier Size " + str(len(SE.frontier.queue)),
        # print "Currently Executing ", SE.currently_executing,
        # print "Ready Queue", SE.ready_queue,
        # print "Task QUeue", SE.dispatch_queue

        if all_processed(SE.dags) == 1:
        
           if len(SE.arriving_dags) > 0:
                t = SE.dag_arrival_time_stamps[0]
                SE.time_stamp = max(SE.time_stamp,t)
                while t <= SE.time_stamp and len(SE.dag_arrival_time_stamps) > 0:
                    dag = SE.arriving_dags.pop(0)
                    SE.dags.append(dag)
                    dag.starting_time = SE.dag_arrival_time_stamps.pop(0)
                    if len(SE.dag_arrival_time_stamps) > 0:
                        t = SE.dag_arrival_time_stamps[0]
                    for task_component in dag.free_task_components:
                        task_component.get_first_kernel().in_frontier = True
                        SE.frontier.put(task_component)
       
        if (SE.nCPU == 0 and SE.mGPU == 0):
            
        
            c,g=SE.update_execution_pool_of_task_sets()
            if is_component_finished(c,g):
                
                update_rewards(c,g)
                next_state = get_state_vector(SE)
                if per:
                    temp = DQNAgentObject.memory.front()
                    new_transition = Transition(temp.state,temp.action,next_state,temp.reward)
                    DQNAgentObject.memory.tree.update_data(new_transition)
                else:
                    DQNAgentObject.memory.front().next_state=next_state 
                # DQNAgentObject.optimize_model()
              
            
        else:
            if (not SE.frontier.empty()):
                task_index_in_queue = 0
                task_object = SE.frontier.queue[0]
                min_rank=SE.frontier.queue[0].get_first_kernel().rank
                min_rank_task_object = SE.frontier.queue[0]
                current_ranks = []
    
                # Get task with earliest DAG deadline
                #####################################################
    
                for i in range(0, len(SE.frontier.queue)):
                    dispatching_task_component = SE.frontier.queue[i]
                    current_task = SE.frontier.queue[i]
                    current_rank = SE.frontier.queue[i].get_first_kernel().rank
                    
                    # 
                    if min_rank > current_rank:
                        task_object = SE.frontier.queue[i]
                        min_rank = current_rank
                        task_index_in_queue = i
                        min_rank_task_object = SE.frontier.queue[i]

                    elif min_rank == current_rank:
                        if min_rank_task_object.dag_id > current_task.dag_id:
                            task_index_in_queue = i
                            min_rank_task_object = SE.frontier.queue[i]

                # min=SE.frontier.queue[0].get_first_kernel().rank
                # current_ranks = []
    
                # # Get task with earliest DAG deadline
                # #####################################################
    
                # for i in range(0, len(SE.frontier.queue)):
                #     dispatching_task_component = SE.frontier.queue[i]
                #     current_task = SE.frontier.queue[i]
                #     task_rank = SE.frontier.queue[i].get_first_kernel().rank
                #     current_ranks.append(task_rank)
                    
                #     if current_task.dag_id < task_object.dag_id:
                #         task_object = SE.frontier.queue[i]
                #         min = task_rank
                #         task_index_in_queue = i

                #     elif task_rank < min:
                #         task_object = SE.frontier.queue[i]
                #         min = task_rank
                #         task_index_in_queue = i
                #####################################################
        
                dispatching_task_component = SE.frontier.queue[task_index_in_queue]
                
    
                dag = SE.dags[dispatching_task_component.dag_id]
                dispatching_task_set = None

                state= get_state_vector(SE)
                state_tensor=FloatTensor(state)
                action,time_taken=DQNAgentObject.get_action(state_tensor)
                global avg_time
                global count_for_time
                new_avg = avg_time*count_for_time
                new_avg += time_taken
                count_for_time += 1
                new_avg /= count_for_time
                avg_time = new_avg
                TransitionMap[dispatching_task_component]=(state,action)              

                cpu = -1
                gpu = -1

                print "state",state
                dispatch_device, depth = parse_action(action)
                
                
                if (dispatch_device == "gpu" ):
                    if depth > 0:
                        dispatching_task_set = dag.kernel_fusion(dispatching_task_component,depth, SE.frontier)
                    else:
                        dispatching_task_set = dispatching_task_component
                    dispatching_task_set.Class="TEN"
                    
                    if SE.mGPU > 0:
                        gpu = SE.ready_queue['gpu'].popleft()
                        dispatching_task_component.get_first_kernel().in_frontier=False
                        SE.frontier.queue.remove(dispatching_task_component)
                        # print "SCHEDULE: Dispatching " + str(list(dispatching_task_set.get_kernel_ids())) + " of DAG " + str(
                            # dispatching_task_set.dag_id) + " to GPU " + str(gpu)  
                        SE.dispatch_task_set(cpu, gpu, dispatching_task_set, SE.dags[dispatching_task_set.dag_id])
                    else:
                        SE.dispatch_queue['gpu'].append(dispatching_task_set)
                        dispatching_task_component.get_first_kernel().in_frontier=False
                        SE.frontier.queue.remove(dispatching_task_component)
                                                
                         

                elif dispatch_device == 'cpu':
                    if depth > 0:
                        dispatching_task_set = dag.kernel_fusion(dispatching_task_component,depth, SE.frontier)
                    else:
                        dispatching_task_set = dispatching_task_component
                                      
                   
                    dispatching_task_set.Class="ZERO"
                    if SE.nCPU > 0:
                        
                        cpu = SE.ready_queue['cpu'].popleft()
                        dispatching_task_component.get_first_kernel().in_frontier=False
                        SE.frontier.queue.remove(dispatching_task_component)
                        # print "SCHEDULE: Dispatching " + str(list(dispatching_task_set.get_kernel_ids())) + " of DAG " + str(
                            # dispatching_task_set.dag_id) + " to CPU" + str(cpu) 
                        SE.dispatch_task_set(cpu, gpu, dispatching_task_set, SE.dags[dispatching_task_set.dag_id])
                    else:
                        SE.dispatch_queue['cpu'].append(dispatching_task_set)
                        dispatching_task_component.get_first_kernel().in_frontier=False
                        SE.frontier.queue.remove(dispatching_task_component)
                next_state = get_state_vector(SE)
                transition_tuple = Transition(state,action,next_state,0)
                DAGTransitionMap[dag.dag_id].append(transition_tuple)
                if per:
                    max_prio_val = DQNAgentObject.memory.tree.get_max_prio()
                    DQNAgentObject.memory.push(max_prio_val, transition_tuple)
                else:
                    DQNAgentObject.memory.push(transition_tuple)

            else:
                c,g=SE.update_execution_pool_of_task_sets()
                if is_component_finished(c,g):
                    # print "UPDATE MEMORY"
                    # state,action,next_state,reward=get_transition(c,g,SE,TransitionMap,rates)
                    update_rewards(c,g)
                    next_state = get_state_vector(SE)
                    if per:
                        temp = DQNAgentObject.memory.front()
                        new_transition = Transition(temp.state,temp.action,next_state,temp.reward)
                        DQNAgentObject.memory.tree.update_data(new_transition)
                    else:
                        DQNAgentObject.memory.front().next_state = next_state
                    # DQNAgentObject.optimize_model()
        SE.makespan = SE.time_stamp
    return period_values

############## Generate Period Configuration Functions #######################################

def episode_set_periods():

    possible_periods_task1 = [485,970,1455]
    possible_periods_task2 = [970,1940,2910]
    possible_periods_task3 = [485,970,1455]
    possible_periods_task4 = [970,1940,2910]

    possible_periods_hyper_cycle = []

    episode = 0
    for p in possible_periods_task1:
        for q in possible_periods_task2:
            for r in possible_periods_task3:
                for s in possible_periods_task4:
                    l = [p,q,r,s]
                    print episode, l
                    episode += 1
                    
    # random.shuffle(possible_periods_hyper_cycle)
    return possible_periods_hyper_cycle

def generate_periods(wcet):  # Not used anymore
    possible_periods = [wcet*1,wcet*2,wcet*3,wcet*4]
    p = []
    for i in range(4):
        p.append(random.choice(possible_periods))
    return p

def generate_task_set_periods(period_configuration_file):

    # possible_periods_task1 = [28,28,42]
    # possible_periods_task2 = [12,24,36]
    # possible_periods_task3 = [14,28,42]
    # possible_periods_task4 = [12,24,36]

    # possible_periods_task1 = [30,45,90]
    # possible_periods_task2 = [30,45,90]
    # possible_periods_task3 = [30,45,90]
    # possible_periods_task4 = [30,45,90]

    contents = open(period_configuration_file,'r').readlines()
     
    
    possible_periods_task1 = eval(contents[0].strip("\n").split("=")[1])
    possible_periods_task2 = eval(contents[1].strip("\n").split("=")[1])
    possible_periods_task3 = eval(contents[2].strip("\n").split("=")[1])
    possible_periods_task4 = eval(contents[3].strip("\n").split("=")[1])
    
    #possible_periods_task1 = [490,735,1470]
    #possible_periods_task2 = [980,1470,2940]
    #possible_periods_task3 = [490,735,1470]
    #possible_periods_task4 = [980,1470,2940]

    possible_periods_hyper_cycle = []
    # print "Number of runs", args.numruns
    for p in possible_periods_task1:
        for q in possible_periods_task2:
            for r in possible_periods_task3:
                for s in possible_periods_task4:
                    l = [p,q,r,s]
                    possible_periods_hyper_cycle.append(l)
                  
    return possible_periods_hyper_cycle

##########################################################################################################

if __name__ == '__main__':
    
    # episode_set_periods()
    
    model_file = "Results/" + args.model
    trace_folder = "Results/" + args.trace_f
    statistics_file = "Results/" + args.stats
    os.system("mkdir Results/" + args.trace_f)
    os.system("mkdir Results/"+args.trace_f+"/Pos")
    os.system("mkdir Results/"+args.trace_f+"/Neg")
    period_configuration = args.period
    global_map = {}
    global_file_list = open("global_map.txt", "r").readlines()
    global_map = make_dict(global_file_list)
    kernel_info_list = [obtain_kernel_info(key) for key in global_map]
    ex_cpu = {}
    ex_gpu = {}
    ex_map = (ex_cpu, ex_gpu)
    DC = DAGCreator()
    
    if dnn_t == 1:
        input_file1="ADAS_Graphs/dnn.graph"
    else:
        input_file1="ADAS_Graphs/dnn_reduced.graph"
    input_file2="ADAS_Graphs/cnn.graph"

    dag1 = DC.create_adas_job_from_file_with_fused_times(input_file1, 0, global_map, ex_map, 0,0.0)
    dag2 = DC.create_adas_job_from_file_with_fused_times(input_file2, 1, global_map, ex_map, 1,0.0)
    dag3 = DC.create_adas_job_from_file_with_fused_times(input_file1, 2, global_map, ex_map, 2,0.0)
    dag4 = DC.create_adas_job_from_file_with_fused_times(input_file2, 3, global_map, ex_map, 3,0.0)


    dags=[dag1, dag2, dag3, dag4]

    num_jobs = 4
   
    DQNAgentObject.policy_net.load_state_dict(torch.load(model_file))

    mean_rewards =  []
    tardiness_values = []
    deadlines_missed_values = []
    avg_makespan_values = []
    period_values = []
    lateness_values=[]
    max_lateness_values=[]
    percent_lateness_values=[]
    task_periods = generate_task_set_periods(args.period)
    num_episodes = len(task_periods)
    pos = 0 
    neg = 0
    zeros_t = 0
    
    for i_episode in range(num_episodes):
        
		print "EPISODE: ",i_episode
		#ex_map -> execution map key -> task_name value -> time
		SE = ScheduleEngine(1, 1, "local_deadline", ex_map,True,global_map,num_jobs=num_jobs)
		periods = task_periods[i_episode]
		period_values.append(periods)
		schedule_workflows(SE,dags,periods, i_episode)
		# schedule_workflows(SE,dags,periods)

		print "MAKESPAN", SE.makespan 
		avg_lateness = SE.calculate_avg_lateness()
		max_lateness = SE.calculate_max_lateness()

		SE.calculate_avg_makespan()
		print "AVG MAKESPAN", SE.avg_makespan

		avg_makespan_values.append(SE.avg_makespan)
		tardiness,deadlines_missed = SE.calculate_avg_tardiness()
		print "DEADLINES MISSED", deadlines_missed        
		print "MAX LATENESS",max_lateness
		tardiness_values.append(tardiness)
		deadlines_missed_values.append(float(deadlines_missed)/len(SE.dags))
		lateness_values.append(avg_lateness)
		max_lateness_values.append(max_lateness)
		percent_lateness = SE.calculate_percent_lateness()
		percent_lateness_values.append(percent_lateness)

		file_name_dispatch = "dispatch_history_" + str(i_episode)+".stats"
		file_name_tasks = "task_set_history_" + str(i_episode)+".stats"
		if max_lateness > 0.0:
			file_name_dispatch = "Results/"+args.trace_f+"/Neg/"+file_name_dispatch
			file_name_tasks = "Results/"+args.trace_f+"/Neg/"+file_name_tasks
			SE.get_dispatch_history(file_name_dispatch)
			SE.get_task_set_history(file_name_tasks)
			neg +=1
		else:
			file_name_dispatch = "Results/"+args.trace_f+"/Pos/"+file_name_dispatch
			file_name_tasks = "Results/"+args.trace_f+"/Pos/"+file_name_tasks
			SE.get_dispatch_history(file_name_dispatch)
			SE.get_task_set_history(file_name_tasks)
			pos +=1

			if not max_lateness < 0.0:
				zeros_t += 1

        # command = "scp "+reward_filename + " anirban@10.5.19.217:~/Programs/"
        # os.system(command)
    print "Accuracy: ", float(pos) /(pos+neg)
    print "Pos: ", pos-zeros_t
    print "Neg: ", neg
    print "Zeros: ", zeros_t
    # ft = open("Results/experiments_4_9_19_2/Acc.stats",'a+')
    # ft.write(model_file+"---> Neg = "+str(neg)+",Pos = "+str(pos-zeros_t)+",Zeros = "+str(zeros_t)+",Accuracy = "+str(float(pos) /(pos+neg))+"\n")
    # ft.close()
    f = open(statistics_file,'w')
    for i in range(num_episodes):
        stats_string = str(task_periods[i]) + ":" + str(deadlines_missed_values[i]) + "," + str(max_lateness_values[i])+ "," + str(tardiness_values[i]) + ","+ str(avg_makespan_values[i]) + ","+ str(lateness_values[i])+"\n"
        f.write(stats_string)
    f.close()

    global avg_time
    f = open(statistics_file[:-6] + "_forward_pass_time.stats", "w")
    print >> f, "avg_time =",avg_time
    f.close()

