from engine_baseline import *

from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import math

import argparse 

def parse_arg(args=None):
    parser = argparse.ArgumentParser(
        description='Global EDF Fusion Scheduler')
    parser.add_argument('-p','--period',
                        help='Period Configuration File Name',
                        default='period_tests.stats')
    parser.add_argument('-m', '--model',
                        help='Model File Name',
                        default='test.pth')
    parser.add_argument('-t', '--trace_f',
                        help='Trace Folder Name',
                        default='sample_trace')
    parser.add_argument('-s', '--stats',
                        help='Statistics Filename',
                        default='sample_statistics.stats')
    parser.add_argument('-nr', '--numruns',
                        help='Number of Runs for Each Period Configuration',
                        default='100')
    parser.add_argument('-ne', '--numepochs',
                        help='Number of Epochs for entire RL Training',
                        default='1')
    
    return parser.parse_args(args)


args = parse_arg(sys.argv[1:])

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

state_size=10
hidden_size=512
num_actions=5

steps_done = 0

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        a = random.sample(self.memory, batch_size)
        # print(a)
        return a

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(DQN, self).__init__()                    # Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer:  (input data) -> (hidden node)
        self.relu = nn.ReLU()                          # Non-Linear ReLU Layer: max(0,x)
        # self.fc2 = nn.Linear(hidden_size, hidden_size) # 2nd Full-Connected Layer:  (hidden node) -> (output class)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        # self.softmax = nn.Softmax(dim=num_actions)

    def forward(self, x):                              # Forward pass: stacking each layer together
        # print "FORWARD PASS"
        x = x.view(-1, state_size)
        out = self.fc1(x)
        # print out.shape
        out = self.relu(out)
        # print out.shape
        out = self.fc2(out)

        # out = self.relu(out)
        # out = self.fc3(out)
     
        # print out.shape
        # out = self.softmax(out)
        return out



BATCH_SIZE = 32
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 5

rewards = []
policy_net = DQN(state_size,hidden_size,num_actions)
target_net = DQN(state_size,hidden_size,num_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if use_cuda:
    policy_net.cuda()
    target_net.cuda()

# optimizer = optim.RMSprop(policy_net.parameters())
optimizer = optim.Adam(policy_net.parameters(),lr=0.01)
memory = ReplayMemory(1000)

# class RL(object):
    
#     def __init__():

def optimize_model():
    print("Optimize Model called!")
    if len(memory) < BATCH_SIZE:
        return
    # print "OPTIMIZE"
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))
    # print(batch.next_state)
    #print(batch)
    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    # print state_batch.shape,action_batch.shape,reward_batch.shape
    # print policy_net(state_batch).shape
    # print(policy_net(state_batch))
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    # print(next_state_values[non_final_mask])
    # print(non_final_next_states)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Undo volatility (which was used to prevent unnecessary gradients)
    expected_state_action_values = Variable(expected_state_action_values.data)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


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


def get_currently_executing_queue_times(currently_executing, nCPU, mGPU):
    queue_execution_times = {'cpu': {}, 'gpu': {}}
    for i in range(0,nCPU):
        queue_execution_times['cpu'][i] = 0.0
    for i in range(0,mGPU):
        queue_execution_times['gpu'][i] = 0.0

    for key in ['cpu', 'gpu']:
        if len(currently_executing[key]) > 0:
            task, dag, device_id = currently_executing[key][0]
            queue_execution_times[key][device_id] = task.projected_ex_time

    return queue_execution_times


def get_dispatch_queue_times(dispatch_queue, nCPU, mGPU, ex_map):
    ex_cpu, ex_gpu = ex_map
    queue_execution_times = {'cpu': {}, 'gpu': {}}
    for i in range(0, nCPU):
        queue_execution_times['cpu'][i] = 0.0
    for i in range(0, mGPU):
        queue_execution_times['gpu'][i] = 0.0   
    
    for i in range(0, nCPU):
        if len(dispatch_queue['cpu'][i]) > 0:
            for tc in dispatch_queue['cpu'][i]:
                task_name = tc.get_first_kernel().name
                queue_execution_times['cpu'][i] += ex_cpu[task_name]
    
    for i in range(0, mGPU):
        if len(dispatch_queue['gpu'][i]) > 0:
            for tc in dispatch_queue['gpu'][i]:
                task_name = tc.get_first_kernel().name
                queue_execution_times['gpu'][i] += ex_gpu[task_name]

    return queue_execution_times    


def get_state_vector(currently_executing,task_id,job_id,rates):
    state=[]
    
    for key in ['cpu', 'gpu']:
        if len(currently_executing[key]) > 0:
            task, dag, device_id = currently_executing[key][0]
            state.append(task.projected_ex_time)
        else:
            state.append(0.0)
    
    prt = SE.get_dag_prt("blevel_wcet")
    # print "REMAINING", prt
    state.extend(prt)
    state.extend(rates)
    # print "GET STATE VECTOR", state
    return state

def select_action(state):
    
    global steps_done
    
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        action =  policy_net(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        action = LongTensor([[random.randrange(num_actions)]])

    # print(action)
    return action
    

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
    next_state=get_state_vector(SE.currently_executing,edf_t,edf_d,rates)
    # print "REWARD", t.finish_time, t.rank
    reward = 0
    
    reward=-(t.finish_time-t.rank)
    rewards.append(reward)
    reward = Tensor([reward])
    
    # print state
    state = FloatTensor(state)
    next_state = FloatTensor(state)
    return state,action,next_state, reward

def parse_action(a):
    action=a.cpu().numpy()[0][0]
    # print "action", action
    if action == 0:
        return 'cpu', 0
    if action == 1:
        return 'gpu', 0
    if action > 1:
        return 'gpu', action-1


def schedule_workflows(SE, dags,periods=None,dag_history="",episode=-1):


    # print "Schedule Engine for Task Sets Initiated"
    TransitionMap = {}
    
    rates, period_values=SE.generate_periodic_schedule(dags,periods,dag_history,episode)
    
    # print "Rechecking task component ids"
    dt_max_value = 0.0

    SE.adas=True
    # print "Setting up device queues"

    for i in range(0, SE.nCPU):
        SE.ready_queue['cpu'].append(i)
    for i in range(0, SE.mGPU):
        SE.ready_queue['gpu'].append(i)


    ex_cpu, ex_gpu = SE.ex_map
 
    r = Ranker(SE.dags, SE.ex_map)

    r.compute_rank_dags("blevel_wcet", SE.num_CPU_devices, SE.num_GPU_devices)
    r.reset_ranks("blevel_wcet")
    r.compute_rank_dags(SE.rank_name, SE.num_CPU_devices, SE.num_GPU_devices)
    
    # Initialize frontier
    
    # print "Initializing Frontier of Task Components"

	#Not relevant here    
    for dag in dags:
        if dt_max_value < dag.get_max_dt():
            dt_max_value = dag.get_max_dt()
        dag.assign_device_lookahead(SE.available_device_lookahead)
        dag.assign_currently_executing(SE.currently_executing)

	    
    for dag in SE.dags:
        for task_component in dag.free_task_components:
            task_component.get_first_kernel().in_frontier = True
            SE.frontier.put(task_component)
    
    # ################## Testing###################
    
    # task_index_in_queue = 0
    # task_object = None
    # min = 0
    # for i in range(0, len(SE.frontier.queue)):
    #     dispatching_task_component = SE.frontier.queue[i]
    #     task_rank = SE.frontier.queue[i].get_first_kernel().rank
    #     if task_rank < min:
    #         task_object = SE.frontier.queue[i]
    #         min = task_rank
    #         task_index_in_queue = i
    # dispatching_task_component = SE.frontier.queue[task_index_in_queue]
    # # print "SELECTED TC_GEN of ", dispatching_task_component.get_kernel_ids()
    # dag = SE.dags[dispatching_task_component.dag_id]
    # state= get_state_vector(SE.currently_executing,dispatching_task_component.get_first_kernel().id,dag.job_id,rates)
    # print state
    # action=select_action(state)    
    # print parse_action(action)
    
    
    # print len(SE.dags)
    # print [tc.get_kernel_ids() for tc in SE.frontier.queue]
    # dispatching_task_component = SE.frontier.queue[0]
    # dag = SE.dags[dispatching_task_component.dag_id]
    # dispatching_task_set = dag.kernel_fusion(dispatching_task_component,1, SE.frontier)
    # print "FUSION" + str(list(dispatching_task_set.get_kernel_ids_sorted(dag))) + " of DAG " + str( dispatching_task_set.dag_id)
    # print dag.fused_kernel_timings
    # dag.reduce_adas_execution_time(dispatching_task_set.get_kernels_sorted(dag)[-1],dispatching_task_set)
    
    #############################################
    
    
    
    # Scheduling iteration behaviour
    # print "Primary Scheduling Iteration Starts here with " + str(SE.nCPU) + " CPU and " + str(
    #     SE.mGPU) + "GPU devices"


    while (not all_processed(SE.global_dags)):
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
            # print "New DAG has arrived"
           
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
            
           
            
        else:
            
            if (not SE.frontier.empty()):
                task_index_in_queue = 0
                task_object = SE.frontier.queue[0]
                
                min_rank=SE.dags[task_object.dag_id].deadline 
                
                # Get task with earliest DAG deadline
                #####################################################
                for i in range(0, len(SE.frontier.queue)):
                    
                    current_task = SE.frontier.queue[i]
                    task_rank = SE.dags[current_task.dag_id].deadline 
                    if task_rank < min_rank:
                        min_rank = task_rank
                        task_index_in_queue = i


                #####################################################
                # print "SCHEDULE: ", current_ranks
                dispatching_task_component = SE.frontier.queue[task_index_in_queue]
                
                # print "SELECTED TC_GEN of ", dispatching_task_component.get_kernel_ids(), 
                dag = SE.dags[dispatching_task_component.dag_id]
                # print "DAG ", dag.dag_id, "with local deadline", min
                dispatching_task_set = dispatching_task_component
                task_name = dispatching_task_component.get_first_kernel().name
                task_id = dispatching_task_component.get_first_kernel().id
               
                
                currently_executing_times = get_currently_executing_queue_times(SE.currently_executing, SE.num_CPU_devices, SE.num_GPU_devices)
                dispatch_queue_times = get_dispatch_queue_times(SE.dispatch_queue,SE.num_CPU_devices,SE.num_GPU_devices,SE.ex_map)

                ready_times = {'cpu': {}, 'gpu': {}}

                for i in range(0, SE.num_CPU_devices):
                    ready_times['cpu'][i] = 0.0

                for i in range(0, SE.num_GPU_devices):
                    ready_times['gpu'][i] = 0.0
                    
                for key in ['cpu','gpu']:
                    for i in ready_times[key].keys():
                        # print key,i,currently_executing_times[key][i], dispatch_queue_times[key][i]
                        ready_times[key][i] = currently_executing_times[key][i] + dispatch_queue_times[key][i]

                min_ready_time = float('inf')
                dispatch_device = ""
                device_id = 0
                depth = 0
                for key in ['cpu','gpu']:
                    for i in ready_times[key].keys():
                        task_time,d = dag.get_min_fused_timing_and_depth(dispatching_task_component,key)
                        
                        if min_ready_time > ready_times[key][i] + task_time:
                            dispatch_device = key
                            device_id = i
                            min_ready_time = ready_times[key][i] + task_time
                            depth = d

                cpu = -1
                gpu = -1
                print "Selected ", dispatch_device, device_id, depth
                if depth > 0:
                    dispatching_task_set = dag.kernel_fusion(dispatching_task_component,depth, SE.frontier)
                else:
                    dispatching_task_set = dispatching_task_component

                if (dispatch_device == "gpu" ):
                    
                    dispatching_task_set.Class="TEN"                    
                    if device_id in SE.ready_queue['gpu']:
                        SE.ready_queue['gpu'].remove(device_id)
                        gpu = device_id
                        # SE.frontier.get()
                        dispatching_task_component.get_first_kernel().in_frontier=False
                        SE.frontier.queue.remove(dispatching_task_component)
                        # print "SCHEDULE: Dispatching " + str(list(dispatching_task_set.get_kernel_ids())) + " of DAG " + str(
                            # dispatching_task_set.dag_id) + " to GPU " + str(gpu)  
                        SE.dispatch_task_set(cpu, gpu, dispatching_task_set, SE.dags[dispatching_task_set.dag_id])
                    else:
                        SE.dispatch_queue['gpu'][device_id].append(dispatching_task_set)
                        dispatching_task_component.get_first_kernel().in_frontier=False
                        SE.frontier.queue.remove(dispatching_task_component)
                        # SE.dispatch_task_set(cpu, gpu, dispatching_task_set, SE.dags[dispatching_task_set.dag_id])
                        
                         

                elif dispatch_device == 'cpu':
                    
                    dispatching_task_set.Class="ZERO"
                    if device_id in SE.ready_queue['cpu']:
                        SE.ready_queue['cpu'].remove(device_id)
                        cpu = device_id
                        # SE.frontier.get()
                        dispatching_task_component.get_first_kernel().in_frontier=False
                        SE.frontier.queue.remove(dispatching_task_component)
                        # print "SCHEDULE: Dispatching " + str(list(dispatching_task_set.get_kernel_ids())) + " of DAG " + str(
                            # dispatching_task_set.dag_id) + " to GPU " + str(gpu)  
                        SE.dispatch_task_set(cpu, gpu, dispatching_task_set, SE.dags[dispatching_task_set.dag_id])
                    else:
                        SE.dispatch_queue['cpu'][device_id].append(dispatching_task_set)
                        dispatching_task_component.get_first_kernel().in_frontier=False
                        SE.frontier.queue.remove(dispatching_task_component)
                        # SE.dispatch_task_set(cpu, gpu, dispatching_task_set, SE.dags[dispatching_task_set.dag_id])
            else:
                c,g=SE.update_execution_pool_of_task_sets()
                
        SE.makespan = SE.time_stamp
    return period_values


def generate_periods(wcet):
    possible_periods = [wcet*1,wcet*2,wcet*3,wcet*4]
    p = []
    for i in range(4):
        p.append(random.choice(possible_periods))
    return p


def generate_task_set_periods(period_configuration_file,ex_map):
    def combination_with_repetition(possible_periods_hyper_cycle,hyper_period,did,dag_period_map):
        if did == len(dag_period_map.keys())-1:
            for p in dag_period_map[did]:
                period = deepcopy(hyper_period)
                period[did] = p
                possible_periods_hyper_cycle.append(period)
        else:
            for p in dag_period_map[did]:
                period = deepcopy(hyper_period)
                period[did] = p
                combination_with_repetition(possible_periods_hyper_cycle, period, did+1, dag_period_map)

               
    contents = open(period_configuration_file,'r').readlines()
    dag_period_map = {}
    dags = []
    counter_id = 0
    global_file_list = open("global_map.txt", "r").readlines()
    global_map = make_dict(global_file_list)
    DC = DAGCreator()
    for line in contents:
        dag_name,period_configuration = line.strip("\n").split("=")
        dag_period_map[counter_id] = eval(period_configuration)
        input_file = "ADAS_Graphs/"+dag_name
        dags.append(DC.create_adas_job_from_file_with_fused_times(input_file, counter_id, global_map, ex_map, counter_id,0.0))
        counter_id += 1
    possible_periods_hyper_cycle = []
    hyper_period = [0 for d in dag_period_map.keys()]
    combination_with_repetition(possible_periods_hyper_cycle,hyper_period,0,dag_period_map)
    return dags,possible_periods_hyper_cycle


def generate_old_task_set_periods(period_configuration_file):

    # possible_periods_task1 = [28,28,42]
    # possible_periods_task2 = [12,24,36]
    # possible_periods_task3 = [14,28,42]
    # possible_periods_task4 = [12,24,36]

    # possible_periods_task1 = [30,45,90]
    # possible_periods_task2 = [30,45,90]
    # possible_periods_task3 = [30,45,90]
    # possible_periods_task4 = [30,45,90]

    contents = open("PeriodConfiguration/period3.stats",'r').readlines()
     
    
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
        

if __name__ == '__main__':

    
    trace_folder = "Results/" + args.trace_f
    statistics_file = "Results/" + args.stats
    os.system("mkdir Results/" + args.trace_f)
    os.system("mkdir Results/"+args.trace_f+"/Pos")
    os.system("mkdir Results/"+args.trace_f+"/Neg")
    period_configuration = args.period
    global_map = {}
    pos = 0
    neg = 0
    # global_file_list = open("global_map.txt", "r").readlines()
    # global_map = make_dict(global_file_list)
    # kernel_info_list = [obtain_kernel_info(key) for key in global_map]
    ex_cpu = {}
    ex_gpu = {}
    ex_map = (ex_cpu, ex_gpu)
    # DC = DAGCreator()
    
    # input_file1="ADAS_Graphs/dnn.graph"
    # input_file2="ADAS_Graphs/cnn.graph"
    
    # dag1 = DC.create_adas_job_from_file_with_fused_times(input_file1, 0, global_map, ex_map, 0,0.0)
    # dag2 = DC.create_adas_job_from_file_with_fused_times(input_file2, 1, global_map, ex_map, 1,0.0)
    # dag3 = DC.create_adas_job_from_file_with_fused_times(input_file1, 2, global_map, ex_map, 2,0.0)
    # dag4 = DC.create_adas_job_from_file_with_fused_times(input_file2, 3, global_map, ex_map, 3,0.0)

    # dags=[dag1, dag2, dag3, dag4]
    
    dags,task_periods = generate_task_set_periods(period_configuration,ex_map)
    
    
    num_jobs = len(dags)
    # print dag.deadline, dag.wcet, dag.rates
    # print dag.fused_kernel_timings
    # print ex_map
    # periods=[42*3, 42, 42, 42]
    # SE = ScheduleEngine(1, 1, "local_deadline",ex_map,True,global_map)
    # schedule_workflows(SE,dags,periods)
    # print SE.makespan, SE.calculate_avg_lateness()
    # periods=[25.0, 75.0, 25.0, 25.0]
    
    # f = open("continous_rewards_50_per_run.stats", "w")
    
    mean_rewards =  []
    tardiness_values = []
    deadlines_missed_values = []
    avg_makespan_values = []
    period_values = []
    lateness_values=[]
    percent_lateness_values=[]
    
    num_episodes = len(task_periods)
    max_lateness_values = []
    for i_episode in range(num_episodes):
        print "EPISODE: ",i_episode
        #ex_map -> execution map key -> task_name value -> time
        SE = ScheduleEngine(1, 1, "local_deadline",ex_map,True,global_map,num_jobs=num_jobs)
        periods = task_periods[i_episode]
        period_values.append(periods)
        # print("periods",periods)
        # print("period_values",period_values)
        schedule_workflows(SE,dags,periods,trace_folder,i_episode)
        # schedule_workflows(SE,dags,periods)
        print "MAKESPAN", SE.makespan 
        avg_lateness = SE.calculate_avg_lateness()
        max_lateness = SE.calculate_max_lateness()
        SE.calculate_avg_makespan()
        print "AVG MAKESPAN", SE.avg_makespan
        avg_makespan_values.append(SE.avg_makespan)
        tardiness,deadlines_missed = SE.calculate_avg_tardiness()
        tardiness_values.append(tardiness)
        deadlines_missed_values.append(float(deadlines_missed)/len(SE.dags))
        max_lateness_values.append(max_lateness)
        lateness_values.append(avg_lateness)
        percent_lateness = SE.calculate_percent_lateness()
        percent_lateness_values.append(percent_lateness)
       
        file_name_dispatch = "dispatch_history_" + str(i_episode)+".stats"
        file_name_tasks = "task_set_history_" + str(i_episode)+".stats"
        if max_lateness > 0.0 :
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
            
      
    # torch.save(target_net.state_dict(),"./target_network_continous_50_per_run.pth")
    print "Accuracy: ", float(pos) /(pos+neg)
    f = open(statistics_file,'w')
    for i in range(num_episodes):
        stats_string = str(task_periods[i]) + ":" + str(deadlines_missed_values[i]) + "," + str(max_lateness_values[i])+ "," + str(tardiness_values[i]) + ","+ str(avg_makespan_values[i]) + ","+ str(lateness_values[i])+"\n"
        f.write(stats_string)
    f.close()
    