from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import math
import random
import numpy as np
import sumTree as st
import time

torch.manual_seed(1000)
np.random.seed(1000)
random.seed(1000)

# use_cuda = torch.cuda.is_available()
use_cuda = False
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

def make_layers(cfg, num_states, a_type, batch_norm):
    layers = []
    num_in = num_states
    for layer_type,num_out in cfg:
        if layer_type == 'L':
            layers += [nn.Linear(num_in, num_out)]
            if batch_norm:
	            layers += [nn.BatchNorm1d(num_out)]
            if a_type == '1':
                layers += [nn.ReLU()]
            else:
                layers += [nn.Sigmoid()]
            num_in = num_out
    layers = layers[:-2]
    return nn.Sequential(*layers)

TransitionTuple = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class Transition(object):
    
    def __init__(self, state,action,next_state,reward):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def __repr__(self):
        tu = "(" + str(self.state) + "," + str(self.action.cpu().numpy()[0][0]) + "," + str(self.next_state) + "," + str(self.reward) +")"
        return tu

    def get_tuple(self):
        
        return TransitionTuple(FloatTensor(self.state),self.action,FloatTensor(self.next_state),Tensor([self.reward]))

DAGTransitionMap = {}

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, TransitionObject):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = TransitionObject
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        import random
        a = random.sample(self.memory, batch_size)
        transitions = [t.get_tuple() for t in a]
        return transitions

    def front(self):
        return self.memory[self.position-1]

    def len_data(self):
        return len(self.memory)

class PrioritizedReplayMemory(object):  
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_inc_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = st.SumTree(capacity)
        self.capacity   = capacity
        # self.tree.init_data(Transition)
    
    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def push(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_inc_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data.get_tuple())
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight
    
    def update_priorities(self, indices, error):
        p = self._get_priority(error)
        self.tree.update(indices, p)

    def front(self):
        return self.tree.data[self.tree.cur_position()]

    def len_data(self):
        return len(self.tree.data)

class NN(nn.Module):
    def __init__(self, cfg, num_states,num_actions,a_type,batch_norm):
        super(NN, self).__init__()                    # Inherited from the parent class nn.Module
        self.num_states = num_states
        self.num_actions = num_actions
        self.classifier = make_layers(cfg,num_states,a_type,batch_norm)  
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):                              # Forward pass: stacking each layer together
        # print "FORWARD PASS"
        x = x.view(-1, self.num_states)
        out = self.classifier(x)
        out = self.softmax(out)
        return out

class DuelingNN(nn.Module):
    def __init__(self, cfg, num_states,num_actions,a_type,batch_norm):
        super(DuelingNN, self).__init__()                    # Inherited from the parent class nn.Module
        self.num_states = num_states
        self.num_actions = num_actions

        self.classifier_common = make_layers(cfg,num_states,a_type,batch_norm)

        self.adv1 = nn.Linear(cfg[-1][-1], 64)
        self.adv2 = nn.Linear(64, self.num_actions)

        self.val1 = nn.Linear(cfg[-1][-1], 64)
        self.val2 = nn.Linear(64, 1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):                              # Forward pass: stacking each layer together
        # print "FORWARD PASS"
        x = x.view(-1, self.num_states)
        out = self.classifier_common(x)

        adv = F.relu(self.adv1(out))
        adv = self.adv2(adv)

        val = F.relu(self.val1(out))
        val = self.val2(val)

        out = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        # out = self.softmax(out)
        return out

# Xavier Initialisation

def init_weights_x(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    
# Kaiming Initialisation

def init_weights_k(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0.01)
    

# DQN Agent

class DQNAgent(object):

    def __init__(self,cfg,num_states,num_actions,replay_size,per,duel,ddqn,initn,a_type,l_type,rate,bn,environment_params):

        self.duel = duel
        self.per = per
        self.ddqn = ddqn

        self.l_type = l_type

        if self.duel:
            self.policy_net = DuelingNN(cfg,num_states,num_actions,a_type,bn)
            self.target_net = DuelingNN(cfg,num_states,num_actions,a_type,bn)
        else:
            self.policy_net = NN(cfg,num_states,num_actions,a_type,bn)
            self.target_net = NN(cfg,num_states,num_actions,a_type,bn)

        if initn == '1':
            self.policy_net.apply(init_weights_x)
        else:
            self.policy_net.apply(init_weights_k)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr=float(rate))

        if self.per:
            self.memory = PrioritizedReplayMemory(replay_size)
        else:
            self.memory = ReplayMemory(replay_size)

        self.batch_size, self.gamma,self.eps_start,self.eps_end,self.eps_decay = environment_params
        self.rewards_history = []
        self.steps_done = 0
        self.num_states = num_states
        self.num_actions = num_actions
        self.replay_size = replay_size
        
    def optimize_model(self):
    
        #Double Q Learning
        if self.ddqn:
            if self.memory.len_data() < self.batch_size:
            	return -1
            
            if self.per:
                transitions, mem_indices, mem_weights = self.memory.sample(self.batch_size)
            else:
                transitions = self.memory.sample(self.batch_size)
            
            # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
            # detailed explanation).
            batch = Transition(*zip(*transitions))
            
            # Compute a mask of non-final states and concatenate the batch elements
            
            non_final_mask = ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))
            non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)
            
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.cat(batch.action))
            reward_batch = Variable(torch.cat(batch.reward))

            if self.per:
                weights_batch = Variable(torch.FloatTensor(mem_weights))

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Double DQN - Compute V(s_{t+1}) for all next states.
            V_next_state = Variable(torch.zeros(self.batch_size).type(Tensor))

            _, next_state_actions = self.policy_net(non_final_next_states).max(1, keepdim=True)
            V_next_state[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_state_actions)

            # Remove Volatile as it sets all variables computed from them volatile.
            # The Variable will just have requires_grad=False.
            V_next_state.volatile = False

            # Compute the target Q values
            target_Q_state_action = reward_batch + (self.gamma * V_next_state)

            # Change shape of target_Q_state_action
            temp = []
            for tt in target_Q_state_action.data.numpy():
                temp.append([tt])
            temp = np.array(temp)
            modified_target_Q_state_action = torch.from_numpy(temp)
            target_Q_state_action = Variable(modified_target_Q_state_action)

            # Calc Error and Update Priorities
            if self.per:
                errors = []
                for a,b in zip(state_action_values.data.numpy(),target_Q_state_action.data.numpy()):
                    errors.append(abs(a-b))
                for i in range(self.batch_size):
                    self.memory.update_priorities(mem_indices[i], errors[i])

            # print "state_action_values =",state_action_values
            # print "target_Q_state_action =",target_Q_state_action
            # Compute Huber/Cross-Entropy loss
            if self.l_type == '1':
                loss = F.smooth_l1_loss(state_action_values, target_Q_state_action)
            else:
                loss = F.cross_entropy(state_action_values, target_Q_state_action)
            if self.per:
                loss = weights_batch * loss
                loss = loss.mean()

            # Optimize the model
            
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # print "Loss value in rlnetworks_all.py"
            # print loss.data[0]
            return loss.data[0]

        # No Double Q Learning
        else:
            if self.memory.len_data() < self.batch_size:
            	return -1
            
            if self.per:
                transitions, mem_indices, mem_weights = self.memory.sample(self.batch_size)
            else:
                transitions = self.memory.sample(self.batch_size)
            
            # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
            # detailed explanation).
            batch = Transition(*zip(*transitions))
            
            
            # Compute a mask of non-final states and concatenate the batch elements
            
            non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)))
            non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                        if s is not None]),
                                            volatile=True)
            
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.cat(batch.action))
            reward_batch = Variable(torch.cat(batch.reward))
            if self.per:
                weights_batch = Variable(torch.FloatTensor(mem_weights))

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            
            # print action_batch
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            
            next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))
            
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            
            # Compute the expected Q values
            
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            
            # Undo volatility (which was used to prevent unnecessary gradients)
            
            expected_state_action_values = Variable(expected_state_action_values.data)

            # Change shape of expected_state_action_values
            temp = []
            for tt in expected_state_action_values.data.numpy():
                temp.append([tt])
            temp = np.array(temp)
            modified_val = torch.from_numpy(temp)
            expected_state_action_values = Variable(modified_val)

            if self.per:
                errors = []
                for a,b in zip(state_action_values.data.numpy(),expected_state_action_values.data.numpy()):
                    errors.append(abs(a-b))
                for i in range(self.batch_size):
                    self.memory.update_priorities(mem_indices[i], errors[i])

            # Compute Huber/Cross-Entropy loss
            if self.l_type == '1':
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            else:
                loss = F.cross_entropy(state_action_values, expected_state_action_values)
            if self.per:
                loss = weights_batch * loss
                loss = loss.mean()

            # Optimize the model
            
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # print "Loss value in rlnetworks_all.py"
            # print loss.data[0]
            return loss.data[0]
    
    def select_action(self,state):        
        sample = random.random()
        eps_threshold = self.eps_end+ (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        
        self.steps_done += 1
        time_taken = -1
        if sample > eps_threshold:
            t0 = time.time()
            self.policy_net.eval()
            action =  self.policy_net(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
            t1 = time.time()
            time_taken = t1-t0
        else:
            action = LongTensor([[random.randrange(self.num_actions)]])
        
        # return action
        ##TEST##
        return action, time_taken
        ##END TEST##

        
    def get_action(self,state):
        t0 = time.time()
        self.policy_net.eval()
        action_val = self.policy_net(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
        t1 = time.time()
        time_taken = t1-t0
        return action, time_taken

