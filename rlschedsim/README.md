# RLSchedSim

# Training Parameters Used for experiments
```
Double Q Learning
layer structure: [('L', 18), ('L', 30), ('L', 30), ('L', 24)] # Here L stands for Linear or fully connected. The numbers stand for number of hidden units in each layer
activations used for each layer: relu
num_states=10
num_actions=24
replay_size=30000
BATCH_SIZE=16
GAMMA=1
EPS_START=0.9
EPS_END=0.05
EPS_DECAY=100
Learning Rate: 0.001
```
