from mpi4py import MPI
import os
import time

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # configs = open("Configurations/dump_config.stats",'r').readlines()
    # stride = len(configs)/48

    # BATCH 1
    # configs = []
    # periods = [1,2,4]
    # confs = [1,3,5,6]
    # modes = {'dqn':1,'ddqn':3}
    # activation = {'relu':1,'sigmoid':2}

    # for p in periods:
    #     for i in confs:
    #         for j in modes:
    #             for k in activation:
    #                 configs.append([str(p),str(i),j,k])
    # stride = len(configs)/48

    # for i in range(stride*rank, (rank+1)*stride):
    #     period_file = "Temp_Period_Configs1/period"+configs[i][0]+".stats"
    #     config_file = "ConfigurationsCorrect/configuration_"+configs[i][1]+".stats"
    #     model_file = "experiments_5_9_19_1/model_"+configs[i][1]+"_"+configs[i][2]+"_"+configs[i][0]+"_"+configs[i][3]+".pth"
    #     rewards_file =  "experiments_5_9_19_1/rewards_"+configs[i][1]+"_"+configs[i][2]+"_"+configs[i][0]+"_"+configs[i][3]+".stats"
    #     statistics_file = "experiments_5_9_19_1/stats_"+configs[i][1]+"_"+configs[i][2]+"_"+configs[i][0]+"_"+configs[i][3]+".stats"

    #     command = "python train_dqn_agent_all.py -p " + period_file + " -c "+ config_file+ " -m "+model_file + " -r " +rewards_file +" -s "+statistics_file +" -nr 100 -ne 10 -mo " + str(modes[configs[i][2]]) + " -i 1 -a " + str(activation[configs[i][3]])
    #     print command
    #     os.system(command)
        

    # # BATCH 2
    # configs = []
    # # periods = [1,2,3]
    # confs = [1,2,3,4]
    # lr = [0.005,0.001,0.0005]
    # modes = {'dqn':1,'ddqn':3}
    # # activation = {'relu':1,'sigmoid':2}

    # for i in confs:
    #     for j in modes:
    #         for k in lr:
    #             configs.append([str(i),j,str(k)])
    # stride = len(configs)/12

    # for i in range(stride*rank, (rank+1)*stride):
    #     time.sleep(i+1)
    #     period_file = "PeriodConfiguration/period3.stats"
    #     config_file = "Configurations_New/configuration_"+configs[i][0]+".stats"
    #     model_file = "experiments_22_10_19/model_"+configs[i][0]+"_"+configs[i][1]+"_"+"3"+"_"+"relu"+"_"+"lr"+"_"+configs[i][2]+".pth"
    #     rewards_file =  "experiments_22_10_19/rewards_"+configs[i][0]+"_"+configs[i][1]+"_"+"3"+"_"+"relu"+"_"+"lr"+"_"+configs[i][2]+".stats"
    #     statistics_file = "experiments_22_10_19/stats_"+configs[i][0]+"_"+configs[i][1]+"_"+"3"+"_"+"relu"+"_"+"lr"+"_"+configs[i][2]+".stats"

    #     command = "python train_dqn_agent_all.py -p " + period_file + " -c "+ config_file+ " -m "+model_file + " -r " +rewards_file +" -s "+statistics_file +" -nr 100 -ne 10 -mo " + str(modes[configs[i][1]]) + " -i 1 -a 1 -dnn 2 -l 2 -lr " + configs[i][2]
    #     print command
    #     # os.system(command)

    # BATCH Test
    configs = [ "python train_dqn_agent_all.py -p PeriodConfiguration/period3.stats -c Configurations_New/configuration_1.stats -m t1.pth -r t1r.stats -s t1s.stats -nr 100 -ne 10 -mo 3 -i 1 -a 1 -dnn 2 -l 1 -lr 0.001",
                "python train_dqn_agent_temp1.py -p PeriodConfiguration/period3.stats -c Configurations_New/configuration_1.stats -m t2.pth -r t2r.stats -s t2s.stats -nr 100 -ne 10 -mo 3 -i 1 -a 1 -dnn 2 -l 1 -lr 0.001",
                "python train_dqn_agent_temp2.py -p PeriodConfiguration/period3.stats -c Configurations_New/configuration_1.stats -m t3.pth -r t3r.stats -s t3s.stats -nr 100 -ne 10 -mo 3 -i 1 -a 1 -dnn 2 -l 1 -lr 0.001",
                "python train_dqn_agent_temp3.py -p PeriodConfiguration/period3.stats -c Configurations_New/configuration_1.stats -m t4.pth -r t4r.stats -s t4s.stats -nr 100 -ne 10 -mo 3 -i 1 -a 1 -dnn 2 -l 1 -lr 0.001"]

    if rank < 4:
        time.sleep(2*rank)
        command = configs[rank]
        print command
        os.system(command)
