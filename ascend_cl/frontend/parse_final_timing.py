import sys
import os
import json
from collections import OrderedDict

if len(sys.argv)<2:
    iteration_id=0
    time_filename="./profiling/final_0_revised.timing"
else:
    iteration_id=sys.argv[1]

if len(sys.argv)<3:
    time_filename="./profiling/final_"+iteration_id+"_mean.timing"
else:
    time_filename=sys.argv[2]


def create_maps(filename):
    
    deadline_map = {}
    execution_time_map = {}
    with open(filename,'r') as file_content:

        for line in file_content:
            # dag_details = file_content.read()
            # print line
            dag_array = line.split(" ")
            deadline_key = dag_array[0] + "_" + dag_array[1]
            deadline_value = dag_array[2]
            deadline_map[deadline_key] = float(deadline_value)

            num_of_nodes = 0
            if int(dag_array[0]) == 0 or int(dag_array[0]) == 2:
                network = 'yololite'
                num_of_nodes = 12
            elif int(dag_array[0]) == 1 or int(dag_array[0]) == 3:
                network = 'edlenet'
                num_of_nodes = 9

            
            for i in range(num_of_nodes):
                key_extime_map = dag_array[0] + "_" + dag_array[1] + "_" + str(i)

                execution_time_map[key_extime_map+"_ndrange"] = ('dev',0,0)
                execution_time_map[key_extime_map+"_read"] = ('dev',0,0)
                execution_time_map[key_extime_map+"_write"] = ('dev',0,0)
        
    return deadline_map, execution_time_map


def populate_ex_time_map(filename, execution_time_map):
    
    with open(filename,'r') as file_content:
        json_time=json.loads(file_content.read())

    dag_finish_map = {}
    for key in json_time:
        key_array = key.split("_")
        map_key=key_array[1] + "_" + key_array[2] + "_" + key_array[3] + "_" + key_array[0].split("-")[0]
        # print map_key

        map_value=(str(key_array[4]),float(json_time[key]['end']),float(json_time[key]['start']))
        
        # print map_key

        if execution_time_map[map_key] == ('dev',0,0):
            execution_time_map[map_key] = map_value 
            # print map_value
        else:
            print "Key not found", map_key

    return execution_time_map


def find_offset(execution_time_map):

    offset=float('inf')
    for key in execution_time_map:
        if  execution_time_map[key][2]>0 and execution_time_map[key][2]<offset:
            # print execution_time_map[key][2]
            offset=execution_time_map[key][2]

    # print offset

    return offset

def update_time_map(execution_time_map,offset):

    updated_time_map = {}
    for key in execution_time_map:
        end = (execution_time_map[key][1]-offset)*1e-6
        start = (execution_time_map[key][2]-offset)*1e-6
        # print key , offset, " : ", end, start, end-start
        updated_time_map[key] = (execution_time_map[key][0],end,start,end-start)

    return updated_time_map


def check_deadline_misses(deadline_map,execution_time_map):

    misses = 0
    count = 0
    for dag in deadline_map:
        deadline = deadline_map[dag]

        ex_time_map_key = dag + "_" + str(11 if int(dag.strip("_")[0])%2 == 0 else 8) 
        
        if execution_time_map[ex_time_map_key+"_read"][1] == 0:
            finish_time = execution_time_map[ex_time_map_key+"_ndrange"][1]
        else:
            finish_time = execution_time_map[ex_time_map_key+"_read"][1]

        lateness = finish_time-deadline

        if lateness > 0:
            misses = misses+1
            print dag , " : " ,deadline, " - " ,finish_time, " - ", lateness

        count=count+1

    print misses, count

    miss_rate =float(misses)/count

    return miss_rate


if __name__ == "__main__":

    deadline_map, time_map = create_maps("./database/test_config/period_instances/period_instances_"+str(iteration_id)+".stats")   

    # ex_time_map = populate_ex_time_map("./lowlevel_results/final_"+str(iteration_id)+"_revised.timing",time_map)
    ex_time_map = populate_ex_time_map(time_filename,time_map)


    time_filename
    offset = find_offset(ex_time_map)

    execution_time_map = update_time_map(ex_time_map, offset)

    # for key in execution_time_map:
    #     print key , "-> ", execution_time_map[key]

    total_misses = check_deadline_misses(deadline_map,execution_time_map)

    print "Miss rate: " , total_misses
    
