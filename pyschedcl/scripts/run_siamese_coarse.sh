for (( headoncpu = 0 ; headoncpu <= 2; headoncpu++ ))
do
    python scheduling/create_siamese_network.py $1 $headoncpu 2 
    #python scheduling/generate_buffer_times.py -f ./dag_info/dag_siamese/ -rc
    python scheduling/setup_cq_cluster_and_deploy.py -f ./dag_info/dag_siamese/ -ng 1 -nc 1 -rc  -ef './logs/siamese_profiling_'$1'_with_delays.json' -fdp './profiling/dumps_siamese/coarse_siamese_'$1'_'$headoncpu'.json' $2
#    echo 'scheduling/setup_cq_cluster_and_deploy.py -f ./dag_info/dag_siamese/ -ng 5 -nc 5 -rc -ef ./logs/siamese_profiling_'$1'_with_delays.json -fdp ./profiling/dumps_siamese/q_assignment_'$1'_'$headoncpu'.json' 

done
