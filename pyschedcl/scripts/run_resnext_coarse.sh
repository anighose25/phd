 for (( headoncpu = 0 ; headoncpu <=0 ; headoncpu++ ))
 do
    python scheduling/create_resnext.py $1 $headoncpu 32 
    #python scheduling/generate_buffer_times.py -f ./dag_info/dag_resnext/ -rc
    python scheduling/setup_cq_cluster_and_deploy.py -f ./dag_info/dag_resnext/ -ng 1 -nc 1 -rc -ef './logs/resnext_profiling_'$1'.json' -fdp './profiling/dumps_resnext/coarse_resnext_'$1'_'$headoncpu'.json' $2
    echo !!
done
