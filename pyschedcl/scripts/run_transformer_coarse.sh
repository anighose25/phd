set -o history -o histexpand
for (( size = 6 ; size <= 6 ;size++ ))

do
        powersize=$((2**$size))
        for (( headoncpu = 0 ; headoncpu <= 0; headoncpu++ ))
        do
                python scheduling/create_transformer.py $size $headoncpu 16 
                python scheduling/generate_buffer_times.py -f ./dag_info/dag_transformer/ -rc
                echo $size
                python scheduling/setup_cq_cluster_and_deploy.py -f ./dag_info/dag_transformer/ -ng 1 -nc 1 -rc -ef './logs/transformer_profiling_'$powersize'_'$powersize'_'$powersize'_with_delays.json' -fdp './profiling/dumps_transformer/coarse_transformer_'$powersize'_'$headoncpu'.json' $1
                
                
        done

      
done  
