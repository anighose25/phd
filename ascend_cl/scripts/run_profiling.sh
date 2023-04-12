bash ./scripts/copy_configurations.sh 64 
bash ./scripts/edlenet_normal.sh
bash ./scripts/yololite_normal.sh
bash ./scripts/copy_results.sh edlenet 64 normal
bash ./scripts/copy_results.sh yololite 64  normal
bash ./scripts/edlenet_interference.sh
bash ./scripts/yololite_interference.sh
bash ./scripts/copy_results.sh edlenet 64 interference
bash ./scripts/copy_results.sh yololite 64 interference
