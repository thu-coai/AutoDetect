CUDA_VISIBLE_DEVICES=0

tasks=("string_processing" "data_container_operations" "logic_and_control_flow")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_code.py \
    --category ${task} 
done


tasks=("functions_and_modules" "mathematics_and_algorithms")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_code.py \
    --category ${task} 
done


tasks=("data_structures" "file_handling")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_code.py \
    --category ${task} 
done