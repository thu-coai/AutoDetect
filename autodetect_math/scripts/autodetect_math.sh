CUDA_VISIBLE_DEVICES=0

tasks=("arithmetic" "polynomials" "equations_and_inequalities")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_math.py \
    --category algebra \
    --task ${task} 
done


tasks=("functions" "sequences_and_series" "complex_numbers")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_math.py \
    --category algebra \
    --task ${task} 
done


tasks=("trigonometry" "coords_and_shapes" "concepts_of_space_and_form")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_math.py \
    --category geometry \
    --task ${task} 
done


tasks=("analysis" "number_theory" "probability_theory")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_math.py \
    --task ${task} 
done