CUDA_VISIBLE_DEVICES=0

tasks=("word_constraint" "specific_sentence" "specific_genre")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_if.py \
    --category content \
    --task ${task} 
done


tasks=("length_constraint" "text_format" "character_format" "punctuation_format" "numeric_format")
for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_if.py \
    --category format \
    --task ${task} 
done


tasks=("multi_lingual" "scenario_simulation" "language_style" "emotional_tone")

for task in ${tasks[@]};
do
    echo ${task}
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python autodetect_if.py \
    --category "general constraints" \
    --task ${task} 
done