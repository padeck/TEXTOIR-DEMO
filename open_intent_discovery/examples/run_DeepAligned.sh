#!/usr/bin bash


for dataset in 'labeled_5' #'clinc'
do
    for known_cls_ratio in 0.25 0.5 0.75 #0.75 0.9
    do
        for seed in 0 
        do 
            python run.py \
            --dataset $dataset \
            --method 'DeepAligned' \
            --setting 'semi_supervised' \
            --known_cls_ratio $known_cls_ratio \
            --seed $seed \
            --backbone 'bert' \
            --config_file_name 'DeepAligned' \
            --gpu_id '0' \
            --train \
            --save_results \
            --save_frontend_results \
            --results_file_name 'results_DeepAligned.csv' 
        done
    done
done
