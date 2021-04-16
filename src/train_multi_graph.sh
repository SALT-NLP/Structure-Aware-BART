CUDA_VISIBLE_DEVICES=0 python train.py \
    --data_dir ./data \
    --learning_rate=3e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --check_val_every_n_epoch 1 \
    --early_stopping_patience 4 \
    --max_source_length 800 \
    --task summarization \
    --label_smoothing 0.1 \
    --model_name_or_path facebook/bart-base \
    --cache_dir ./cache \
    --output_dir ./composit \
    --lr_scheduler polynomial \
    --weight_decay 0.01 --warmup_steps 120 --num_train_epochs 20 \
    --max_grad_norm 0.1 \
    --dropout 0.1 --attention_dropout 0.1 \
    --train_batch_size 4 \
    --eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --sortish_sampler \
    --seed 42 \
    --val_metric loss \
    --discourse_graph \
    --relation \
    --lr_new 10 \
    --warm_up_new 60 \
    --warm_up_new_2 60 \
    --discourse_attn_head 2\
    --action_graph \
    --action_encoder_attn_head 2 --composit \
    --config_name bart_config_10_7.json \
    --num_workers 1 \
    --logger_name wandb \
    "$@"

    

