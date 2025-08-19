model_name=TimeLLM
train_epochs=100
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=1
batch_size=4
d_model=32
d_ff=128

comment='TimeLLM-ETTh1'
export CUDA_VISIBLE_DEVICES=1
# accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model 32 \
#   --d_ff 128 \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --use_accelerator \
#   --llm_model 'LLAMA' \
#   --llm_dim 4096 \
#   --draw_mse

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port 12346 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --use_accelerator \
  --llm_model 'LLAMA' \
  --llm_dim 4096 \
  --draw_mse

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port 12347 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'COS'\
  --learning_rate 0.001 \
  --use_accelerator \
  --llm_model 'LLAMA' \
  --llm_dim 4096 \
  --draw_mse

accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port 12348 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --use_accelerator \
  --llm_model 'LLAMA' \
  --llm_dim 4096 \
  --draw_mse
  
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --use_accelerator \
  --llm_model 'LLAMA' \
  --llm_dim 4096 \
  --draw_mse

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --use_accelerator \
  --llm_model 'LLAMA' \
  --llm_dim 4096 \
  --draw_mse

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'cosine'\
  --learning_rate 0.001 \
  --use_accelerator \
  --llm_model 'LLAMA' \
  --llm_dim 4096 \
  --draw_mse

accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --use_accelerator \
  --llm_model 'LLAMA' \
  --llm_dim 4096 \
  --draw_mse