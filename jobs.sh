# Jobs for running model training pipelines
python bert.py --learning_rate 5e-5 --repr concat_after_full --save bert_full_concat_after
python bert.py --learning_rate 5e-5 --repr concat --save bert_full_concat_before

python lstm.py --dropout 0.35 --dataset full --hidden_dim 128 --num_shot 5 --train_steps 100000 --learning_rate 1e-5 --repr concat_after_full --eval_freq 500 --save lstm_full_concat_after
python lstm.py --dropout 0.35 --dataset full --hidden_dim 128 --num_shot 5 --train_steps 100000 --learning_rate 1e-5 --repr concat --eval_freq 500 --save lstm_full_concat_before
