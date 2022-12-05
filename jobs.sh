# Jobs for running model training pipelines
python bert.py --learning_rate 5e-5 --repr concat_smiles_vaeprot --save bert_vae_prot

python bert.py --learning_rate 5e-5 --repr concat --save bert_full_prot
python hw1_copy.py --dropout 0.35 --dataset full --hidden_dim 128 --num_shot 5 --train_steps 150000 --learning_rate 1e-5 --repr concat --save lstm_vae_prot
