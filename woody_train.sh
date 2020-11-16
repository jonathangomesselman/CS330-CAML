#!/bin/bash

seed=0
ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path final_report_results/ --batch_size 1 --log_every 100 --samples_per_task 10 --data_file mnist_rotations.pt --cuda no --seed"

# echo "Beginning Online Learning" "( seed =" $seed ")"
# echo "MNIST Rotations:"
# python3 main.py $ROT $seed --model online --lr 0.1

# echo "Beginning Independent Model Per Task" "( seed =" $seed ")"
# echo "MNIST Rotations:"
# python3 main.py $ROT $seed --model independent --lr 0.1

# echo "Beginning Task Input Learning" "( seed =" $seed ")"
# echo "MNIST Rotations:"
# python3 main.py $ROT $seed --model taskinput --lr 0.1

# echo "Beginning MER (Algorithm 1) With 5120 Memories" "( seed =" $seed ")"
# echo "MNIST Rotations:"
# python3 main.py $ROT $seed --model meralg1 --lr 0.1 --beta 0.03 --gamma 1.0 --memories 128 --replay_batch_size 5 --batches_per_example 5

echo "Beginning CAML (Make sure you set the caml_priority variable) With 5120 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model caml1 --lr 0.1 --beta 0.03 --gamma 1.0 --memories 128 --replay_batch_size 5 --batches_per_example 5 --caml_priority dynamic --softmax_temperature 0.88 --s 1