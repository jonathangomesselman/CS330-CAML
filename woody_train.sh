#!/bin/bash

seed=1
ROT="--n_layers 2 --n_hiddens 100 --data_path data/ --save_path woody_results/ --batch_size 1 --log_every 100 --samples_per_task 100 --data_file mnist_rotations.pt --cuda no --seed"

# echo "Beginning Online Learning" "( seed =" $seed ")"
# echo "MNIST Rotations:"
# python3 main.py $ROT $seed --model online --lr 0.0003

# echo "Beginning Independent Model Per Task" "( seed =" $seed ")"
# echo "MNIST Rotations:"
# python3 main.py $ROT $seed --model independent --lr 0.01

# echo "Beginning Task Input Learning" "( seed =" $seed ")"
# echo "MNIST Rotations:"
# python3 main.py $ROT $seed --model taskinput --lr 0.01

# echo "Beginning MER (Algorithm 1) With 5120 Memories" "( seed =" $seed ")"
# echo "MNIST Rotations:"
# python3 main.py $ROT $seed --model meralg1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 5120 --replay_batch_size 100 --batches_per_example 10

echo "Beginning CAML (Algorithm 1) With 5120 Memories" "( seed =" $seed ")"
echo "MNIST Rotations:"
python3 main.py $ROT $seed --model caml1 --lr 0.03 --beta 0.03 --gamma 1.0 --memories 5120 --replay_batch_size 100 --batches_per_example 10 --caml_priority loss --softmax_temperature 1.0