for trial in 1 2 3 4 5 6 7 8 9 10
do
CUDA_VISIBLE_DEVICES=0 python regdb_train.py --iters 100 --momentum 0.1 --trial $trial
done
echo 'Done'