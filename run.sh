export CUDA_VISIBLE_DEVICES="1"

#if [ -d "log_root" ] && [ "$1" == "-r" ]; then
# echo "removing logdir"
# rm -r log_root
#fi

python train.py /nobackup/titans/s155992/Warwick_Dataset
