python ./src/main.py nerf --device cuda -e exp1 --dataset "lego400 config=lego400_cuda load_device=cuda" --model "nerf_c config=nerf" "nerf_f config=nerf" --trainchunk "config=nerfchunk_c" "config=nerfchunk_f" --t-near 2.0 --t-far 6.0 --iter 200000 --batch-size 1 --n-samples 64 --n-samples-fine 128 -cci 500 --ray-batch-size 1024

python ./src/main.py nerf -d cuda -e funny_exp1 -ds "lego400 config=lego400_skewed_cuda deviation=0.2" -m "nerf_c config=nerf" "nerf_f config=nerf" --trainchunk "config=nerfchunk_c" "config=nerfchunk_f" --t-near 2.0 --t-far 6.0 --ray-batch-size 1024 -cci 500 --batch-size 64 --n-samples 64 --n-samples-fine 128 --iter 50000 -s --train --test
python ./src/main.py nerf -d cuda -e funny_exp2 -ds "lego400 config=lego400_skewed_cuda deviation=0.4" -m "nerf_c config=nerf" "nerf_f config=nerf" --trainchunk "config=nerfchunk_c" "config=nerfchunk_f" --t-near 2.0 --t-far 6.0 --ray-batch-size 1024 -cci 500 --batch-size 64 --n-samples 64 --n-samples-fine 128 --iter 50000 -s --train --test
python ./src/main.py nerf -d cuda -e funny_exp3 -ds "lego400 config=lego400_skewed_cuda deviation=0.6" -m "nerf_c config=nerf" "nerf_f config=nerf" --trainchunk "config=nerfchunk_c" "config=nerfchunk_f" --t-near 2.0 --t-far 6.0 --ray-batch-size 1024 -cci 500 --batch-size 64 --n-samples 64 --n-samples-fine 128 --iter 50000 -s --train --test
python ./src/main.py nerf -d cuda -e funny_exp4 -ds "lego400 config=lego400_skewed_cuda deviation=0.8" -m "nerf_c config=nerf" "nerf_f config=nerf" --trainchunk "config=nerfchunk_c" "config=nerfchunk_f" --t-near 2.0 --t-far 6.0 --ray-batch-size 1024 -cci 500 --batch-size 64 --n-samples 64 --n-samples-fine 128 --iter 50000 -s --train --test


HR
