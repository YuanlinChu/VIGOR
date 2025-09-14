import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--model_type', type=str, default=None)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--image_data_path', type=str, default=None)
parser.add_argument('--json_data_path', type=str, default=None)

args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

s = 0
e = 68000 - 1
# e = 68 - 1
# gpus = [[0], [1], [2], [3], [4], [5], [6], [7]]
# gpus = [[0]]
gpus = [[0], [1], [2], [3]]
num_p = len(gpus)

if args.model_type == 'llava_v15_v':
    outdir = '{}/llava_v15_v_{}_{}'.format(args.outdir,s,e)
elif args.model_type == 'llava_v15_t':
    outdir = '{}/llava_v15_t_{}_{}'.format(args.outdir,s,e)
elif args.model_type == 'qwen2_vl_v':
    outdir = '{}/qwen2_vl_v_{}_{}'.format(args.outdir,s,e)
elif args.model_type == 'qwen2_vl_t':
    outdir = '{}/qwen2_vl_t_{}_{}'.format(args.outdir,s,e)



def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    # gpu_index_str = [str(i) for i in gpu_index]
    # gpu_index_str=','.join(gpu_index_str)
    gpu_index = gpus[i]
    gpu_index_str = ' '.join(map(str, gpu_index))
    # gpu_index_str='['+gpu_index_str+']'

    if args.model_type == 'llava_v15_t' or args.model_type == 'llava_v15_v':
        command = "python ge_data_all_llava15.py --start={} --end={} --index={} --gpu_index {} --outdir {} --model {} --image_data_path {} --json_data_path {}".format(start, end, index,gpu_index_str, outdir, args.model, args.image_data_path, args.json_data_path)
    elif args.model_type == 'qwen2_vl_t' or args.model_type == 'qwen2_vl_v':
        command = "python ge_data_all_qwen2vl.py --start={} --end={} --index={} --gpu_index {} --outdir {} --model {} --image_data_path {} --json_data_path {}".format(start, end, index,gpu_index_str, outdir, args.model, args.image_data_path, args.json_data_path)


    commands.append(command)
# run_command(commands[0])
# commands=commands[:1]
with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)
