"""

Program for task generation for hyper-parameter grid-search performance

Plan on how to format this process properly:
1) parse arguments from task_generator input;
2) parse rest of the arguments to another args object;
3) Get grid size so can loop N times and create N tasks;
4) Save said tasks to their respective files and end generation.

"""


from __future__ import print_function
import argparse
import subprocess
import platform
import re
import itertools
from pathlib import Path
from argparse import Namespace
from sklearn.model_selection import ParameterGrid

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_string(s):
    try:
        str(s)
        return True
    except ValueError:
        return False

# Extra parsing function for other arguments:
def combine_args(args, args_other):
    # Potential working fix for this:
    # TODO: Check args if parameter_grid contains parameters gotten from args_other:
    args = args
    arguments = vars(args)
    values = []
    # Iterate over args_other, add them to args:
    for i in range(len(args_other)):
        if '--' in args_other[i]:
            arg = args_other[i]
        else:
            # Check if integer, float or string:
            if is_int(args_other[i]):
                value = int(args_other[i])
            elif is_float(args_other[i]):
                value = float(args_other[i])
            else:
                value = str(args_other[i])

            values.append(value)

        # If last element in args_other, append and end this process:
        if i + 1 == len(args_other):
            if len(values) > 1:
                arguments[arg.strip('--')] = list(values)
            else:
                arguments[arg.strip('--')] = value
            values.clear()
            break

        # If next element contains '--', new argument:
        if '--' in args_other[i + 1]:
            if len(values) > 1:
                arguments[arg.strip('--')] = list(values)
            else:
                arguments[arg.strip('--')] = value
            values.clear()
            continue

    return args


# Function for preparing arguments to use in other programs:
def prepare_args_for_programs(args):

    list_of_args = []
    parameter_grid = {}
    # Need to:
    # Check out parameter grid and duplicate parameter line N times (N - number of grid combinations):
    # Insert grid combinations in duplicated lists (Since there's no positional requirements it should be fairly simple)
    # Then need to insert them into generator script:
    # Check for arguments needed to form parameter grid:
    for param in args.parameter_grid:
        parameter_grid[param] = []

        # Check args if contain said argument:
        for arg in vars(args):
            values = getattr(args, arg)

            # Check if argument is the same as parameter from grid:
            if arg == param:
                # Now check values, if empty or 1 value then fuck off:
                if values is None:
                    raise Exception(f'PepeHands No values given for argument: {arg}')
                elif len(values) < 2:
                    raise Exception(f'WeirdChamp Only one value for argument: {arg}')
                else:
                    # Got values, making grid Pog (Breaking loop, because no other values for this argument):
                    parameter_grid[param] += values
                    break

    # Grid parameters are ready, time to create grid:
    parameter_grid = ParameterGrid(parameter_grid)

    # Now forming strings to generate tasks:
    for idx, sample in enumerate(parameter_grid):

        arg_string = []
        # Loop through all arguments in args:
        for arg in vars(args):

            # Semantic check for added argument:
            add_arg = False
            # Get value of argument:
            value = getattr(args, arg)

            # Take one key from parameter grid sample:
            for key in sample:
                # Compare key from sample to arg, if match, add value from sample instead of key:
                if arg == key:
                    arg_string.append('--' + key)
                    arg_string.append(str(sample[key]))
                    # Exit argument loop to get
                    add_arg = True

            # If already added argument from sample, skip over this addition:
            if add_arg:
                continue
            # Append to list a key and its elements:
            arg_string.append('--' + arg)
            arg_string.append(str(value))

        # Lastly add index as a separate parameter, will get absorbed by args_other and turned in for csv writer:
        arg_string.extend(['--index', str(idx)])

        # Turn this list into a string object and append to list of all parameter strings:
        arg_string = " ".join(arg_string)
        list_of_args.append(arg_string)

    return list_of_args


# Argument parser for setting up task generator:
parser = argparse.ArgumentParser(add_help=True, description="Task generator")

parser.add_argument('-m', '--main', type=str, required=True, help="name of main executable file")
parser.add_argument('-ptm', '--path_to_main', type=str, required=True, help="path to main executable file")
parser.add_argument('-r', '--report', type=str, required=True,
                    help="path to report file which will contain all information")
parser.add_argument('-psl', '--path_status_logs', type=str, required=True, help="path to where status logs are saved")
parser.add_argument('-pg', '--parameter_grid', nargs='*', required=True, help="parameters to be grid searched")
parser.add_argument('-sn', '--script_name', type=str, required=True, help="name of exeutable script (Full path pls)")
parser.add_argument('-q', '--queue', default='batch', type=str, help="queue type for working environment")
parser.add_argument('-gpu', '--gpu_type', type=str, default='v100', required=False,
                    help="GPU type: v100 or k40 and 1 by default")
parser.add_argument('-cpu', '--cpu_count', type=int, default=12, required=False, help="CPU count for all tasks")
parser.add_argument('-mpc', '--memory_per_cpu', type=int, required=False, default=5, help="RAM per CPU")
parser.add_argument('-pn', '--process_name', type=str, required=False, default='Training',
                    help="Process name for script naming.")

args, args_other = parser.parse_known_args()

# Parse arguments from other args:
args = combine_args(args=args, args_other=args_other)
# Prepare arguments for programs:
program_arguments = prepare_args_for_programs(args)

# Set other parameters needed:
if args.memory_per_cpu == 0:
    memory = 5 * args.cpu_count   # If memory is 0, then make it 5gb which is close to max value
else:
    memory = args.memory_per_cpu * args.cpu_count


if args.queue == 'fast':
    walltime = 12
elif args.queue == 'long':
    walltime = 336
else:
    walltime = 96


# Start writing to .sh file:
filepath = args.script_name
with open(filepath, 'w') as f:

    f.write('#!/bin/sh -v\n')
    f.write(f'#PBS -N {args.process_name}\n')
    f.write(f'#PBS -o {args.path_status_logs}\n')
    f.write(f'#PBS -e {args.path_status_logs}\n')
    f.write(f'#PBS -q {args.queue}\n')
    f.write(f'#PBS -l nodes=1:ppn={args.cpu_count}:gpus=1:shared,feature={args.gpu_type}\n')
    f.write(f'#PBS -l mem={memory}gb\n')
    f.write(f'#PBS -l walltime={walltime}:00:00\n\n')
    f.write('module load conda\n')
    f.write('eval "$(conda shell.bash hook)"\n')
    f.write('source activate machine_learning\n')
    f.write(f'cd {args.path_to_main}\n')
    
    # Now write all strings from list of args to the file for execution:
    for arguments in program_arguments:

        # Check if not last argument line:
        if arguments != program_arguments[-1]:
            f.write(f'python {args.path_to_main}/{args.main} {arguments} &\n')
        else:
            f.write(f'python {args.path_to_main}/{args.main} {arguments} \n')
            f.write('wait')   


# Close file, say "done" and exit program:
f.close()
print("done")
exit(0)