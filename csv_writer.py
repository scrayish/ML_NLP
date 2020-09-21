"""

File for writing results into .csv file. Includes all file locking/unlocking locally in one file for now:


"""


from __future__ import print_function
import os
import csv
import time
import platform
import sys
from pathlib import Path

PLATFORM_WINDOWS = 'Windows'

if platform.system() == PLATFORM_WINDOWS:
    import win32file, win32con, pywintypes
else:
    import fcntl


# Lock file:
def lock_file(f):
    while True:
        try:
            if platform.system() == PLATFORM_WINDOWS:
                hfile = win32file._get_osfhandle(f.fileno())
                win32file.LockFileEx(hfile, win32con.LOCKFILE_FAIL_IMMEDIATELY | win32con.LOCKFILE_EXCLUSIVE_LOCK,
                                     0, 0xffff0000, pywintypes.OVERLAPPED())
            else:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except:
            time.sleep(0.1)


# Unlock file:
def unlock_file(f):
    while True:
        try:
            if platform.system() == PLATFORM_WINDOWS:
                hfile = win32file._get_osfhandle(f.fileno())
                win32file.UnlockFileEx(hfile, 0, 0xffff0000, pywintypes.OVERLAPPED())
            else:
                fcntl.flock(f, fcntl.LOCK_UN)
            break
        except:
            time.sleep(0.1)


# Create file:
def create_file(path: Path):
    path.touch(exist_ok=True)


# Writing function:
def write_to_csv(args, epoch, param_dict):
    
    # Define all necessary variables to write information with:
    filepath = Path(args.report)
    batch = args.batch_size
    lr = args.learning_rate
    model = args.model
    idx = args.index

    # Define fieldnames (Hard-coded for now, maybe changing later):
    fieldnames = ['id', 'model', 'batch', 'lr', 'epoch', 'train loss', 'test loss', 'train acc', 'test acc']

    # Check if file exists, if not, then create file:
    if not filepath.is_file():
        filepath.touch(exist_ok=True)
    
    # Check if file is empty, if yes, open it for writing by default, if not, read content beforehand:
    content = []
    if os.stat(filepath).st_size > 0:
        with filepath.open(mode='r') as csvfile:
            lock_file(csvfile)
            reader = csv.DictReader(csvfile)
            content = list(reader)
            #csvfile.flush()
            #os.fsync(csvfile)
            unlock_file(csvfile)
            csvfile.close()
    
    # Concatenate all parameters into list:
    param_list = [idx, model, batch, lr, epoch]
    for param in param_dict:
        param_list.append(param_dict[param].value()[0])

    # Form input information as dictionary for csv.DictWriter()
    info = dict(zip(fieldnames, param_list))
    
    # Check content, if empty, append regularly, if not, then update specific space according to index value:
    if len(content) == 0:
        content.append(info)
    else:
        new_line = False
        for row in content:
            if row['id'] == str(info['id']):
                for key in row:
                    # If info contains key for row, then can change values, otherwise fuck off (if done correctly all keys should match):
                    if key in info:
                        row[key] = info[key]

                break
            # Check if last line of content, if yes, append new line:
            if row is content[-1]:
                new_line = True

        if new_line:
            # Didn't find index, so has to be new, append as new:
            content.append(info)

    with filepath.open(mode='w') as csvfile:
        lock_file(csvfile)
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        writer.writerows(content)
        csvfile.flush()
        os.fsync(csvfile)
        unlock_file(csvfile)
        csvfile.close()
