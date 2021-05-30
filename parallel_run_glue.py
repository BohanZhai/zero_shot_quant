#!/usr/bin/python
#!/usr/bin/python3

# This script assume exclusive usage of the GPUs.
# If you have limited usage of GPUs, you can limit the range of gpu indices you are using.


import threading
import time
import os
import numpy as np


import gpustat
import logging

import itertools

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'
FORMAT_MINIMAL = '%(message)s'

logger = logging.getLogger('runner')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


exitFlag = 0
GPU_MEMORY_THRESHOLD = 1000 # MB?
NUM_GPUS = 1
def get_free_gpu_indices():
    '''
        Return an available GPU index.
    '''
    while True:
        stats = gpustat.GPUStatCollection.new_query()
        # print('stats length: ', len(stats))
        return_list = []
        for i, stat in enumerate(stats.gpus):
            memory_used = stat['memory.used']
            if memory_used < GPU_MEMORY_THRESHOLD:
                return_list.append(i)
                if len(return_list) == NUM_GPUS:
                    return return_list

        logger.info("Waiting on GPUs")
        time.sleep(10)


class DispatchThread(threading.Thread):
    def __init__(self, threadID, name, counter, bash_command_list):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.bash_command_list = bash_command_list

    def run(self):
        logger.info("Starting " + self.name)
        # print_time(self.name, self.counter, 5)
        threads = []
        for i, bash_command in enumerate(self.bash_command_list):

            cuda_device = get_free_gpu_indices()
            thread1 = ChildThread(1, f"{i}th + {bash_command}", 1, cuda_device, bash_command)
            thread1.start()
            import time
            time.sleep(30)
            threads.append(thread1)

        # join all.
        for t in threads:
            t.join()
        logger.info("Exiting " + self.name)



class ChildThread(threading.Thread):
    def __init__(self, threadID, name, counter, cuda_device, bash_command):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.cuda_device = cuda_device
        self.bash_command = bash_command

    def run(self):
        if len(self.cuda_device) == 4:
            os.environ['CUDA_VISIBLE_DEVICES'] = f'{int(self.cuda_device[0])},{int(self.cuda_device[1])},{int(self.cuda_device[2])},{int(self.cuda_device[3])}'
        elif len(self.cuda_device) == 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = f'{int(self.cuda_device[0])}'
        elif len(self.cuda_device) == 2:
            os.environ['CUDA_VISIBLE_DEVICES'] = f'{int(self.cuda_device[0])},{int(self.cuda_device[1])}'
        elif len(self.cuda_device) == 8:
            os.environ['CUDA_VISIBLE_DEVICES'] = f'{int(self.cuda_device[0]),{int(self.cuda_device[1])},{int(self.cuda_device[2])},{int(self.cuda_device[3])},{int(self.cuda_device[4])},{int(self.cuda_device[5])},{int(self.cuda_device[6])},{int(self.cuda_device[7])} }'
        
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cuda_device}'
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'0,1'
        bash_command = self.bash_command

        logger.info(f'executing {bash_command} on GPU: {self.cuda_device}')
        # ACTIVATE
        os.system(bash_command)
        import time
        import random
        time.sleep(random.random() % 5)

        logger.info("Finishing " + self.name)


BASH_COMMAND_LIST = []

# for weight_trainable in [" ", "--weight_trainable"]:
model='bert_base'
gradient_accumulation_steps=1

# model='bert_large'
# gradient_accumulation_steps=4


GLUE_DIR="/dnn/sheng.s/glue_data/"
FINETUNED_MODEL_DIR=f"/rscratch/sheng.s/model/{model}/"

############################
# general finetuning
############################
# for TASK_NAME in ["SST-2", "RTE",  "QNLI", "MRPC", "QQP", "STS-B", "CoLA", "MNLI"]:
for TASK_NAME in ["QNLI"]:
# for TASK_NAME in ["STS-B"]:
    command = f''' python run_glue.py \
                --task_name {TASK_NAME} \
                --data_dir {GLUE_DIR}/{TASK_NAME}/ \
                --student_model {FINETUNED_MODEL_DIR} \
                --output_dir {model}/base_{TASK_NAME} \
                --learning_rate 2e-5 \
                --train_batch_size 32 \
                --do_lower_case --seed 1 --gradient_accumulation_steps {gradient_accumulation_steps}'''

    # BASH_COMMAND_LIST.append(command)


############################
# quantization aware finetuning
# add --do_eval means directly apply the post quantization scheme
# tune weight_bit, group_num for weight quantization
############################
# for TASK_NAME in ["SST-2", "RTE", "MRPC", "QNLI", "QQP", "STS-B", "MNLI", "CoLA"]:
for TASK_NAME in ["SST-2"]:
    for weight_bit in [4, 6, 8]:
        for group_num in [1, 12, 64]:
            command = f''' python quant_run_glue.py \
                        --task_name {TASK_NAME} \
                        --data_dir {GLUE_DIR}/{TASK_NAME}/ \
                        --student_model {model}/base_{TASK_NAME} \
                        --output_dir {model}/q{weight_bit}a8_g{group_num}/base_{TASK_NAME} \
                        --learning_rate 2e-5 \
                        --train_batch_size 32 --quant_group_number {group_num} \
                        --do_lower_case --seed 1 --gradient_accumulation_steps {gradient_accumulation_steps} --weight_bit {weight_bit}'''

            # BASH_COMMAND_LIST.append(command)


############################
# quantization aware finetuning on generated dataset
# add --do_eval means directly apply the post quantization scheme
# tune weight_bit, group_num for weight quantization
############################
# for TASK_NAME in ["SST-2", "RTE", "MRPC", "QNLI", "QQP", "STS-B", "MNLI", "CoLA"]:
for TASK_NAME in ["CoLA"]:
    for weight_bit in [6, 8, 4]:
        for group_num in [1]:
            for aug_text in ["_gen_ood"]:
                command = f''' python quant_run_glue.py \
                            --task_name {TASK_NAME} \
                            --data_dir {GLUE_DIR}/{TASK_NAME}/ \
                            --student_model {model}/base_{TASK_NAME} \
                            --output_dir {model}/q{weight_bit}a8_g{group_num}/{aug_text}/base_{TASK_NAME} \
                            --learning_rate 2e-5 \
                            --train_batch_size 32 --quant_group_number {group_num} --aug_train {aug_text} \
                            --do_lower_case --seed 1 --gradient_accumulation_steps {gradient_accumulation_steps} \
                            --weight_bit {weight_bit}'''
                BASH_COMMAND_LIST.append(command)

############################
# post quantization, no finetuning
# add --do_eval means directly apply the post quantization scheme
# tune weight_bit, group_num for weight quantization, activation_bit 
############################
for TASK_NAME in ["CoLA"]:
    for weight_bit in [4]:
        for group_num in [1]:
            for activation_bit in [32]:
                command = f''' python quant_run_glue.py \
                            --task_name {TASK_NAME} \
                            --data_dir {GLUE_DIR}/{TASK_NAME}/ \
                            --student_model {model}/base_{TASK_NAME} \
                            --output_dir {model}/base_{TASK_NAME}/res/ \
                            --learning_rate 2e-5 --activation_bit {activation_bit}\
                            --train_batch_size 32 --quant_group_number {group_num} \
                            --do_lower_case --do_eval --seed 1 --gradient_accumulation_steps {gradient_accumulation_steps} \
                            --weight_bit {weight_bit} | tee -a {model}/base_{TASK_NAME}/res/post_q{weight_bit}a{activation_bit}_g{group_num}_res,txt '''

                # BASH_COMMAND_LIST.append(command)
                                            



print('Number of Jobs: ',len(BASH_COMMAND_LIST))
# Create new threads
dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST)
# dispatch_thread = DispatchThread(2, "Thread-2", 4, BASH_COMMAND_LIST[13:14])

# Start new Threads
dispatch_thread.start()
dispatch_thread.join()

import time
time.sleep(5)

logger.info("Exiting Main Thread")