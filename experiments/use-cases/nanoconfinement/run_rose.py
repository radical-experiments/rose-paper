import os, sys

import radical.pilot as rp

from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine

engine = ResourceEngine({'resource': 'anl.polaris', 
                         'runtime' : 600, 
                         'access_schema':'interactive',
                         'project' : "RECUP",
                         'cores'   : 128, 
                         'gpus'    : 16})

learner = ActiveLearner(engine=engine)
code_path = f'{sys.executable} {os.getcwd()}'

DATA_DIR="/eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/all_data_4050"
#NEW_SAMPLE_SIZE_LIST=[2000]
#NEW_SAMPLE_SIZE_LIST=[400,1600]
NEW_SAMPLE_SIZE_LIST=[150,50,50,50,50,50,50,50,50,500,500,500,500,500,500]
MAX_ITER=len(NEW_SAMPLE_SIZE_LIST)
EPOCHS=20000
NUM_TRAIN=4
NUM_PIPELINE=4
SEED=42
EXP_DIR="/eagle/RECUP/twang/rose/nanoconfinement/nanoconfinement-md/python/surrogate_samplesize/exp"

@learner.training_task
def training(*args, iter_id, instance_id, data_dir, pipeline_dir, epochs, seed):
    print(f"Submit training with --iter {iter_id} --instance {instance_id} --data_dir {data_dir} --pipeline_dir {pipeline_dir} --epochs {epochs} --seed {seed}")
    pre_exec_list = ["source /home/twang3/useful_script/conda_rose.sh"]
    return Task(executable=f'{code_path}/train.py --iter {iter_id} --instance {instance_id} --data_dir {data_dir} --pipeline_dir {pipeline_dir} --epochs {epochs} --seed {seed}',
                cores_per_rank = 8,
                gpus_per_rank = 1,
                threading_type = rp.OpenMP,
                gpu_type = rp.CUDA,
                pre_exec = pre_exec_list)


@learner.active_learn_task
def active_learning(*args, iter_id, data_dir, pipeline_dir, new_sample_size, seed):
    print(f"Submit active learning with --iter {iter_id} --data_dir {data_dir} --pipeline_dir {pipeline_dir} --new_sample_size {new_sample_size} --seed {seed}")
    pre_exec_list = ["source /home/twang3/useful_script/conda_rose.sh"]
    return Task(executable=f'{code_path}/active_learning.py --iter {iter_id} --data_dir {data_dir} --pipeline_dir {pipeline_dir} --new_sample_size {new_sample_size} --seed {seed}',
                cores_per_rank = 2,
                threading_type = rp.OpenMP,
                pre_exec = pre_exec_list)

def teach_single_pipeline(pipeline_dir, seed):
    iter_id = 0
    this_seed = seed
    while iter_id < MAX_ITER:
        active_learn = active_learning(iter_id=iter_id, data_dir=DATA_DIR, pipeline_dir=pipeline_dir, new_sample_size=NEW_SAMPLE_SIZE_LIST[iter_id], seed=this_seed)
        train_tasks = []
        for i in range(NUM_TRAIN):
            this_seed += 1
            train = training(active_learn, iter_id=iter_id, instance_id=i, data_dir=DATA_DIR, pipeline_dir=pipeline_dir, epochs=EPOCHS, seed=this_seed)
            train_tasks.append(train)
        [t.result() for t in train_tasks]   #FIXME: DO I need this?
        this_seed += 1
        iter_id += 1
    #FIXME: Do I need to return anything?

def teach(seed):
    this_seed = seed
    submitted_pipelines = []
    async_pipeline = learner.as_async(teach_single_pipeline)
    for pipeline in range(NUM_PIPELINE):
        pipeline_dir = os.path.join(EXP_DIR, f"pipeline_{pipeline}")
        os.makedirs(pipeline_dir, exist_ok=False)
        os.makedirs(os.path.join(pipeline_dir, 'model'), exist_ok=False)
        os.makedirs(os.path.join(pipeline_dir, 'loss'), exist_ok=False)
        
        submitted_pipelines.append(async_pipeline(pipeline_dir=pipeline_dir, seed=this_seed))
        print(f'Pipeline-{pipeline} is submitted for execution')
        this_seed += 8192
    [p.result() for p in submitted_pipelines]

if __name__ == "__main__":
    teach(seed=SEED)
    print("Everything is done!")
