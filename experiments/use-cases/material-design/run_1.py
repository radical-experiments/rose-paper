import pre_start
import logic1
import logic2
import logic3
import warnings
import pickle
from packages.fmfn.bayesian_optimization import Events

import os
import sys
import shutil
import numpy as np

import radical.pilot as rp

# ------------------ ROSE Releated imports -------------
from rose.learner import ActiveLearner
from rose.engine import Task, ResourceEngine
# ------------------------------------------------------
engine = ResourceEngine({'resource': 'anl.polaris', 
                         'runtime' : 720,
                         'access_schema':'interactive',
                         'project' : "RECUP",
                         'cores'   : 32,
                         'gpus'    : 4})
#engine = ResourceEngine({'resource': 'anl.polaris', 
#                         'runtime' : 60,
##                         'runtime' : 500,
#                         'queue'   : 'debug',
##                         'queue'   : 'preemptable',
#                         'cores'   : 32,
#                         'gpus'    : 4,
#                         'project' : "RECUP"})

learner = ActiveLearner(engine=engine)
code_path = f'{sys.executable} {os.getcwd()}'

def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))

def gofr_fn(phi,dirname):
    file_name = os.path.join(dirname, 'gofr.txt')
    with open(file_name, 'r') as ff:
        lines = ff.read().splitlines()
        last_50 = lines[-50:]
    data = []
    for line in last_50 :
        number_strings = line.split() # Split the line on runs of whitespace
        numbers = [float(n) for n in number_strings] # Convert to integers
        data.append(numbers) # Add the "row" to your list.
    data = np.asarray(data)
    gr_phi = data[:,2]*phi
    y1 = np.amax(gr_phi)
    return y1 


# step-1 prepare the simulation input locally
def prepare_simualtion_input(V0, phi, kappa, exp_dir, file_template):
    t = 0
    while os.path.exists(os.path.join(exp_dir, f"phi{t}")):
        t += 1

    dirname = os.path.join(exp_dir, f"phi{t}")
    os.makedirs(dirname, exist_ok=False)
    with open(file_template, 'r') as file:
        template = file.read()

    # Replace placeholders
    filled = template.replace('{{V0_placeholder}}', str(V0)) \
                    .replace('{{phi_placeholder}}', str(phi)) \
                    .replace('{{kappa_placeholder}}', str(kappa))

    # Write filled input to new file (or overwrite original if needed)
    lammps_file = os.path.join(dirname, 'in.lammps')
    with open(lammps_file, 'w') as file:
        file.write(filled)
    
    return lammps_file,dirname




#This call it works
#prepare_simualtion_input(123,456,789,
#                         "/eagle/RECUP/twang/rose/material_design/test_rose_component/experiment", 
#                         "/eagle/RECUP/twang/rose/material_design/test_rose_component/in.lammps.template")



# step-2 execute the simulation with the prepared input file on HPC
# Here file_id is the file absolute path of the lammps.in to be simulate
# dir_name is the directory where we want to copy/link the output gofr.txt into
@learner.simulation_task
def simulation(*args, file_id, dir_name):
    print(f"DEBUG!! file_id = {file_id}, dir_name = {dir_name}")
    pre_exec_list = ["source /eagle/RECUP/twang/rose/material_design/setup_lammps_gnu.sh", 
                     "export MPICH_GPU_SUPPORT_ENABLED=1"]
    return Task(executable=f'/eagle/RECUP/twang/rose/material_design/lammps-29Aug2024/src/lmp_polaris_gnu_kokkos -in {file_id} -k on g 1 t 8 -sf kk -pk kokkos', 
                cores_per_rank = 8,
                gpus_per_rank = 1,
                threading_type = rp.OpenMP,
                gpu_type = rp.CUDA,
                pre_exec = pre_exec_list,
                output_staging = [{'source': 'gofr.txt',  'target': f'{dir_name}/gofr.txt'}])


@learner.training_task
def training(*args):
    return Task(executable=f'{code_path}/training.py', cores_per_rank = 1)

@learner.active_learn_task
def active_learning(*args):
    return Task(executable=f'{code_path}/active_learning.py', cores_per_rank = 1)


def teach():
    pre_start.pre_start()   
    
    iter_id = 0
#    path = "/home/sv6234/Downloads/material_test/exp_res/2d/toy_2d_rose/"
    path = "/eagle/RECUP/twang/rose/material_design/test_rose_paper/"
    path_phi_dir = "/eagle/RECUP/twang/rose/material_design/test_rose_paper/phi_dir"
#    template_path ="/home/sv6234/Downloads/material_test/exp_res/2d/toy_2d_rose/template.in"
    template_path ="/eagle/RECUP/twang/rose/material_design/test_rose_component/in.lammps.template"
    path_sData = path + "s_data.pkl"
    path_targetSpace = os.path.join(path, "target_space.pkl")
    MAX_ITER = 25
    while iter_id < MAX_ITER:
        # we start with training as we have a pre-simulated data
        # the iter_id will help of avoiding overwriting the files
        logic1.logic_1(iter_id, path)
        
        train = training()
        active_learn = active_learning(train)
        active_learn.result()

        is_x1, is_x2 = logic2.logic_2(iter_id, path=path)

        simulation_tasks = []
        if is_x1:
            with open(path_sData, "rb") as f:
                s_data = pickle.load(f)
            s_data.set_x1()
            x_probe1,_,_,_=s_data.get_data()
            print("DEBUG!! x_probe1 = ", x_probe1)
            x_probe1 = np.array(list(x_probe1.values()))
#            x_probe1 = np.asarray(x_probe1, dtype=float)
            V0_1 = x_probe1[0]
            phi_1 = x_probe1[1]
            file_1,dirname_1=prepare_simualtion_input(V0_1, phi_1, 50,path_phi_dir,template_path)
            with open(path_targetSpace, "rb") as f:
                target_space = pickle.load(f)
            try:
                target_1 = target_space._cache[_hashable(x_probe1)]
            except KeyError:
                sim1 = simulation(active_learn, file_id=file_1, dir_name=dirname_1)
                simulation_tasks.append(sim1)
            with open(path_targetSpace, "wb") as f:
                pickle.dump(target_space, f)
            with open(path_sData, "wb") as f:
                pickle.dump(s_data, f)


        if is_x2:
            with open(path_sData, "rb") as f:
                s_data = pickle.load(f)
            s_data.set_x1()
            _,x_probe2,_,_=s_data.get_data()
#            x_probe2 = np.asarray(x_probe2, dtype=float)
            x_probe2 = np.array(list(x_probe2.values()))
            V0_2 = x_probe2[0]
            phi_2 = x_probe2[1]
            file_2,dirname_2=prepare_simualtion_input(V0_2, phi_2, 50,path_phi_dir,template_path)
            with open(path_targetSpace, "rb") as f:
                target_space = pickle.load(f)
            try:
                target_2 = target_space._cache[_hashable(x_probe2)]
            except KeyError:
                sim2 = simulation(active_learn, file_id=file_2, dir_name=dirname_2)
                simulation_tasks.append(sim2)
            with open(path_targetSpace, "wb") as f:
                pickle.dump(target_space, f)
            with open(path_sData, "wb") as f:
                pickle.dump(s_data, f)

        [s.result() for s in simulation_tasks]
        if is_x1:
            target_1 = gofr_fn(phi_1,dirname_1)
            with open(path_targetSpace, "rb") as f:
                target_space = pickle.load(f)
            target_space.register(x_probe1, target_1)
            print(f"x_probe1 = {x_probe1}, target_1 = {target_1}")
            with open(path_targetSpace, "wb") as f:
                pickle.dump(target_space, f)
         
        if is_x2:
            target_2 = gofr_fn(phi_2,dirname_2)
            with open(path_targetSpace, "rb") as f:
                target_space = pickle.load(f)
            target_space.register(x_probe2, target_2)
            print(f"x_probe2 = {x_probe2}, target_2 = {target_2}")
            with open(path_targetSpace, "wb") as f:
                pickle.dump(target_space, f)

        with open(path_targetSpace, "rb") as f:
            target_space = pickle.load(f)
        print(f"target_space.params = {target_space.params}, target_space.target = {target_space.target}")
		
        
        with open(path +'bo.pkl', "rb") as f:
          bo = pickle.load(f)
        bo.dispatch(Events.OPTIMIZATION_STEP)
        logic3.logic3(iter_id, path,is_x1,is_x2)
        with open(path +'bo.pkl', "rb") as f:
            bo = pickle.load(f)
        obj = bo._gp
        st = 5 + iter_id
        bo.save_object(obj,st, path)
        iter_id +=1
        with open(path +'bo.pkl', "wb") as f:
            pickle.dump(bo, f)

if __name__ == "__main__":
    teach()
