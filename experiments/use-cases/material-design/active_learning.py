import os, sys
import pickle
sys.path.append("/eagle/RECUP/twang/rose/material_design/material-design-bo/")
from packages.fmfn.util_parallel import acq_max_parallel_3d
from packages.fmfn.target_space import TargetSpace
from surrogates.sklearn_gp import GaussianProcessRegressor
from s_data import S_Data

def active_learning():
#    path = "/home/sv6234/Downloads/material_test/exp_res/2d/toy_2d_rose/"
    path = "/eagle/RECUP/twang/rose/material_design/test_rose_paper/"
    path_targetSpace = os.path.join(path, "target_space.pkl")
    path_gp = os.path.join(path, "gp.pkl")
    path_random_state = os.path.join(path, "random_state.pkl")
    path_acq = os.path.join(path, "acq.pkl")
    path_sData = os.path.join(path, "s_data.pkl")
    with open(path_targetSpace, "rb") as f:
        targetSpace = pickle.load(f)
    print("Loading GP in active_learning from:", path_gp)
    with open(path_gp, "rb") as f:
        gp = pickle.load(f)
    print("Just loaded GP, does it have X_train_?", hasattr(gp, 'X_train_'))
    with open(path_random_state, "rb") as f:
        random_state = pickle.load(f)
    with open(path_acq, "rb") as f:
        util_3d = pickle.load(f)
    print(f'typre of util_3d: {type(util_3d)}')
    
    s1 = acq_max_parallel_3d(
        ac=util_3d,
        axis =1,
        gp=gp,
        y_max=targetSpace.target.max(),
        bounds= targetSpace.bounds,
        random_state=random_state
    )
    
    s2 = acq_max_parallel_3d(
        ac=util_3d,
        axis=2,
        gp=gp,
        y_max=targetSpace.target.max(),
        bounds= targetSpace.bounds,
        random_state=random_state
    )
    # What is the shape of s1_arr?
    s1_arr = s1
    s2_arr = s2
    s1 = targetSpace.array_to_params(s1)
    s2 = targetSpace.array_to_params(s2)
    print(f's1: {s1}, s2: {s2}')
    print(f's1_arr: {s1_arr.shape}, s2_arr: {s2_arr.shape}')
    with open(path_sData, "rb") as f:
        s_data = pickle.load(f)
    s_data.set_data(s1,s2,s1_arr,s2_arr)
    with open(path_sData, "wb") as f:
        pickle.dump(s_data, f)
        
#if __name__ == "__main__":
active_learning()
