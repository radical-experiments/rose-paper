from pyDOE import lhs
import sys
import multiprocessing
import warnings
import time
sys.path.append("/eagle/RECUP/twang/rose/material_design/material-design-bo/")
from packages.fmfn.target_space import TargetSpace
from packages.fmfn.event import Events, DEFAULT_EVENTS
from packages.fmfn.logger import _get_default_logger

from packages.fmfn.util import UtilityFunction, acq_max, ensure_rng
from packages.fmfn.util_parallel import UtilityFunction_parallel, ensure_rng,acq_max_parallel_3d
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.gaussian_process.kernels import Matern
import sys
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
# sys.path.append("/home/mz1482/project/material_bo/parallel_bo_dgp/")
#sys.path.append("/home/pb8294/Documents/Projects/material/material_test/")
#sys.path.append("/home/sv6234/Downloads/material_test/")
from surrogates.sklearn_gp import GaussianProcessRegressor
from packages.fmfn.bo_plots import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import cross_val_score, KFold
from multiprocessing import Process
import concurrent.futures
import pickle
from random import uniform
import os
from s_data import S_Data

#sys.path.append("/home/sv6234/Downloads/material_test/")
import numpy as np
from scipy.stats.distributions import uniform
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C,Matern, WhiteKernel
from packages.fmfn import BayesianOptimization
from objective_examples.experiments2d import colloid_3d_slice, colloid_toy, colloid_3d_slice_fixed_v
from sklearn.metrics.pairwise import pairwise_distances
#init_data_path = "/home/sv6234/Downloads/material_test/training_data/init_2d/"
init_data_path = "/eagle/RECUP/twang/rose/material_design/material-design-bo/rose_polaris/"
def pre_start():
    # Pre-start function to initialize the environment and load initial data
    pbounds = {'x1': (0.01, 5), 'x2': (0.01, 0.5)}
    #length_scale_bounds = [(0.035,1),(0.035,1)]
    init_data = np.loadtxt(init_data_path + "init_data_kappa_50.csv", delimiter=",")
    x_train = init_data[:, :-1]
    n_dimensions = x_train.shape[1]
    print(x_train.shape)
    print(x_train)

    # Compute initial length scales based on initial data
    n_dimensions = x_train.shape[1]
    initial_length_scales = np.zeros(n_dimensions)
    for i in range(n_dimensions):
        
        Xi = x_train[:, i].reshape(-1, 1)
        print(Xi)
        pairwise_dists = pairwise_distances(Xi)
        print(pairwise_dists)
        non_zero_dists = pairwise_dists[np.nonzero(pairwise_dists)]
        if len(non_zero_dists) > 0:
            initial_length_scales[i] = np.median(non_zero_dists)
        else:
            initial_length_scales[i] = 1.0  # Default value if insufficient data

    print("Initial length scales:", initial_length_scales)

    # Define length scale bounds based on initial length scales
    factor_lower = 0.1
    factor_upper = 2.0

    lower_bounds = initial_length_scales * factor_lower
    upper_bounds = initial_length_scales * factor_upper

    length_scale_bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]

    print("Length scale bounds:", length_scale_bounds)
    #TODO comment this
    #length_scale_bounds = [(0.035,1),(0.035,1)]
    #initial_length_scales= [ 0.56413823, 0.56687677]
    kernel = C(constant_value=1, constant_value_bounds=(0.1, 10)) * RBF(length_scale=initial_length_scales, length_scale_bounds=length_scale_bounds) + WhiteKernel(noise_level=0.5)

    func = colloid_3d_slice(kappa=50, jobname="v2_aniso_2d_kappa_50")
    #func = colloid_toy()
    beta_list = np.array([1,3.5])

    p1 = 0.10
    p2 = 0.20
    
    v1_radius = np.array([0.4, 0.075])  # Radius for the first volume of interest
    v2_radius = np.array([0.25, 0.018])  # Radius for the second volume of interest
    path = "/eagle/RECUP/twang/rose/material_design/test_rose_paper/"
#    path = "/home/sv6234/Downloads/material_test/exp_res/2d/toy_2d_rose/"
    bo = BayesianOptimization(f=func.f_fmfn, pbounds=pbounds, kernel=kernel, random_state=None)
    #bo.parallel_2d(init_data=init_data,path=path, n_iter=10, acq1="ucb_parallel_2d_alpha", beta=beta_list,p1=p1,p2=p2,v1_radius=v1_radius,v2_radius = v2_radius)

    bo._prime_subscriptions()
    bo.dispatch(Events.OPTIMIZATION_START)
    bo.set_gp_params()
    init_len = len(init_data)
    acq1 = "ucb_parallel_2d_alpha"
    beta= beta_list     
    util1 = UtilityFunction_parallel(kind=acq1,kappa=beta,v1_radius=v1_radius,v2_radius = v2_radius)
    acq_fn1 = util1.utility_2d

    bo.space.init_register(init_data[:,0:2],init_data[:,2])
    path_targetSpace = os.path.join(path, "target_space.pkl")
    path_gp = os.path.join(path, "gp.pkl")
    path_random_state = os.path.join(path, "random_state.pkl")
    path_acq = os.path.join(path, "acq.pkl")
    path_sData = os.path.join(path, "s_data.pkl")
    path_bo = os.path.join(path, "bo.pkl")
    with open(path_targetSpace, "wb") as f:
        pickle.dump(bo.space, f)
    with open(path_gp, "wb") as f:
        pickle.dump(bo._gp, f)
    with open(path_random_state, "wb") as f:
        pickle.dump(bo._random_state, f)
    with open(path_acq, "wb") as f:
        pickle.dump(acq_fn1, f)
    with open(path_bo, "wb") as f:
        pickle.dump(bo, f)
    s_data = S_Data()
    s_data.set_p1_p2(p1, p2)
    with open(path_sData, "wb") as f:
        pickle.dump(s_data, f)
    
    
if __name__ == "__main__":
    pre_start()
