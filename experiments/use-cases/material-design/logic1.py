import pickle
import os
import warnings
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import time
def logic_1(iteration,path):
    path_bo = os.path.join(path, "bo.pkl")
    path_sData = os.path.join(path, "s_data.pkl")
    path_gp = os.path.join(path, "gp.pkl")
    path_targetSpace = os.path.join(path, "target_space.pkl")
    with open(path_bo, "rb") as f:
        bo = pickle.load(f)
    with open(path_gp, "rb") as f:
        gp = pickle.load(f)
    with open(path_targetSpace, "rb") as f:
        targetSpace = pickle.load(f)
    if iteration != 0 and ((iteration < 10 and iteration % 2 == 0) or (iteration >= 10 and iteration % 5 == 0)):

        initial_length_scales = bo.compute_initial_length_scales(targetSpace.params)
        length_scale_bounds = bo.update_length_scale_bounds(initial_length_scales, factor_lower=0.1, factor_upper=3.0,min_lower_apply_x = None, max_upper_apply_x = None, min_lower_apply_y = 0.08, max_upper_apply_y = 0.2 )
#        length_scale_bounds = bo.update_length_scale_bounds(initial_length_scales, factor_lower=0.1, factor_upper=3.0,min_lower_apply = False)
        # Update the GP kernel

        # Update the GP kernel with new length scales and bounds
        gp.kernel = (
            C(constant_value=1,constant_value_bounds=(0.1, 10))
            * RBF(length_scale=initial_length_scales, length_scale_bounds=length_scale_bounds)
            + WhiteKernel(noise_level=0.5)
        )
        gp.kernel_ = None
        
        #bo.update_gp_kernel(initial_length_scales, length_scale_bounds)
        print(f"Updated length scales at iteration {iteration}: {initial_length_scales}")
        print(f"Updated length scale bounds: {length_scale_bounds}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(gp, 'X_train_'):
                print("Inside logic1, before gp.fit, gp.X_train_ = {gp.X_train_}")
            gp=gp.fit(targetSpace.params, targetSpace.target)
            if hasattr(gp, 'X_train_'):
                print("Inside logic1, after gp.fit, gp.X_train_ = {gp.X_train_}")
            bo._gp = gp
        print("Fitting GP with updated kernel...")
        print(f"Optimized Kernel after fitting at iteration {iteration}: {bo._gp.kernel_}")

        with open(path_bo, "wb") as f:
            pickle.dump(bo, f)
        with open(path_gp, "wb") as f:
            pickle.dump(gp, f)
        with open(path_targetSpace, "wb") as f:
            pickle.dump(targetSpace, f)
    # Why first 5
    # --- At beginning we don't have enough data to check derivative magnitude so we run without checking for 5 points.
    if iteration < 5:
        if len(bo._space) == 0:
            print(bo._space)
            print(len(bo._space))
            return bo._space.array_to_params(bo._space.random_sample())
