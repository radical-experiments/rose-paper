import pickle
import os, sys
sys.path.append("/eagle/RECUP/twang/rose/material_design/material-design-bo/")
from packages.fmfn.target_space import TargetSpace
from surrogates.sklearn_gp import GaussianProcessRegressor

def training():
#    path = "/home/sv6234/Downloads/material_test/exp_res/2d/toy_2d_rose/"        
    path = "/eagle/RECUP/twang/rose/material_design/test_rose_paper/"        
    path_targetSpace = os.path.join(path, "target_space.pkl")
    path_gp = os.path.join(path, "gp.pkl")
    path_bo = os.path.join(path, "bo.pkl")
    
    with open(path_targetSpace, "rb") as f:
        targetSpace = pickle.load(f)
    print("Loading GP from:", path_gp)
    with open(path_gp, "rb") as f:
        gp = pickle.load(f)
    print("Before fit, does GP have X_train_?", hasattr(gp, 'X_train_'))
    if hasattr(gp, 'X_train_'):
        print(f"Before fit, Length = {len(gp.X_train_)}, gp.X_train_ = {gp.X_train_}")
    gp =gp.fit(targetSpace.params, targetSpace.target)
    print("After fit, does GP have X_train_?", hasattr(gp, 'X_train_'))
    if hasattr(gp, 'X_train_'):
        print(f"After fit, Length = {len(gp.X_train_)}, gp.X_train_ = {gp.X_train_}")
    
    with open(path_gp, "wb") as f:
        pickle.dump(gp, f)
    with open(path_bo, "rb") as f:
        bo = pickle.load(f)
    bo._gp = gp
    with open(path_bo, "wb") as f:
        pickle.dump(bo, f)
    
    print("Saved fitted GP back to pickle.")
if __name__ == "__main__":
    training()
