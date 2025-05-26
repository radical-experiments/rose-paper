import pickle
import os
import numpy as np
def logic_2(iteration,path):
    
    path_sData = os.path.join(path, "s_data.pkl")
    path_bo = os.path.join(path, "bo.pkl")
    with open(path_bo, "rb") as f:
        bo = pickle.load(f)
    if iteration<5:
        return True,True
    else:
        with open(path_sData, "rb") as f:
            s_data = pickle.load(f)
        x_probe1,x_probe2,xp1,xp2 =s_data.get_data()
        x_der = s_data.get_x_der()
        y_der = s_data.get_y_der()
        X_test  = np.vstack((xp1,xp2))
        x1_der,x2_der,_,_ = bo.derivative_mag(X_test)
        print("x_der",x_der)
        _,_,x1_der_max,_ = bo.derivative_mag(x_der)
        _,_,_,x2_der_max = bo.derivative_mag(y_der)
        p1,p2 = s_data.get_p1_p2()
        with open(path_bo, "wb") as f:
            pickle.dump(bo, f)
        with open(path_bo, "wb") as f:
            pickle.dump(bo, f)
        if x1_der<p1*x1_der_max and x2_der<p2*x2_der_max:
            print("both blocked")
            # need to break the loop
            return False,False
        elif x1_der<p1*x1_der_max and x2_der>p2*x2_der_max:
            print("x derivative is blocked")
            return False,True
        elif x1_der>p1*x1_der_max and x2_der<p2*x2_der_max:
            print("y derivative is blocked")
            return True,False
        else:
            print("none is blocked")
            return True,True
        
    