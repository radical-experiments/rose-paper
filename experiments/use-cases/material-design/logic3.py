import numpy as np
import pickle
import os
def logic3(iteration,path,is_x1,is_x2):
    """
    This function returns a string indicating that it is logic3.
    """
    path_sData = os.path.join(path, "s_data.pkl")
    with open(path_sData, "rb") as f:
        s_data = pickle.load(f)
    x_probe1,x_probe2,xp1,xp2 =s_data.get_data()
    if iteration <5:
        x_der=s_data.get_x_der()
        print(" get x_der",x_der)
        y_der=s_data.get_y_der()
        x_der = np.append(x_der,xp1.reshape(1,-1),axis=0)
        y_der = np.append(y_der,xp2.reshape(1,-1),axis=0)
        s_data.set_x_der(x_der)
        s_data.set_y_der(y_der)
        print("x_der",x_der)
        print(s_data.get_x_der())
#                 plot_object.gp_with_derivative_selections(self._gp,testx,5,x_der,y_der)
        f=open(path+"x_der.csv","a")
        f.write(str(xp1[0])+","+str(xp1[1])+","+str(iteration)+",first5"+"\n")
        f.close()
        f=open(path+"y_der.csv","a")
        f.write(str(xp2[0])+","+str(xp2[1])+","+str(iteration)+",first5"+"\n")
        f.close()
        with open(path_sData, "wb") as f:
            pickle.dump(s_data, f)
    else:
        x_der=s_data.get_x_der()
        y_der=s_data.get_y_der()
        if is_x1 and is_x2:
            
            x_der = np.append(x_der,xp1.reshape(1,-1),axis=0)
            y_der = np.append(y_der,xp2.reshape(1,-1),axis=0)
#                     plot_object.gp_with_derivative_selections(self._gp,testx,5,x_der,y_der)
            f=open(path+"x_der.csv","a")
            f.write(str(xp1[0])+","+str(xp1[1])+","+str(iteration)+",noblock"+"\n")
            f.close()
            f=open(path+"y_der.csv","a")
            f.write(str(xp2[0])+","+str(xp2[1])+","+str(iteration)+",noblock"+"\n")
            f.close()
            s_data.set_x_der(x_der)
            s_data.set_y_der(y_der)
        elif is_x1:
            x_der = np.append(x_der,xp1.reshape(1,-1),axis=0)
#                     plot_object.gp_with_derivative_selections(self._gp,testx,5,x_der,y_der)
            f=open(path+"x_der.csv","a")
            f.write(str(xp1[0])+","+str(xp1[1])+","+str(iteration)+",yder_x2_block"+"\n")
            f.close()
            s_data.set_x_der(x_der)
        elif is_x2:
            y_der = np.append(y_der,xp2.reshape(1,-1),axis=0)
#                     plot_object.gp_with_derivative_selections(self._gp,testx,5,x_der,y_der)
            f=open(path+"y_der.csv","a")
            f.write(str(xp2[0])+","+str(xp2[1])+","+str(iteration)+",xder_x1_block"+"\n")
            f.close()
            s_data.set_y_der(y_der)
        with open(path_sData, "wb") as f:
            pickle.dump(s_data, f)
        
    
            
            
        
    
    