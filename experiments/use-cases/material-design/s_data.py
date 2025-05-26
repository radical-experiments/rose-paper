import numpy as np
class S_Data:
    def __init__(self):
        self.s1 = None
        self.s2 = None
        self.s1_arr = None
        self.s2_arr = None
        self.x1 = False
        self.x2 = False
        self.x_der = np.empty((0,2))
        self.y_der = np.empty((0,2))
    def set_data(self, s1, s2, s1_arr, s2_arr):
        self.s1 = s1
        self.s2 = s2
        self.s1_arr = s1_arr
        self.s2_arr = s2_arr
    def set_p1_p2(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    def get_p1_p2(self):
        return self.p1, self.p2
    def get_data(self):
        return self.s1, self.s2, self.s1_arr, self.s2_arr
    def set_x1(self):
        self.x1 = True
        self.x2 = False
    def set_x2(self):
        self.x2 = True
        self.x1 = False
    def set_x_der(self, x_der):
        self.x_der = x_der
    def set_y_der(self, y_der):
        self.y_der = y_der
    def get_x_der(self):
        return self.x_der
    def get_y_der(self):
        return self.y_der
    
