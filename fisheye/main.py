from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist

def is_iterable(obj):
    try:
        some_object_iterator = iter(obj)
        return True
    except TypeError as te:
        return False

class fisheye():

    def __init__(self,R,mode='default',d=4,xw=0.25):
        self.R = R
        self.d = d

        assert(xw >= 0.0 and xw<=1.0)

        if mode == 'Sarkar':
            self.xw = 0
        else:
            self.xw = xw
        self.mode = mode

        self._compute_parameters()

    def _compute_parameters(self):
        d = self.d
        xw = self.xw

        if self.mode in ('default', 'Sarkar'):
            self.f1 = lambda x: (x+self.d*x) / (self.d*x + self.A2)
            self.f1_inverse = lambda x: self.A2 * x / (self.d * (1-x) + 1)
        elif self.mode == 'sqrt':
            self.f1 = lambda x: (self.d/self.A2*x)**(1./self.d)
            self.f1_inverse = lambda x: self.A2 / self.d * x**self.d

        self.f2 = lambda x: 1 - (-1./self.A1 + np.sqrt(1/self.A1**2 + 2*(1-x)/self.A1))
        self.f2_inverse = lambda x: x - self.A1/2 * (1-x)**2

        if xw == 0.0:
            self.A1 = 0
            self.A2 = 1
        elif xw == 1.0:
            self.A2 = 0
            self.A1 = 1
        else:


            if self.mode in ('default', 'Sarkar'):
                X = np.array([[ xw**2/2., 1 - ((d+1)*xw / (d*xw+1)) ],
                              [ xw,         - (d+1) / (d*xw+1)**2   ]]);
            elif self.mode == 'sqrt':
                X = np.array([[ xw**2/2, ((1-xw)**d)/d],
                              [xw, -(1-xw)**(d-1)]])

            b = -np.array([xw-1,1])
            self.A1, self.A2 = np.linalg.inv(X).dot(b)

        xc = self.A1/2 * xw**2 + xw
        self.xc = 1 - xc




    def set_magnification(self,d):
        self.d = d
        self._compute_parameters()

    def set_demagnification_width(self,xw):
        assert(xw >= 0.0 and xw<=1.0)
        if self.mode == 'Sarkar':
            self.xw = 0
        else:
            self.xw = xw
        self._compute_parameters()

    def set_radius(self,R):
        self.R = R

    def set_mode(self,mode):
        self.mode = mode
        self._compute_parameters()

    def set_focus(self,focus):

        if not is_iterable(focus):
            focus = np.array([focus])

        if not type(focus) == np.ndarray:
            focus = np.array(focus)

        self.focus = focus

    def radial(self,pos):

        is_scalar = not is_iterable(focus)

        if is_scalar:
            pos = np.array([[pos]])
        else:
            if len(pos.shape) == 1:
                pos = pos.reshape((len(pos.shape),1))

    def fisheye_raw(self,r):
        result = np.copy(r)
        if self.xc > 0 and self.xc < 1:
            result[r<=self.xc] = self.f1(result[r<=self.xc])
            result[r>self.xc] = self.f2(result[r>self.xc])
        elif self.xc == 0:
            result = self.f1(result)

        return result

    def fisheye_raw_inverse(self,r):
        result = np.copy(r)
        if self.xw > 0 and self.xw < 1:
            result[r<=1-self.xw] = self.f2_inverse(result[r<=1-self.xw])
            result[r>1-self.xw] = self.f1_inverse(result[r>1-self.xw])
        elif self.xw == 0:
            result = self.f2_inverse(result)

        return result

    def radial_2D(self,pos,inverse=False):

        if not type(pos) == np.ndarray:
            pos = np.array(pos)

        original_shape = pos.shape

        if len(pos.shape) == 1:
            pos = pos.reshape((1,pos.shape[0]))

        theta = np.arctan(pos[:,1]-self.focus[1], pos[:,0]-self.focus[0])

        x = cdist(pos, self.focus.reshape(1,len(self.focus))).flatten() / self.R


        if not inverse:
            newx = self.fisheye_raw(x)
        else:
            newx = self.fisheye_raw_inverse(x)

        newpos = np.empty_like(pos)

        newpos[:,0] = self.focus[0] + np.cos(theta) * self.R * newx
        newpos[:,1] = self.focus[1] + np.sin(theta) * self.R * newx

        newpos = newpos.reshape(original_shape)

        return newpos

    def inverse_radial_2D(self,pos):
        return self.radial_2D(pos, inverse=True)

if __name__=="__main__":
    F = fisheye(1,mode='sqrt',xw=0)
    
    F.set_focus([0.5,0.5])

    print(F.focus)
    result = F.inverse_radial_2D(F.radial_2D([0.55,0.5]))
    print(result, result.shape)
    
