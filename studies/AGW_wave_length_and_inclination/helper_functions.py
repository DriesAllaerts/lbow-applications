"""
Helper functions for the AGW wave length and inclination study
"""


import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, argrelmax
from lbow.oneDimensional.steadystate import HalfPlaneModel
from skimage.feature import peak_local_max

# ---------------------------------------------------
# Functions related to simple wavelength calculations
# ---------------------------------------------------

def autocorrelation_fft(f):
    fc = np.fft.fft(f)
    S = fc*np.conj(fc)
    R = np.fft.ifft(S)
    return R/R[0]

def effective_wavelength(w,dx):
    return 2*dx*np.argmin(autocorrelation_fft(w)[:int(w.size/2)])

# -------------------------------------------------------
# Functions related to finding the wave field inclination
# -------------------------------------------------------

def create_model(U,N,L,verbose=0):
    # Hill shape
    hmax = 0.1*L                     # Height of the hill [m]
    h = lambda x: hmax/(1+(x/L)**2)  # Shape function of the hill

    # Numerical parameters
    Lx = 2000 * L                    # Length of the numerical domain [m]
    Nx = 2*int(1e4)                  # Number of grid points

    Ls = 2*np.pi*U/N
    Lz = 3.5 * Ls
    Nz = int(3.5 * 60 + 1)
    if verbose>0: print('Number of vertical levels is {} '.format(Nz))

    # Numerical grid
    xs,dx = np.linspace(-Lx/2,Lx/2,Nx,endpoint=False,retstep=True)
    if verbose>0: print('Horizontal grid resolution is {} m'.format(dx))

    zs,dz = np.linspace(0,Lz,Nz,retstep=True)
    if verbose>0: print('Vertical grid resolution is {} m'.format(dz))

    return xs,zs,HalfPlaneModel(xs,h(xs),U,N)

def find_peaks(xs, zs, var, threshold_rel=0.5, min_distance=2, exclude_border=True, verbose=0):
    # Use peak_local_max to find coordinates of local extrema
    peak_indices = peak_local_max(np.abs(var.T),
                                  min_distance=min_distance,
                                  threshold_rel=threshold_rel,
                                  exclude_border=exclude_border)
    
    Npeaks = peak_indices.shape[0]
    # Sort based on z coordinate
    peak_indices = peak_indices[np.argsort(peak_indices[:, 0])]
    # Convert from indices to coordinates
    xp = xs[peak_indices[:,1]]
    zp = zs[peak_indices[:,0]]
    
    
    if verbose>0: print('Number of peaks found is {}'.format(Npeaks))
    if verbose>1:
        print('Peak coordinates:')
        for i in range(Npeaks):
            print(xp[i],zp[i])
    
    return xp, zp

def parabolic_slope_fz(x,z):
    # x = f(z)
    def fit_func(z, b):
        # Curve fitting function
        return b * np.sqrt(z)
    
    # Curve fitting
    params = curve_fit(fit_func, z, x)
    
    return params[0][0]

def parabolic_slope_fz_3dof(x,z):
    # x = f(z)
    def fit_func(z, b0, b1, z0):
        # Curve fitting function
        return b0 * np.sqrt(z-z0) + b1
    
    # Curve fitting
    params = curve_fit(fit_func, z, x,
                       bounds=([-np.inf,-np.inf,-np.inf],[np.inf,np.inf,0]),
                       max_nfev=1000*x.size,
                       p0 = [0.1,0.1,-0.1])
    
    return params[0][0], params[0][1], params[0][2]

def linear_slope_fz(z0,b):
    # Slope of line tangent to square root function x = b sqrt(z) at z0
    return 0.5 * b / np.sqrt(z0)

def linear_slope_fz_3dof(z,b0,z0):
    # Slope of line tangent to square root function x = b sqrt(z) at z
    return 0.5 * b0 / np.sqrt(z-z0)

def find_inclination_eta(U,N,L,verbose=0):
    Fr = U/N/L
    Ls = 2*np.pi*U/N
    
    # Create model
    xs,zs,model = create_model(U,N,L,verbose)
    
    # Solve for eta
    eta = model.solve('eta',zs,space='real')
    
    # Find extrema in of the 2D contour
    threshold_rel=min(0.6,1/(3.5*Fr))
    exclude_border = True
    
    xp, zp = find_peaks(xs,zs,eta,min_distance=5,
                        threshold_rel=threshold_rel,
                        exclude_border=exclude_border,
                        verbose=verbose)
    # Find extrema
    # Parabolic fit of the extrema
    b = parabolic_slope_fz(xp, zp)
    # Find extremum closest to the arc with radius 9/8*Ls
    R = np.sqrt(xp**2+zp**2)
    idx = (np.abs(R - 9/8*Ls)).argmin()
    # x coordinate of first maximum above ground
    x1 = b*np.sqrt(zp[idx])
    # slope of parabolic fit at x1
    bprime = linear_slope_fz(zp[idx],b)

    # Inclination
    return np.arctan2(1,bprime), x1, zp[idx]

def find_inclination_w(U,N,L,verbose=0,z1=None):
    Fr = U/N/L
    Ls = 2*np.pi*U/N
    
    # Create model
    xs,zs,model = create_model(U,N,L,verbose)
    
    # Solve for w
    w = model.solve('w',zs,space='real')
    
    # Find extrema in of the 2D contour
    threshold_rel=min(0.5,1/(6*Fr))
    exclude_border = True
    
    xp, zp = find_peaks(xs,zs,w,min_distance=5,
                        threshold_rel=threshold_rel,
                        exclude_border=exclude_border,
                        verbose=verbose)
    
    # Find extrema
    # Parabolic fit of the extrema
    b0, b1, z0 = parabolic_slope_fz_3dof(xp, zp)
    if z1 is None:
        # Find extremum closest to the arc with radius 7/8*Ls
        R = np.sqrt(xp**2+zp**2)
        idx = (np.abs(R - 7/8*Ls)).argmin()
        z1 = zp[idx]
    # x coordinate of first maximum above ground
    x1 = b0*np.sqrt(z1-z0)+b1
    # slope of parabolic fit at x1
    bprime = linear_slope_fz_3dof(z1,b0,z0)

    # Inclination
    return np.arctan2(1,bprime)
