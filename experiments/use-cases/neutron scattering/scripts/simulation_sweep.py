#!/usr/bin/env python

import os,sys
sys.path.insert(0,os.path.expanduser("~/g2full/GSAS-II/GSASII/"))
import GSASIIscriptable as G2sc

from mpi4py import MPI
import numpy as np
import time
import h5py
from contextlib import redirect_stdout      #for debug

import sweep_utils as su

gpx = []
phase = []


def cubic_lattice(prm):
    """ This function uses a 1D grid.

        Parameters for cubic lattice:
        a = b = c
        alpha = beta = gamma = 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[0] # c

    # Angles
    phase['General']['Cell'][4] = 90 # alpha
    phase['General']['Cell'][5] = 90 # beta
    phase['General']['Cell'][6] = 90 # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def trigonal_lattice(prm):
    """ This function uses a 2D grid.

        Parameters for trigonal lattice:
        a = b = c
        alpha = beta = gamma != 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[0] # c

    # Angles
    phase['General']['Cell'][4] = prm[1] # alpha
    phase['General']['Cell'][5] = prm[1] # beta
    phase['General']['Cell'][6] = prm[1] # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y



def tetragonal_lattice(prm):
    """ This function uses a 2D grid.

        Parameters for tetragonal lattice:
        a = b != c
        alpha = beta = gamma = 90
    """

    global gpx
    global phase

    # Unit cell parameters
    # Lattice
    # c should be differe
    phase['General']['Cell'][1] = prm[0] # a
    phase['General']['Cell'][2] = prm[0] # b
    phase['General']['Cell'][3] = prm[1] # c

    # Angles
    phase['General']['Cell'][4] = 90 # alpha
    phase['General']['Cell'][5] = 90 # beta
    phase['General']['Cell'][6] = 90 # gamma

    # Compute simulation
    gpx.data['Controls']['data']['max cyc']=0
    gpx.do_refinements([{}])

    x = gpx.histogram(0).getdata('x')
    y = gpx.histogram(0).getdata('ycalc')

    return x, y


def _sweep_uniform(sz, low, high):
    uniform_numbers = np.linspace(low, high, sz)
    return uniform_numbers.reshape(-1, 1)

def sweep_cubic_sample(sz, a_min, a_max):
    a_arr = _sweep_uniform(sz, a_min, a_max)
    return a_arr

def sweep_trigonal_sample(sz_a, sz_alpha, a_min, a_max, alpha_min, alpha_max):
    a_arr = _sweep_uniform(sz_a, a_min, a_max)
    alpha_arr = _sweep_uniform(sz_alpha, alpha_min, alpha_max)
    a_mesh, alpha_mesh = np.meshgrid(a_arr.flatten(), alpha_arr.flatten(), indexing='ij')
    return np.column_stack((a_mesh.ravel(), alpha_mesh.ravel()))

def sweep_tetragonal_sample(sz_a, sz_c, a_min, a_max, c_min, c_max):
    a_arr = _sweep_uniform(sz_a, a_min, a_max)
    c_arr = _sweep_uniform(sz_c, c_min, c_max)
    a_mesh, c_mesh = np.meshgrid(a_arr.flatten(), c_arr.flatten(), indexing='ij')
    return np.column_stack((a_mesh.ravel(), c_mesh.ravel()))

def _remove_out_of_box(sym, sample_in, bounding_box, min_diff):
    assert bounding_box.shape[1] == 2       #[[begin1, end1], [begin2, end2], [begin3, end3], ...]
    if sym == "cubic":
        assert sample_in.shape[1] == 1
        assert bounding_box.shape[0] == 1   #on dim a
        idx = (sample_in[:,0] >= bounding_box[0,0]) & (sample_in[:,0] <= bounding_box[0,1])
        sample_in = sample_in[idx]
    elif sym == "trigonal":
        assert sample_in.shape[1] == 2
        assert bounding_box.shape[0] == 2   #on dim a, angle
        idx = (sample_in[:,0] >= bounding_box[0,0]) & (sample_in[:,0] <= bounding_box[0,1]) & \
              (sample_in[:,1] >= bounding_box[1,0]) & (sample_in[:,1] <= bounding_box[1,1]) & \
              (abs(sample_in[:,1] - 90) > min_diff)   #angle different from 90 degree
        sample_in = sample_in[idx]
    elif sym == "tetragonal":
        assert sample_in.shape[1] == 2
        assert bounding_box.shape[0] == 2   #on dim a, c
        idx = (sample_in[:,0] >= bounding_box[0,0]) & (sample_in[:,0] <= bounding_box[0,1]) & \
              (sample_in[:,1] >= bounding_box[1,0]) & (sample_in[:,1] <= bounding_box[1,1]) & \
              (abs(sample_in[:,0] - sample_in[:,1]) > min_diff)  #two edge should be different
        sample_in = sample_in[idx]
    else:
        exit("Error! Unrecognized sym argument = {} in _remove_out_of_box function call".format(sym))
    return sample_in

#All rank will execute this function
#Here sz_g is the global total number of samples to simulate, which means each rank will only do sz_g/size samples
#Here x_min, x_max are global range, and need to compute some of local range inside this function
#Different rank will have different local a_min and a_max that are non-overlapping, but alpha/c are the same
def create_all_samples(rank, size, sz_g,
                       a_min1, a_max1, 
                       a_min2, a_max2, alpha_min, alpha_max, 
                       c_min, c_max, 
                       cubic_bounding_box, 
                       trigonal_bounding_box, alpha_diff_trigonal,
                       tetragonal_bounding_box, ac_diff_tetragonal):

    sz = int(sz_g / size)
    diff = (a_max1 - a_min1) / (sz * size - 1)
    start_off = a_min1 + diff * sz * rank
    end_off   = a_min1 + diff * (sz * (rank + 1) - 1)
    print("IMPORTANT: rank = {}, for cubic, sz = {}, start_off = {}, end_off = {}!!!".format(rank, sz, start_off, end_off))

    sample_cubic = sweep_cubic_sample(sz, start_off, end_off)
    sample_cubic = _remove_out_of_box("cubic", sample_cubic, cubic_bounding_box, None)

    sz_dim2 = int(np.sqrt(sz_g))
    sz_a = int(sz_dim2 / size)
    assert sz_a >= 1, "Each thread only has little a_grid to do, need different algorithm!"
    diff = (a_max2 - a_min2) / (sz_a * size - 1)
    start_off = a_min2 + diff * sz_a * rank
    end_off   = a_min2 + diff * (sz_a * (rank + 1) - 1)
    print("IMPORTANT: rank = {}, for trigonal/tetragonal, sz_a = {}, sz_dim2 = {}, start_off = {}, end_off = {}!!!".format(rank, sz_a, sz_dim2, start_off, end_off))

    sample_trigonal = sweep_trigonal_sample(sz_a, sz_dim2, start_off, end_off, alpha_min, alpha_max)
    sample_trigonal = _remove_out_of_box("trigonal", sample_trigonal, trigonal_bounding_box, alpha_diff_trigonal)

    sample_tetragonal = sweep_tetragonal_sample(sz_a, sz_dim2, start_off, end_off, c_min, c_max)
    sample_tetragonal = _remove_out_of_box("tetragonal", sample_tetragonal, tetragonal_bounding_box, ac_diff_tetragonal)

    return sample_cubic, sample_trigonal, sample_tetragonal

def _sim_impl(rank, size, sym, conffile_name, sample):
    start = time.time()

    gParameters = su.read_config_file(conffile_name)
    # Create project
    name = gParameters['name'] +'_rank' + str(rank)
    path_in = gParameters['path_in']
    path_out = gParameters['path_out']
    name_out = gParameters['name_out']
    global gpx
    gpx = G2sc.G2Project(newgpx=path_out+name+'.gpx')
    
    # Add phase: Requires CIF file
    cif = path_in + gParameters['cif']
    global phase
    phase = gpx.add_phase(cif,phasename=name,fmthint='CIF')
    
    # Get instrument file specification
    instprm = path_in + gParameters['instprm']
    # Histogram range
    Tmin = gParameters['tmin']
    Tmax = gParameters['tmax']
    Tstep = gParameters['tstep']
    hist = gpx.add_simulated_powder_histogram(name+'TOFsimulation',instprm,Tmin,Tmax,Tstep,phases=gpx.phases())
    hist.SampleParameters['Scale'][0] = 1000.
    # Set to no-background
    hist['Background'][0][3]=0.0
    
    symmetry = gParameters['symmetry']
    assert symmetry == sym, "symmetry = {} and sym = {}".format(symmetry, sym)
    
    # Configure sweep according to symmetry
    if symmetry == 'cubic':
        sweepf_ = cubic_lattice
    elif symmetry == 'trigonal':
        sweepf_ = trigonal_lattice
    elif symmetry == 'tetragonal':
        sweepf_ = tetragonal_lattice
    else:
        exit("Do not recognize symmetry of {}".format(symmetry))
    
    # Distribute computation
    nsim, histosz = su.grid_sample(rank, size, sweepf_, sample, path_out, name_out + '_' + symmetry)
    end = time.time()
    print('----------------------------------------------------------')
    print("Rank = {}, Number of simulations ({}): {}, size of histogram: {}, cost {} seconds".format(rank, symmetry, nsim, histosz, end-start))


def main():

    start = time.time()
    if ( len ( sys.argv ) != 5 ) :
        print(sys.argv)
        sys.stderr.write("Usage: python simulation_sweep.py num_sample_total "
                         "conf_name_cubic, conf_name_trigonal, conf_name_tetragonal\n")
        sys.exit(0)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print("Rank = {} out of size = {}".format(rank, size))

    num_sample_total = int(sys.argv[1])
    conf_name_cubic = sys.argv[2]
    conf_name_trigonal = sys.argv[3]
    conf_name_tetragonal = sys.argv[4]

#FIXME
    a_min1 = 2.5
    a_max1 = 5.5
    a_min2 = 3.5
    a_max2 = 4.5
    alpha_min = 30.0
    alpha_max = 119.5
    c_min = 3.5005
    c_max = 4.4995

    cubic_bounding_box = np.array([[a_min1, a_max1]])
    trigonal_bounding_box = np.array([[a_min2, a_max2], [alpha_min, alpha_max]])
    tetragonal_bounding_box = np.array([[a_min2, a_max2], [c_min, c_max]])
    alpha_diff_trigonal = 0.2
    ac_diff_tetragonal = 0.001

    sample_cubic, sample_trigonal, sample_tetragonal = create_all_samples(rank, size,
                       num_sample_total,
                       a_min1, a_max1, 
                       a_min2, a_max2, alpha_min, alpha_max, 
                       c_min, c_max, 
                       cubic_bounding_box, 
                       trigonal_bounding_box, alpha_diff_trigonal,
                       tetragonal_bounding_box, ac_diff_tetragonal)
    print(sample_cubic)
    print(sample_trigonal)
    print(sample_tetragonal)

    _sim_impl(rank, size, "cubic", conf_name_cubic, sample_cubic)
    _sim_impl(rank, size, "trigonal", conf_name_trigonal, sample_trigonal)
    _sim_impl(rank, size, "tetragonal", conf_name_tetragonal, sample_tetragonal)


if __name__ == '__main__':
    main()

