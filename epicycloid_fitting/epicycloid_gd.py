import numpy as np
from scipy.optimize import leastsq
import progressbar
import epicycloid_process_data as epd

def epi_x(t, params):
    """
    params is an array with the five different epicycloid parameters
    """
    (a,b,c,d,R) = params
    return (a+b)*np.cos(c*(t-d)) - R*np.cos((a+b)/b*(c*(t-d)))

def epi_y(t, params):
    """
    params is an array with the five different epicycloid parameters
    """
    (a,b,c,d,R) = params
    return (a+b)*np.sin(c*(t-d)) - R*np.sin((a+b)/b*(c*(t-d)))

def squared_err(params, flat_data_scl, timestamps_scl):
	x_pred = epi_x(timestamps_scl, params)
	y_pred = epi_y(timestamps_scl, params)
	x_true = flat_data_scl[:,0]
	y_true = flat_data_scl[:,1]

	x_squared_err = (x_pred-x_true)**2
	y_squared_err = (y_pred-y_true)**2

	return np.sum(x_squared_err + y_squared_err)

def fit(flat_data_scl, timestamps, period=2, ab_ratio=11):
	"""
	Requires prior knowledge about how many periods are represented by data
	Note: scale the params
	"""
	x_true = flat_data_scl[:,0]
	y_true = flat_data_scl[:,1]

	t0 = min(timestamps)
	timestamps0 = timestamps - t0
	delta_t = max(timestamps0)


	guess_amp = 3*np.std(x_true)/(2**0.5)/(2**0.5)
	guess_a = guess_amp
	guess_b = guess_a/ab_ratio
	guess_c = period/delta_t*2*np.pi
	guess_d = 0
	guess_R = guess_a/2

	params0 = [guess_a, guess_b, guess_c, guess_d, guess_R]

	optimize_func_x = lambda x: epi_x(timestamps0, x) - x_true
	optimize_func_y = lambda y: epi_y(timestamps0, y) - y_true
	params_x = leastsq(optimize_func_x, params0)[0]
	params_y = leastsq(optimize_func_y, params0)[0]

	params = (params_x+params_y)/2
	params[3] = params[3] + t0 #readjust d

	return params

def predict(timestamps, params, normal):
	x_pred = epi_x(timestamps, params)
	y_pred = epi_y(timestamps, params)
	z_pred = np.zeros(len(timestamps))

	pred = np.array([x_pred, y_pred, z_pred]).T
	pred = epd.reverse_z_alignment(pred,normal)
	return pred

def calc_mse(data_true, data_pred):
	n = len(data_true)
	error = data_true-data_pred

	total_squared_error = 0
	for i in range(n):
		total_squared_error += np.linalg.norm(error[i,:])**2
	mse = total_squared_error/n
	return mse
