import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

def datetime_to_float(d):
    # total_seconds will be in decimals (millisecond precision)
    return d.timestamp()

def float_to_datetime(fl):
    return dt.datetime.fromtimestamp(fl)

def datetime_to_timestamp_array(times):
    timestamps = np.array([time.timestamp() for time in times])
    return timestamps

def perform_scale(X, upper = 1, lower = -1):
    ## TODO: vectorize, taking too long
    n = len(X)
    X_std = (np.array([((X[i] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))) for i in range(0,n)]))*(upper-lower)
    X_std = X_std + lower
    return X_std

def draw_vector(vector, ax, scale=1):
	"""
	helper function to draw a 3d vector
	"""
	vector = vector
	origin = np.array([0,0,0])
	vec = np.vstack((origin, vector))*scale
	ax.plot(vec[:,0],vec[:,1],vec[:,2])

def generate_plane_vector(data, n=1000,local=True):
	"""
	Takes cartesian coordinates
	produces a normal vector in cartesian
	"""
	unit_plane = np.array([0.0,0.0,0.0])

	for i in range(n):
		index1, index2, index3 = 0,0,0
		if local:
			## Get three points that are spatially close together
			## This method seems better and more consistent
			index1 = np.random.randint(0, len(data)-4)
			index2 = index1 + 1
			index3 = index2 + 1
		else:
			## Get three random points
			index1 = np.random.randint(0, len(data))
			index2 = np.random.randint(0, len(data))
			index3 = np.random.randint(0, len(data))

		p1 = data[index1,0:3]
		p2 = data[index2,0:3]
		p3 = data[index3,0:3]

		vec1 = p2-p1
		vec2 = p3-p1
		plane_vector = np.cross(vec1, vec2)
		plane_vector = plane_vector/np.linalg.norm(plane_vector)

		unit_plane += plane_vector

	unit_plane = unit_plane/n
	unit_plane = unit_plane/np.linalg.norm(unit_plane)
	return unit_plane

def Rotate_z(data, theta):
	"""
	Rotate data theta clockwise around z axis
	"""
	Rz = np.array([[np.cos(theta),-np.sin(theta), 0],
		[np.sin(theta), np.cos(theta), 0],
		[0, 0, 1]])
	new_data = Rz @ data.T
	return new_data.T

def Rotate_y(data, theta):
	"""
	Rotate data theta clockwise around y axis
	"""
	Ry = np.array([[np.cos(theta),0, np.sin(theta)],
		[0,1,0],
		[-np.sin(theta), 0, np.cos(theta)]])
	new_data = Ry @ data.T
	return new_data.T

def perform_z_alignment(data, normal):
	phi = np.arccos(normal[2])
	theta = np.arctan(normal[1]/normal[0])
	if normal[0] < 0:
		theta = -theta
	new_data = Rotate_z(data, -theta)
	new_data = Rotate_y(new_data, -phi)
	return new_data

def reverse_z_alignment(data, normal):
	phi = np.arccos(normal[2])
	theta = np.arctan(normal[1]/normal[0])
	if normal[0] < 0:
		theta = -theta
	new_data= Rotate_y(data, phi)
	new_data = Rotate_z(new_data, theta)
	return new_data

def pre_process_data(filename):
    """
    -Processes data into numpy arrays
    -Rotates the orbit to align with the z-axis
    -converts all datetimes into floating point timestamps

    if scale is true:
        x,y,z, are scaled to range from [-1,1]
        timestamps are scaled to range [0,1]

    Returns
    astro_true: The scaled position data
    astro_normal: The normal vector of the orbit
    astro_flat: The scaled rotated orbit
    timestamps: The scaled array of timestamps
    """

    astro_data = pd.read_csv(filename, parse_dates=['Time'], date_parser = pd.to_datetime)
    datetimes = astro_data['Time']
    timestamps = datetime_to_timestamp_array(datetimes)
    astro_true = astro_data[['X','Y','Z']].to_numpy()

    astro_normal = generate_plane_vector(astro_true)
    astro_flat = perform_z_alignment(astro_true, astro_normal)

    return astro_true, astro_normal, astro_flat, timestamps
