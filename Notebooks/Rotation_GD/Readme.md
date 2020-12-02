## Rotation_GD.ipynb:

This is the main notebook to run. It contains the RotationRegressor class that performs fit(), predict() similar to an sklearn method. 
To choose the planet to predict and the timestamp, simply change: 

planet = 'Jupiter'
test_pnt = '2021-11-26 18:00:00'

This notebook also contains contextual information that builds on the description of this problem given by the writeup.

## Rotation_Results.ipynb:

Running this notebook will produce RMSE values for for all 7 planets over a test range of ~5 years in the future

Dependencies, Helper files, etc.

### convert2d.py

converts 3D cartesian data to 2D by rotating the epicycloid such that it is parallel to the xy plane.

### Helpers_fixed.py

contains helper functions and gradient descent method. The grad_descent2 method is the one used. The other GD method was for a different attempt at parameter estimation.

### epicycloid_process_data.py 

Used to convert points back to 3D


