#!/usr/bin/env python
# coding: utf-8

# In[7]:


#TEAM VNS HELPERS https://github.com/ArnaudFickinger/astronomical_dataset_generator

#add noise to data, data must be in [x,y,z] format or you know what you are doing
#possible type of noise are gaussian/uniform random noise
def addnoise(data,mu,sigma,typ="gaussian"):
    print("adding ",typ," noise to data......")
    print("mu=",mu," and sigma=",sigma)
    ldata=np.copy(data)
    shape=np.shape(ldata)
    ldata=ldata.flatten()
    size=len(ldata)
    if typ=="gaussian":
        ldata+=np.random.normal(mu,sigma,size=size)
    elif typ=="uniform":        
        ldata+=sigma*(np.random.rand(size)-0.5+mu)
    else:
        print("invalid type of noise!")
        print("available types of noise are gaussian or uniform.")
        exit(1)
    return ldata.reshape(shape)
#randomly delete percentage of data assigned by percent
#you may want to do this to sparsify the data due random conditions like weathers, people in charge of observing ask for a day off or so.
def sparsify(data,percent):
    n=len(data)
    ndrawn=int(n*(1-percent))
    print("sparsify ",n,"data into ",n-ndrawn," data...")
    dellist=random.sample(range(n),ndrawn)
    return np.delete(data,dellist,axis=0)
def find_cycle(data):
  time = []
  max_val = -1* np.inf
  for i in range(len(data)):
    if data['Right Ascension'][i] > max_val:
      max_val = data['Right Ascension'][i]
    elif (max_val - data['Right Ascension'][i]) > 300:
      max_val = data['Right Ascension'][i]
      time.append(data['Time'][i])
  return time
#Fourier feature generating function
def fourier_featurize(X, d = 5, freq = 1):
  data = []
  for i in range(d):
    if i == 0:
      data.append(np.cos(i * freq * X).reshape(1,-1)[0])
    else:
      data.append(np.sin(i * freq * X).reshape(1,-1)[0])
      data.append(np.cos(i * freq * X).reshape(1,-1)[0])
  return np.array(data).T
# estimate_frequency(X.reshape(len(X)), Y_new)


# In[8]:


#helper functions to reset time so that it cycle and function to control the scafcle of the data
def reset_time(Time,cycle):
    New_time = Time - Time.min(axis = 0)
    mod_time = New_time % cycle
    return mod_time
    
def scale(X,max_val):
    n = len(X)
    X_std = (np.array([((X[i] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))) for i in range(0,n)]))*max_val 
    return X_std


# In[2]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body, get_moon
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
from time import mktime

#Helpers
#Cartesian Coordinates relative to earth, conversion implemented from http://faraday.uwyo.edu/~admyers/ASTR5160/handouts/51605.pdf
def convert_cartesian(ra, decl, dist):
  x = dist * np.cos([ra]) * np.cos([decl])
  y = dist * np.sin([ra]) * np.cos([decl])
  z = dist * np.sin([decl])
  return x[0], y[0], z[0]
#Inverse of Cartesian coordinate function, returns cartesian coordinates to ephemeri, RA/DECL/DIST, returns radians or degrees
def convert_ephemeri(x, y , z , radian = True):
  dist = np.sqrt(x**2 + y**2 + z**2)
  ra = np.arctan2(y,x)
  decl = np.arcsin(z/dist)
  #TODO: Finish this
  if radian:
    return ra, decl, dist
  else:
    return math.degrees(ra), math.degrees(decl), dist
#Progress bar helper function
# from IPython.display import HTML, display

def progress(value, max=100):
    return HTML("""
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(value=value, max=max))
#Data Generator
def generate_ephemeri(body, date_range, earth_loc):
#   out = display(progress(0, periods), display_id=True)
  ras = []
  decs = []
  diss = []
  xs = []
  ys = []
  zs = []
  #Generating dates
  loc = EarthLocation.of_site(earth_loc)

  #Generating Ephemeri's
  i = 0
  for t in date_range:
    i += 1
    t = Time(str(t))
    with solar_system_ephemeris.set('jpl'): #Need jplephem library for this
      b = get_body(body, t, loc)
    ra = b.ra.degree
    dec = b.dec.degree
    dis = b.distance.value
    x, y, z = convert_cartesian(b.ra.radian, b.dec.radian, dis)

    ras.append(ra)
    decs.append(dec)
    diss.append(dis)
    xs.append(x)
    ys.append(y)
    zs.append(z)
    #update prog bar    
#     out.update(progress(i, periods))
  data = pd.DataFrame({"Time": date_range, "Right Ascension": ras, "Declination": decs, 'Distance': diss, "X": xs, "Y": ys, "Z": zs})
  return data
def find_cycle(data):
  time = []
  max_val = -1* np.inf
  for i in range(len(data)):
    if data['Right Ascension'][i] > max_val:
      max_val = data['Right Ascension'][i]
    elif (max_val - data['Right Ascension'][i]) > 300:
      max_val = data['Right Ascension'][i]
      time.append(data['Time'][i])
  return time


# In[2]:


def scale(X,upper = 1, lower = 0):
    ## TODO: vectorize, taking too long
    n = len(X)
    X_std = (np.array([((X[i] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))) for i in range(0,n)]))*(upper-lower)
    X_std = X_std + lower
    return X_std

def scale1d(X,upper = 1, lower = 0):
    ## TODO: vectorize, taking too long
    n = len(X)
    X_std = (np.array([((X[i] - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))) for i in range(0,n)]))*(upper-lower)
    X_std = X_std + lower
    return X_std

def epi_x(t, a, b, R):
  return (a + b)*np.cos(t) - R*np.cos(t*(a+b/b))

def epi_y(t, a, b, R):
  return (a + b)*np.sin(t) - R*np.sin(t*(a+b/b))

def plot_epic(t, a, b, R, cycles):
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    
    # Prepare arrays x, y, z
    theta = np.linspace(0, cycles*2* np.pi, len(t))

    x_small = scale(epi_x(theta, a, b, R), upper=1, lower=-1)
    y_small = scale(epi_y(theta, a, b, R), upper=1, lower=-1)

    ax.scatter(0,0, c='orange', label='Geometric Center')
    ax.scatter(0, np.max(y_small)/3, c='green', label='Earth')
    ax.plot(x_small, y_small, label='Epicycloid function')
    ax.view_init(azim=-90, elev=90)
    ax.set_aspect('equal', adjustable="box")
    ax.legend()

    plt.show()

def grad_loss(t, x, y, a, b, R, func='MSE'):
  """ computes gradient of loss function w.r.t. all training samples (batch GD)
  for the current iterations of the three weights a, b, R
  MSE: 1/2*(epi(t_i, w) - y_i)**2
  """
  n = t.shape[0]
  # grad_x = np.zeros((n, 3))
  # grad_y = np.zeros((n, 3))
  grad = np.zeros(3)
  loss = 0

  if func == 'MSE':
    epic_x = epi_x(t, R, a, b) - x 
    epic_y = epi_y(t, R, a, b) - y 
    
    dydR = (epic_y)*(-np.sin((a+b)*(t/b)))
    dydb = (epic_y)*((a*R*t*np.cos(t*(a+b)/b)/(b**2)) + np.sin(t))
    dyda= (epic_y)*(np.sin(t) - R*t*np.cos(t*(a+b)/b)/b)

    dxdR = (epic_x)*(-np.cos((a+b)*(t/b)))
    dxdb = (epic_x)*(-(a*R*t*np.sin(t*(a+b)/b)/(b**2)) + np.cos(t))
    dxda = (epic_x)*(np.cos(t) + R*t*np.sin(t*(a+b)/b)/b)


    dlda = (1/n) * np.sum((1/2)*np.sqrt((epic_x)**2 + (epic_y)**2)*(2*(dxda + dyda)))
    dldb = (1/n) * np.sum((1/2)*np.sqrt((epic_x)**2 + (epic_y)**2)*(2*(dxdb + dydb)))
    dldR = (1/n) * np.sum((1/2)*np.sqrt((epic_x)**2 + (epic_y)**2)*(2*(dxdR + dydR)))
    

    # grad_y = np.array((dyda, dydb, dydR))

    

    grad = np.array((dlda, dldb, dldR))

    loss = (1/n)* np.sum(np.sqrt((epic_x)**2 + (epic_y)**2))
    # print(loss)
    # loss_x = 1/2*np.mean((epi_x(t, R, a, b) - x)**2)
    # loss_y = 1/2*np.mean((epi_y(t, R, a, b) - y)**2)
  return grad, loss

def grad_descent(t, x, y, a, b, R, lr = 0.001, iters = 1000, epsilon=0.001, num_cycles=2):

#   t_norm = scale(t, upper = 2*np.pi*num_cycles,lower = 0)
#   x_norm = np.array(x.apply(lambda v: (v - min(x))/(max(x) - min(x))))
#   y_norm = np.array(y.apply(lambda e: (e - min(y))/(max(y) - min(y))))
  # print(x_norm)
  # y_norm = 
  # x_cost = np.zeros(iters)
  # y_cost = np.zeros(iters)
  cost = np.zeros(iters)


  for iter in range(iters):

    grad, loss = grad_loss(t, x, y, a, b, R)
    # x_cost[iter] = l_x
    # y_cost[iter] = l_y

    if iter >= 1000000:
      if ((loss - cost[iter -1]) > 0) or (np.abs(loss - cost[iter -1]) < epsilon):
        cost[iter] = loss
        print(cost[:100])

        fig,ax = plt.subplots(figsize=(12,8))
      
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Iterations')
        _=ax.plot(range(iters),cost,'b.')
        return np.array((a, b, R)) 
    cost[iter] = loss


    a = a - lr*grad[0]
#     b = b - lr*grad[1]
#     R = R - lr*grad[2]

  print(cost[:100])

  fig,ax = plt.subplots(figsize=(12,8))
 
  ax.set_ylabel('MSE Loss')
  ax.set_xlabel('Iterations')
  _=ax.plot(range(iters),cost,'b.')
  return np.array((a, b, R)) 



# grad_loss(8, 16, 2, np.arange(0, 10500), merc['X'], merc['Y'])
# x = merc['X']
# c = x.apply(lambda v: (v - min(x))/(max(x) - min(x)))
# c.hist()

# np.arange(0, 10500).shape[0]


# In[8]:


def grad_loss2(t, x, y, theta, offset=15,func='MSE'):
  """ computes gradient of loss function w.r.t. all training samples (batch GD)
  for the current iterations of the rotation parameter theta
  MSE: 1/N ...
  """
  n = t.shape[0]

  grad = 0
  loss = 0
  mse = 0

  if func == 'MSE':
#     offset = 15
    x1 = x[offset:n//2]
    y1 = y[offset:n//2]
    x2 = x[n//2:-offset]
    y2 = y[n//2:-offset]
#     print(x2)
   
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])

    carts = np.vstack((x1, y1))
    carts2 =  np.vstack((x2, y2))
#     print(R.shape, carts.shape)
    rot = R @ carts
#     print(x1.shape, x2.shape, y1.shape, y2.shape)
    dldt = (1/n)*np.sum(x1**2 + y1**2 + x2**2 + y2**2 + 2*(x1*x2 + y1*y2)*np.sin(theta) 
                        + 2*(x1*y2 - y1*x2)*np.cos(theta))
    grad = dldt
#     norm_loss = (1/n)* np.sum(np.sqrt((rot[0,:] - x_test)**2 + (rot[1, :] - y_test)**2))
    two_norm = (1/n) * np.sum(np.linalg.norm((rot - carts2), axis=0)**2)
#     print(two_norm.shape)
#     mse =  (1/n)* np.sum((rot[0,:] - x_test)**2 + (rot[1, :] - y_test)**2)
    
    loss = two_norm
    
  return grad, loss

def grad_descent2(t, x, y, theta, lr = 0.001, iters = 1000, offset=15, epsilon=0.0001, num_cycles=2):

  losses = np.zeros(iters)
  thetas = np.zeros(iters)
  for iter in range(iters):

    grad, loss = grad_loss2(t, x, y, theta, offset)

    if iter >= 1000000:
      if ((loss - cost[iter -1]) > 1000000*epsilon) or (np.abs(loss - cost[iter -1]) < epsilon):
        cost[iter] = loss
        print(cost[:100])
        fig,ax = plt.subplots(figsize=(12,8))      
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Iterations')
        _=ax.plot(range(iters),cost,'b.')
        return theta
    losses[iter] = loss
    thetas[iter]= theta

    theta = theta - lr*grad
#   print(losses[:100])

#   fig,ax = plt.subplots(figsize=(8,8))
#   ax.set_ylabel('MSE Loss')
#   ax.set_xlabel('Iterations')
#   ax.scatter(np.argmin(losses), np.min(losses), c='r', s=100)
#   ax.title('Gradient Descent')
#   _=ax.plot(range(iters),losses,'b.')
  return thetas, losses


# In[1]:


def grad_descent2plot(t, x, y, theta, lr = 0.001, iters = 1000, offset=15, epsilon=0.0001, num_cycles=2):

  losses = np.zeros(iters)
  thetas = np.zeros(iters)
  for iter in range(iters):

    grad, loss = grad_loss2(t, x, y, theta, offset)

    if iter >= 1000000:
      if ((loss - cost[iter -1]) > 1000000*epsilon) or (np.abs(loss - cost[iter -1]) < epsilon):
        cost[iter] = loss
        print(cost[:100])
        fig,ax = plt.subplots(figsize=(12,8))      
        ax.set_ylabel('MSE Loss')
        ax.set_xlabel('Iterations')
        _=ax.plot(range(iters),cost,'b.')
        return theta
    losses[iter] = loss
    thetas[iter]= theta

    theta = theta - lr*grad
  print(losses[:100])

  fig,ax = plt.subplots(figsize=(8,8))
  ax.set_ylabel('MSE Loss')
  ax.set_xlabel('Iterations')
  ax.scatter(np.argmin(losses), np.min(losses), c='r', s=100)
  ax.title('Gradient Descent')
  _=ax.plot(range(iters),losses,'b.')
  return thetas, losses


# In[4]:


def scale_test(x, train, upper=1, lower=-1):
    """ 
    scales test point to [-1, 1] based on train distirbution
    """
    return (upper-lower)*(x - np.min(train))/(np.max(train) - np.min(train)) + lower
    
def unscale_test(x, train, upper=1, lower=-1):
    """ 
    scales test point to [-1, 1] based on train distirbution
    """
    return (upper-lower)*(x + 1)/(2) + lower
   


# In[ ]:




