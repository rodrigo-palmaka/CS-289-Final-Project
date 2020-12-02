#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


def draw_vector(vector, ax, scale=1):
    vector = vector
    origin = np.array([0,0,0])
    vec = np.vstack((origin, vector))*scale
    ax.plot(vec[:,0],vec[:,1],vec[:,2])


# In[3]:


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
#         print(plane_vector)
        plane_vector = plane_vector/np.linalg.norm(plane_vector)
        
        unit_plane += plane_vector
#     print(unit_plane)
    unit_plane = unit_plane / n
    unit_plane = unit_plane / np.linalg.norm(unit_plane)
    return unit_plane


# In[4]:


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


# In[2]:


def convert2d(data='', file=''):
    """
    In: Raw csv data/file path from planet. 
    Uses rotation matrices and normal vector to rotate 3D epicycloid 
    s.t. it is parallel to XY plane.
    Out: Pandas df [Time | 'X', 'Y', 'Z']
    """
    if file == '':
        
        cartesian = data[['X','Y','Z']].to_numpy()
    
        normal = generate_plane_vector(cartesian,local=True)
        phi = np.arccos(normal[2])
        theta = np.arctan(normal[1]/normal[0])
#         print(normal)
        if normal[0] < 0:
            theta = -theta
        flat = Rotate_z(cartesian, -theta)
        flat = Rotate_y(flat, -phi)
    elif data == '':
#         try:
        data = pd.read_csv(file)
#         except:
#             print("Call with file or data")
        cartesian = data[['X','Y','Z']].to_numpy()
        normal = generate_plane_vector(cartesian,local=True)
        phi = np.arccos(normal[2])
        theta = np.arctan(normal[1]/normal[0])
        if normal[0] < 0:
            theta = -theta
        flat = Rotate_z(cartesian, -theta)
        flat = Rotate_y(flat, -phi)
    idx = data['Time']
    flat_df = pd.DataFrame(flat,index=idx, columns=['X', 'Y', 'Z'])
    return flat_df, theta, phi, normal
# ff = convert2d(file="Mercury2x.csv")
# ff
# fig1= plt.figure(figsize=(20,20))
# ax1 = fig1.add_subplot(111)
# plt.plot(ff['X'], ff['Y'])
# ax1.set_aspect('equal', adjustable="box")


# In[1]:


# ur = pd.read_csv('Uranus4x.csv')
# uu,_,_ = convert2d(ur[9135:9143])
# print('---------------------------------')
# # uu1,_,_ = convert2d(ur[9134:9142])
# # uu[1]
# plt.plot(uu['X'], uu['Y'])
# ur.isna().sum()
# ur[:3]
# ur[9135:9143]


# # Example: Mercury 3D to 2D

# In[62]:



# data = pd.read_csv("Mercury2x.csv")
# # 
# mercury_cartesian=data[['X','Y','Z']].to_numpy()
# fig = plt.figure(figsize=(7,7))
# ax = fig.gca(projection='3d')
# for i in range(10):
#     normal = generate_plane_vector(mercury_cartesian,local=True)
#     phi = np.arccos(normal[2])
#     theta = np.arctan(normal[1]/normal[0])
#     normal_prime = Rotate_z(normal, -theta)
#     normal_prime = Rotate_y(normal_prime, -phi)
#     print(normal_prime)
#     draw_vector(normal_prime, ax)

# ax.set_xlim(-2, 2)
# ax.set_ylim(-2, 2)
# ax.set_zlim(0, 4)

# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# plt.show()


# In[61]:


# fig = plt.figure(figsize=(7,7))
# ax = fig.gca(projection='3d')
# for i in range(10):
#     normal = generate_plane_vector(mercury_cartesian,local=True)
#     draw_vector(normal,ax,scale=100000000)
# ax.plot(mercury_cartesian[:,0],mercury_cartesian[:,1],mercury_cartesian[:,2],'b')
# ax.view_init(45,0)
# plt.show()


# In[36]:


# normal = generate_plane_vector(mercury_cartesian,local=True)
# phi = np.arccos(normal[2])
# theta = np.arctan(normal[1]/normal[0])


# In[63]:


# normal_prime = Rotate_z(normal, -theta)
# normal_prime = Rotate_y(normal_prime, -phi)

# mercury_flat = Rotate_z(mercury_cartesian, -theta)
# mercury_flat = Rotate_y(mercury_flat, -phi)

# plt.plot(mercury_flat[:,0], mercury_flat[:,1])
# fig = plt.figure(figsize=(7,7))
# ax = fig.gca(projection='3d')
# draw_vector(normal_prime, ax, scale=20000000)

# ax.plot(mercury_cartesian[:,0],mercury_cartesian[:,1],mercury_cartesian[:,2],'r')
# ax.plot(mercury_flat[:,0],mercury_flat[:,1],mercury_flat[:,2],'b')

# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Z axis')

# ax.view_init(0,0)
# plt.show()


# In[64]:


# plt.plot(mercury_flat[:,2])
# plt.xlabel("Time")
# plt.ylabel("Z")
# plt.show()


# In[65]:


# fig1= plt.figure(figsize=(20,20))
# ax1 = fig1.add_subplot(111)
# plt.plot(mercury_cartesian[:,0], mercury_cartesian[:,1])
# ax1.set_aspect('equal', adjustable="box")

# ax2 = fig2.add_subplot(121)
# # ax2 = fig2.gca(projection='3d')
# plt.plot(mercury_flat[:,0], mercury_flat[:,1])
# ax2.set_aspect('equal', adjustable="box")


# In[ ]:





# In[ ]:




