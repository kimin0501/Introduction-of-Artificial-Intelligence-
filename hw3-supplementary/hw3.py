from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

# load file and center this dataset around the origin
def load_and_center_dataset(filename):
    x = np.load(filename)
    center = x - np.mean(x, axis=0)
    
    return center

# compute the covariance matrix from the dataset
def get_covariance(dataset):
    numImage = 2414
    covariance = (1 / (numImage - 1)) * np.dot(dataset.T , dataset)
    
    return covariance

# get largest eigenvalues and their eigenvectors by doing eigendecomposition
def get_eig(S, m):   
    numRows = S.shape[0]
    eigval, eigvec = eigh(S, subset_by_index=[numRows - m, numRows -1])
    
    eigval[[0, 1]] = eigval[[1, 0]]
    eigvec[:, [0, 1]] = eigvec[:, [1, 0]]
    
    return np.diag(eigval),eigvec

# get all eigenvalues and corresponding eigenvectors by doing eigendecomposition
def get_eig_prop(S, prop):

    numRows = S.shape[0]
    
    eigvals_array = eigh(S, eigvals_only = True)
    sum = 0
    
    for i in range(0, numRows):
        sum += eigvals_array[i]
    
    eigval, eigvec = eigh(S, subset_by_value=[prop * sum, np.inf])
    
    eigval[[0, 1]] = eigval[[1, 0]]
    eigvec[:,[0, 1]] = eigvec[:,[1, 0]]
    
    return np.diag(eigval), eigvec


# project each image into given dimensions
def project_image(image, U):
    alpha = 0
    numCols = U.shape[1]
    
    for i in range(numCols):
        alpha += np.dot(np.dot((U[:,i]).T ,image),U[:,i])

    return alpha

# display both original and projection images side by side
def display_image(orig, proj):
    orig = orig.reshape(32,32).T
    proj = proj.reshape(32,32).T
    
    fig, (ax1, ax2) = plt.subplots(figsize = (9, 3), ncols = 2)

    ax1.set(title = 'Original')
    ax2.set(title = 'Projection')

    col_1 = ax1.imshow(orig, aspect= 'equal')
    col_2 = ax2.imshow(proj, aspect= 'equal')
    
    fig.colorbar(col_1, ax = ax1)
    fig.colorbar(col_2, ax = ax2)

    return fig, ax1, ax2