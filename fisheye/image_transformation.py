import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline

def transform_data(img_data, F):

    orig_shape = img_data.shape

    if len(img_data.shape) == 2:
        data = img_data.reshape((img_data.shape[0],img_data.shape[1],1))
    elif len(img_data.shape) != 3:
        raise ValueError("Wrong shape of data:", img_data.shape)
    else:
        data = img_data.copy()

    fx, fy = F.focus
    R = F.R

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])

    coords = []

    for i in range(max(0, int(fx-R)), min(data.shape[0],int(fx+R))):
        for j in range(max(0, int(fy-R)), min(data.shape[1],int(fy+R))):
            if np.sqrt((fx-i)**2 + (fy-j)**2) < R:
                coords.append((i,j))

    coord_arr = np.array(coords)
    inv_coords = F.inverse_radial_2D(np.array(coord_arr,dtype=float))

    new_data = data.copy()

    for color in range(data.shape[2]):
        transform_function = RectBivariateSpline(x, y, data[:,:,color],kx=1,ky=1)
        transformed = transform_function(inv_coords[:,0].flatten(), inv_coords[:,1].flatten(),grid=False)
        new_data[:,:,color] = data[:,:,color]

        new_data[coord_arr[:,0], coord_arr[:,1], color] = transformed

    return new_data.reshape(orig_shape)

if __name__ == "__main__":
    import matplotlib.pyplot as pl

    data = pl.imread('../sandbox/example.tiff')

    from fisheye import fisheye

    F = fisheye(300,xw=0.3,mode='Sarkar',d=2)
    F.set_focus([400,400])

    transformed = transform_data(data, F)

    pl.imshow(data)

    pl.figure()

    pl.imshow(transformed)
    pl.show()

