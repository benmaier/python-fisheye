import numpy as np
from scipy.interpolate import interp2d, RectBivariateSpline
from scipy.ndimage.filters import gaussian_filter

def apply_to_image(img_data, F, use_cartesian = False, interpolation_order = 1, gaussian_blur=False):
    """Apply a fisheye effect to image data.

    Parameters
    ----------
    img_data : numpy.ndarray
        An array containing the image data. It has to be of shape (m, n) for greyscale,
        or of shape (m, n, k) for k color channels.
    F : fisheye.fisheye
        Instance of a fisheye class with defined radius R and focus set (both in pixel
        coordinates).
    use_cartesian: bool, default = False
        Usually, a radial fisheye function is demanded, this switch enforces the cartesian
        transformation instead.
    interpolation_order : int, default = 1
        Order of interpolation for the RectBivariateSpline algorithm
    gaussian_blur : bool, default = False
        Apply a gaussian blur fliter to the transformed area.

    Returns
    -------
    new_img_data : numpy.ndarray
        The transformed data in the shape of img_data.
    """

    orig_shape = img_data.shape
    k = interpolation_order

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

    if not use_cartesian:
        coords = []

        for i in range(max(0, int(fx-R)), min(data.shape[0],int(fx+R))):
            for j in range(max(0, int(fy-R)), min(data.shape[1],int(fy+R))):
                if np.sqrt((fx-i)**2 + (fy-j)**2) < R:
                    coords.append((i,j))
    else:
        coords = [ (i,j) for i in range(max(0, int(fx-R)), min(data.shape[0],int(fx+R)))\
                         for j in range(max(0, int(fy-R)), min(data.shape[1],int(fy+R))) ]

    coord_arr = np.array(coords)

    if not use_cartesian:
        inv_coords = F.inverse_radial_2D(np.array(coord_arr,dtype=float))
    else:
        inv_coords = F.inverse_cartesian(np.array(coord_arr,dtype=float))

    new_data = data.copy()

    for color in range(data.shape[2]):
        transform_function = RectBivariateSpline(x, y, data[:,:,color],kx=k,ky=k)
        transformed = transform_function(inv_coords[:,0].flatten(), inv_coords[:,1].flatten(),grid=False)
        new_data[:,:,color] = data[:,:,color]

        new_data[coord_arr[:,0], coord_arr[:,1], color] = transformed

        if gaussian_blur:
            new_data[coord_arr[:,0], coord_arr[:,1], color] = gaussian_filter(
                                                new_data[coord_arr[:,0], coord_arr[:,1], color],
                                                sigma = 2
                                                )

    return new_data.reshape(orig_shape)

if __name__ == "__main__":
    import matplotlib.pyplot as pl

    data = pl.imread('../sandbox/example.png')

    from fisheye import fisheye

    F = fisheye(300,xw=0.3,mode='default',d=2)
    F.set_focus([400,400])

    transformed = apply_to_image(data, F)

    pl.imshow(data)

    pl.figure()

    pl.imshow(transformed)

    pl.figure()
    pl.imshow(apply_to_image(data, F, use_cartesian = True))
    pl.show()

