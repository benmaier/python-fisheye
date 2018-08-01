import numpy as np
import matplotlib.pyplot as pl

from fisheye import fisheye

xws = [0.0,0.2,0.4,0.6]

N = 15
R = 0.4
mode = 'default'
dx = 1.0 / N
dir1 = np.linspace(0,1,500)

for imode, mode in enumerate(['default', 'root']):

    if mode == 'root':
        ds = [1.5,2,2.5,3,]
    else:
        ds = [1.5,3,4.5,6,]

    fig, ax = pl.subplots(len(xws),len(ds),figsize=(5,5))

    for i_d, d in enumerate(ds):
        for ixw, xw in enumerate(xws):

            F = fisheye(R,mode=mode,xw=xw,d=d)
            F.set_focus([0.5,0.5])

            for i in range(1,N):
                dir2 = np.ones_like(dir1) * dx * i
                # x - direction
                new_x_1, new_y_1 = [], []
                new_x_2, new_y_2 = [], []
                pos_1 = np.empty((len(dir1),2))
                pos_2 = np.empty((len(dir1),2))
                pos_1[:,0] = dir1
                pos_1[:,1] = dir2
                pos_2[:,0] = dir2
                pos_2[:,1] = dir1

                pos_1 = F.radial_2D(pos_1)
                pos_2 = F.radial_2D(pos_2)

                ax[i_d,ixw].plot(pos_1[:,0], pos_1[:,1],'-k',lw=0.5)
                ax[i_d,ixw].plot(pos_2[:,0], pos_2[:,1],'-k',lw=0.5)

                ax[i_d,ixw].axis('square')
                ax[i_d,ixw].axis('off')
                if i_d == 0:
                    ax[i_d,ixw].text(0.5,1.05,'$x_w = {:3.1f}$'.format(xw).format(xw),
                                     transform = ax[i_d,ixw].transAxes,
                                     ha='center'
                            )
                if ixw == 0:
                    ax[i_d,ixw].text(-0.2,0.5,'$d = {:4.1f}$'.format(d),
                                     transform = ax[i_d,ixw].transAxes,
                                     rotation = 'vertical',va='center'
                            )


    fig.tight_layout()
    pl.subplots_adjust(wspace=0.01,hspace=0.01)
    fig.savefig(mode+'_parameter_scan.pdf')
    fig.savefig(mode+'_parameter_scan.png',dpi=300)

pl.show()
