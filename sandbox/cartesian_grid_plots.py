import numpy as np
import matplotlib.pyplot as pl
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from fisheye import fisheye

xws = [0.0,0.2,0.4,0.6]

N = 10
R = 0.4
mode = 'default'
dx = 1.0 / N
dir1 = np.linspace(0,1,500)


import networkx as nx
from networkx.generators.lattice import hexagonal_lattice_graph

G = hexagonal_lattice_graph(N,N+2,with_positions = True)
G = nx.convert_node_labels_to_integers(G)

pos = [ n[1]['pos'] for n in G.nodes(data = True) ]
pos = np.array(pos)
pos /= np.amax(pos)

N_nodes = G.number_of_nodes()
print(pos.shape)

radii = np.ones(pos.shape[:1]) * 0.01


for imode, mode in enumerate(['default', 'root']):

    if mode == 'root':
        ds = [1.5,2,2.5,3,]
    else:
        ds = [1.5,3,4.5,6,]

    fig, ax = pl.subplots(len(xws),len(ds),figsize=(5,5))

    for i_d, d in enumerate(ds):
        for ixw, xw in enumerate(xws):

            F = fisheye(R,mode=mode,xw=xw,d=d)
            F.set_focus([0.487,0.512])

            _pos, _radii = F.scale_radial_2D(pos, radii)

            lines = [ [ _pos[e,0], _pos[e,1] ] for e in G.edges() ]
            lines = [ list(zip(x, y)) for x, y in lines ]

            lc = LineCollection(lines,
                                color = 'k',
                                lw = 0.5,
                                )
            ax[i_d,ixw].add_collection(lc)


            patches = []
            for n in range(N_nodes):
                x1 = _pos[n,0]
                y1 = _pos[n,1]
                r = _radii[n]
                circle = Circle((x1, y1), r, color='k')
                patches.append(circle)

            pc = PatchCollection(patches, match_original=True)
            ax[i_d,ixw].add_collection(pc)


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
    fig.savefig('hexa_'+mode+'_parameter_scan.pdf')
    fig.savefig('hexa_'+mode+'_parameter_scan.png',dpi=300)

pl.show()
