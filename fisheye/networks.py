import numpy as np
import matplotlib.pyplot as pl
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def plot_network(edges, node_positions, node_radii, F, ax = None, node_colors = None, scale_radii=True):

    N_nodes = len(node_radii)
    pos = node_positions
    radii = node_radii
    
    if ax is None:
        fig, ax = pl.subplots(1,1)

    if node_colors is None: 
        node_colors = np.zeros((len(node_radii),3))

    if scale_radii:
        _pos, _radii = F.scale_radial_2D(pos, radii)
    else:
        _pos = F.radial_2D(pos)
        _radii = radii

    lines = [ [ _pos[e,0], _pos[e,1] ] for e in edges ]
    lines = [ list(zip(x, y)) for x, y in lines ]

    lc = LineCollection(lines,
                        color = 'k',
                        lw = 0.5,
                        )

    ax.add_collection(lc)

    patches = []
    for n in range(N_nodes):
        x1 = _pos[n,0]
        y1 = _pos[n,1]
        r = _radii[n]
        circle = Circle((x1, y1), r, color=node_colors[n])
        patches.append(circle)

    pc = PatchCollection(patches, match_original=True)
    ax.add_collection(pc)


    ax.axis('square')
    ax.axis('off')

    return ax

if __name__ == "__main__":


    import networkx as nx
    from networkx.generators.lattice import hexagonal_lattice_graph

    N = 5

    G = hexagonal_lattice_graph(N,N+2,with_positions = True)
    G = nx.convert_node_labels_to_integers(G)

    pos = [ n[1]['pos'] for n in G.nodes(data = True) ]
    pos = np.array(pos)
    pos /= np.amax(pos)

    N_nodes = G.number_of_nodes()

    radii = np.ones(pos.shape[:1]) * 0.02


    from fisheye import fisheye

    F = fisheye(R=0.4,focus=[0.5,0.48])

    plot_network(list(G.edges()), pos, radii, F)

    pl.show()


