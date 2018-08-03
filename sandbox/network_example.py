import numpy as np
import networkx as nx
from networkx.generators.lattice import hexagonal_lattice_graph

from fisheye import fisheye
from fisheye.networks import plot_network

import matplotlib.pyplot as pl


N = 5

G = hexagonal_lattice_graph(N,N+2,with_positions = True)
G = nx.convert_node_labels_to_integers(G)

pos = [ n[1]['pos'] for n in G.nodes(data = True) ]
pos = np.array(pos)
pos /= np.amax(pos)

N_nodes = G.number_of_nodes()

radii = np.ones(pos.shape[:1]) * 0.02

F = fisheye(R=0.4,focus=[0.5,0.48],d=5)

fig, ax = pl.subplots(1,1,figsize=(3,3))

plot_network(list(G.edges()), pos, radii, F, ax=ax)

fig.savefig('hexa_graph.png',dpi=300)
fig.tight_layout()

pl.show()


