import dgl
import networkx as nx
import community
import matplotlib.pyplot as plt
import numpy as np
G1=nx.Graph()
for line in open('snd_Adj_Layer3.txt'):
    strlist = line.split()
    n1 = int(strlist[0])
    n2 = int(strlist[1])
    G1.add_edges_from([(n1, n2)])
for i in range (71):
    G1.add_node(i)
adj=nx.adjacency_matrix(G1).todense()
np.savetxt(r'layer3.txt',adj, fmt='%d', delimiter=' ')