"""
draw the network of chain
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from graph_builder import *
from networkx.drawing.nx_agraph import graphviz_layout

def get_value_added(pc, scale_factor=2000, sub_only=False):

    firms, G, level = build_dict_and_graph(pc, sub_only=sub_only)
    vas = []
    for firm in firms.values():
        vas.append(firm.value_add * scale_factor)

    return vas, level

def draw_graph(pc, scale_factor=2000, figsizes=(15, 15), sub_only=False):

    firms, G, level = build_dict_and_graph(pc, sub_only=sub_only)
    # position:  A dictionary with nodes as keys and positions as values
    position = graphviz_layout(G, prog='twopi') # twopi, dot, sfdp, circo

    fig, ax = plt.subplots(figsize=figsizes)
    ax.axis('off', frame_on=False)

    vas=[]
    for firm in firms.values():
        vas.append(firm.value_add * scale_factor)
# print(vas)
# print(len(vas), len(firms), sep='-')
    nx.draw(G,
            position,
            node_size=vas,
            node_shape='o',
            alpha=0.4,
            node_color='blue',
            with_labels=False)

    # hide axis text
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

# example
# τ=0.1, α=6, q=1, κ=4, c= e^(15x)-1 => size for in-house < for all outsourcing firms
# τ=0.1, α=0.7, q=0.90, κ=5, c=e^(3.4x)-1
# τ=0.01, α=0.06, q=0.90, κ=7, c=0.2x + 0.8x^2
# τ=0.01, α=0.01, q=0.8, κ=5, c=0.0015(e^(25x)-1)

if __name__ == "__main__":

    pc = Production_Chain(τ=0.01, α=0.001, q=0.9, κ=2,
                          c=lambda x: 0.5*(np.exp(x) - 1),
                          proportional_assembly=True)
# draw_graph(pc, sub_only=False, scale_factor=1e4)
    pc.plot_prices()
    draw_graph(pc, scale_factor=1e5, sub_only=True)
