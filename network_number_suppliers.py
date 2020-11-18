import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from price_choice_over_n import Chain_n
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.stats import linregress
from itertools import groupby


class Firm:
    """object of firms"""
    def __init__(self, va):
        self.value_add = va
        self.subcontractors = []

def build_dict(pc, verbose=False, tol=1e-10, sub=False,):
    """
    pc: a class of production chain
    """

    ell = 1
    level = 1
    num_firms_at_this_level = 1
    current_firm_num = 1
    first_firm_at_level = 1

    length, num_suppliers = pc.level()
    if verbose:
        print(f"length={length}")
        print(f"number of suppliers ={num_suppliers}")
    firms = {}

    while 1:
        if level < length:
            n = num_suppliers[level - 1] # num of upstream partners at this level
            va = pc.α(ell, n)
        else:
            va = pc.c(ell)

        if verbose:
            print(f"current_firm_num={current_firm_num}")
            print(f"level={level}")
            print(f"first_firm_at_level={first_firm_at_level}")
            print(f"num_firms_at_this_level={num_firms_at_this_level}")

        for i in range(num_firms_at_this_level):
            firms[first_firm_at_level + i] = Firm(va)

        if sub and (level == length-1): break
            
        if level == length: break

        # Add subcontractors for each firms at this level
        for i in range(num_firms_at_this_level):
            for k in range(n):
                current_firm_num += 1
                firms[first_firm_at_level + i].subcontractors.append(current_firm_num)

        first_firm_at_level = first_firm_at_level + num_firms_at_this_level
        level += 1
        num_firms_at_this_level *= n
        ell = ell / n

    return firms



def build_dict_and_graph(pc, verbose=False, sub=False):

    firms = build_dict(pc, verbose=verbose, sub=sub)
    G = nx.Graph()

    for firm_no, firm in firms.items():
        for sub in firm.subcontractors:
            G.add_edge(firm_no, sub)
    return firms, G


def get_value_add(pc, scale_factor=2000, sub=False):
    firms, G = build_dict_and_graph(pc, sub=sub)
    vas = []
    for firm in firms.values():
        vas.append(firm.value_add * scale_factor)

    return vas

def draw_graph(pc, scale_factor=2000, figsize=(15, 15)):
    firms, G = build_dict_and_graph(pc)
    position = graphviz_layout(G, prog='twopi')

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off', frame_on=False)

    vas = []
    for firm in firms.values():
        vas.append(firm.value_add * scale_factor)
    nx.draw(G,
            node_size=vas,
            node_shape='o',
            alpha=0.5,
            node_color='blue',
            with_labels=False)

    # hide axis
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)




def zipf_with_regression(pc, scale_factor=2000, plot='link', DoNotPlot=False):

    
    length, num_suppliers = pc.level()
    
    
    
    if plot == 'size':
        vas = get_value_add(pc, scale_factor=scale_factor)
        rank_plot = np.arange(len(vas)) + 1
        # the ranks are [1, 2, 2, 4, 4, 4, ,4 , 8, ...] if κ=2
        vas.sort(reverse=True)
        rank = []
        for key, group in groupby(vas):
            r = len(rank) + 1
            for i in range(len(list(group))):
                rank.append(r)

        ydata = np.log(np.array(vas))

        
    elif plot == 'size_subcontractor':
        vas = get_value_add(pc, scale_factor=scale_factor, sub=True)
        rank_plot = np.arange(len(vas)) + 1
        # the ranks are [1, 2, 2, 4, 4, 4, ,4 , 8, ...] if κ=2
        vas.sort(reverse=True)
        rank = []
        for key, group in groupby(vas):
            r = len(rank) + 1
            for i in range(len(list(group))):
                rank.append(r)

        ydata = np.log(np.array(vas))
        
    
    elif plot == 'link':
        
        num_firms_at_this_level = 1
        links = []
        for n in num_suppliers:
            links.extend(np.ones(num_firms_at_this_level) * n + 1)
#             links.extend(np.ones(num_firms_at_this_level) * n )
            num_firms_at_this_level *= n
        
        for i in range(num_firms_at_this_level):
                links.append(1)
        
        links.sort(reverse=True)
#         print(links)
        rank = []
        for key, group in groupby(links):
            r = len(rank) + 1
            rank.extend(np.ones(len(list(group))) * r) 
            
                
#         print(rank)
        ydata = np.log(np.array(links))

    elif plot == 'suppliers':
        
        num_firms_at_this_level = 1
        links = []
        for n in num_suppliers:
            links.extend(np.ones(num_firms_at_this_level) * n )
            num_firms_at_this_level *= n
        
        links.sort(reverse=True)
        rank = []
        for key, group in groupby(links):
            r = len(rank) + 1
            rank.extend(np.ones(len(list(group))) * r) 
            
                
        ydata = np.log(np.array(links))

        
    else:
        raise TypeError("Plotting type is wrong.")
    
    xdata = np.log(rank)
    if not DoNotPlot:
        fig, ax = plt.subplots(figsize=(10, 6.0))
    
        
    
        ax.plot(xdata, ydata, 'o', markersize=15, alpha=0.6)
        ax.set_xlabel("log rank")
    
        if plot == 'size': ax.set_ylabel("log value added")
        elif plot == 'size_subcontractor': ax.set_ylabel("log value added")
        elif plot == 'link': ax.set_ylabel("log(numer of buyer-customer links)")
        else: ax.set_ylabel("log(number of suppliers)")

        slope, intercept, rvalue, pvalue, stderr = linregress(xdata, ydata)
        line = intercept + slope * xdata

        label = "Slope= %1.2f" %(slope)
        ax.plot(xdata, line, 'k-', label=label)
        ax.legend(loc='upper right', fontsize=14)

    else:
        slope, intercept, rvalue, pvalue, stderr = linregress(xdata, ydata)
        return slope
    
if __name__ == "__main_":
    pc = Chain_n(τ=0.01, α=lambda ell, n: 0.01 * n**1.1,
                 c=lambda ell: np.exp(ell)-1)
    firms, G = build_dict_and_graph(pc, verbose=True)



