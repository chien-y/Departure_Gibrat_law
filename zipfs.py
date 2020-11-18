import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from itertools import groupby
from price import Production_Chain
from graph_builder import *
from draw import *

def show_zipf_plot(pc, scale_factor=2000, sub_only=False):

    vas, level = get_value_added(pc, scale_factor=scale_factor, sub_only=sub_only)

    rank_plot = np.arange(len(vas)) + 1
    # the ranks are [1, 2, 2, 4, 4, 4, ,4 , 8, ...] if κ=2
    # [1, 2, 2, 2, 5, 5, 5, 5, ,5, 5, 5, 5, 5, 14, 14, ...] if  κ=3
    if sub_only:
        rank = [pc.κ**i for i in range(level) for k in range(pc.κ**i)]
#         r = 0
#         rank=[]
#         for i in range(level):
#             if i < 2:
#                 r += 1
#             else:
#                 r += pc.κ**(i-1) 
#             rank.extend(r * np.ones(pc.κ**i))
                
        

    if not sub_only:
        vas.sort(reverse=True)
        rank = []
        for key, group in groupby(vas):
            r = len(rank) + 1
            rank.extend(r * np.ones(len(list(group))))
               

    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw log normal approximation
    z = np.log(vas[:-1])
    mu, sd = z.mean(), z.std()
    Z = mu + sd * np.random.randn(len(vas))
    ln_obs = np.exp(Z)
#     print(np.shape(ln_obs))
    lb_obs = np.sort(ln_obs)[::-1]
    ax.loglog(np.arange(len(ln_obs)) + 1, ln_obs, 'rp', alpha=0.3,
              label='lognormal approximation')


    # plot lo rank
    ax.set_xlabel('log rank', fontsize=14)
    ax.loglog(rank, vas, 'bo', label="observations")

    ax.legend(frameon='False')

def zipf_with_regression(pc, scale_factor=2000, sub_only=False, DoNotPlot=False):

    vas, level = get_value_added(pc, scale_factor=scale_factor, sub_only=sub_only)

    rank_plot = np.arange(len(vas)) + 1
    # the ranks are [1, 2, 2, 4, 4, 4, ,4 , 8, ...] if κ=2
    if sub_only:
        rank = [pc.κ**i for i in range(level) for k in range(pc.κ**i)]
#         r = 0
#         rank=[]
#         for i in range(level):
#             if i < 2:
#                 r += 1
#             else:
#                 r += pc.κ**(i-1) 
#             rank.extend(r * np.ones(pc.κ**i))
#         print(rank)
#         num_firms=0
#         for i in range(level):
#             num_firms += pc.κ**i
#         rank = np.arange(1, num_firms+1,step=1)

    if not sub_only:
        vas.sort(reverse=True)
        rank = []
        for key, group in groupby(vas):
            r = len(rank) + 1
            rank.extend(r * np.ones(len(list(group))))
                
    xdata = np.log(rank)
    ydata = np.log(np.array(vas))
    
    if not DoNotPlot:
        
        fig, ax = plt.subplots(figsize=(10, 6.0))

    #     n = len(vas)


        ax.plot(xdata, ydata, 'o', markersize=15, alpha=0.6)
        ax.set_xlabel("log rank")
        ax.set_ylabel("log value added")

        slope, intercept, rvalue, pvalue, stderr = linregress(xdata, ydata)
        line = intercept + slope * xdata

        label = "Slope= %1.2f" %(slope)
        ax.plot(xdata, line, 'k-', label=label)
        ax.legend(loc='upper right')

    else:
        slope, intercept, rvalue, pvalue, stderr = linregress(xdata, ydata)
        return slope

if __name__=="__main__":
    pc = Production_Chain(τ=0.01, α=0.01, κ=5, q=0.9,
                          c=lambda x: 0.015*(np.exp(25*x)-1),
                          proportional_assembly=True)

    zipf_with_regression(pc, scale_factor=2000, sub_only=True)
    draw_graph(pc, scale_factor=1e4, sub_only=False)
    show_zipf_plot(pc, scale_factor=2000, sub_only=True)
