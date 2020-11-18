"""
Compute the equilibrium prices for the price function

p(ell) = δ min{ min{n p(ell/n ) + α(ell, n)}, c(ell)}
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound



class Chain_n:
    """
    price function
    p(ell) = δ min{ min{n p(ell/n ) + α(ell, n)}, c(ell)}
    """
    def __init__(self,
                 n_grid=5000,
                 max_n=50,
                 τ=0.01,
                 α=lambda ell, n: 0.01*n * (ell)**1,
                 c=lambda ell: np.exp(ell)-1,
                 ):

        self.n_grid = n_grid
        self.max_n = max_n
        self._τ = τ
        self._δ = 1 / (1 - τ)
        self._α = α
        self._c = c

        self.update()

    def update(self):
        # increase the precision for ell < 0.01
        a = np.linspace(0, 0.01, num=(self.n_grid-300), endpoint=False)
        b = np.linspace(0.01, 1, num=300)
        self.grid = np.concatenate((a, b))
        self.p = self.compute_prices()
        self.p_func = lambda x: np.interp(x, self.grid, self.p)

    def get_c(self):
        return self._c

    def set_c(self, c):
        self._c = c
        self.update()


    def get_τ(self):
        return self._τ

    def set_τ(self, τ):
        self._τ = τ
        self._δ = 1 / (1 - τ)
        self.update()


    def get_α(self):
        return self._α

    def set_α(self, α):
        self._α = α
        self.update()

    c = property(get_c, set_c)
    τ = property(get_τ, set_τ)
    α = property(get_α, set_α)

    def solve_min(self, p, ell):
        """
        Solve for the minumum and minimizers when at stage ell, given p.

        p is a function.
        """
        δ, α, c = self._δ, self._α, self._c

        n = [i for i in range(2, self.max_n)]
        n = np.array(n, dtype="float")
        sub = np.amin(n * p(ell/n) + α(ell, n))
        Tp_ell = δ * np.minimum(sub, c(ell))

        return Tp_ell

    def apply_T(self, current_p):

        δ, α, c, = self._δ, self.α, self._c
        p = lambda x: np.interp(x, self.grid, current_p)
        new_p = np.empty(len(self.grid))

        for i, ell in enumerate(self.grid):
            current_function_min = self.solve_min(p, ell)
            new_p[i] = current_function_min

        return new_p

    def compute_prices(self, tol=1e-10, verbose=False):
        """
        Iterate with T.
        The initial condition is p=c.
        """

        δ, α, c = self._δ, self._α, self._c
        # Initial condition
        current_p = c(self.grid)

        error = tol + 1
        while error > tol:
            new_p = self.apply_T(current_p)
            error = np.max(np.abs(current_p - new_p))
            if verbose == True:
                print(error)
            current_p = new_p
        return new_p


    def plot_prices(self, plottype='-', label="$p(\ell)$", cost_label="$\delta c(\ell)$",
                    plot_cost=True, ylim=False, log=False):

        plt.rcParams["figure.figsize"] = (8, 6)
        if log:
            if plot_cost:
                plt.plot(self.grid, np.log(self._δ * self._c(self.grid)), plottype, label=cost_label, alpha=0.6)
            plt.plot(self.grid, np.log(self.p), plottype, label=label, alpha=0.6)
            plt.xlabel("$\ell$",fontsize=14)
            plt.ylabel("log $p(\ell)$",fontsize=14)
        else:
            if plot_cost:
                plt.plot(self.grid, self._δ * self._c(self.grid), plottype, label=cost_label, alpha=0.6)
            plt.plot(self.grid, self.p, plottype, label=label, alpha=0.6)
            plt.xlabel("$\ell$",fontsize=14)
            plt.ylabel("$price$",fontsize=14)
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        plt.legend()
        plt.show()

    def level(self,):
        "retrun number of levels m and number of subcontractors"
        δ, α, c = self._δ, self._α, self._c
        p = self.p_func

        n = [i for i in range(2, self.max_n)]
        n = np.array(n, dtype="float")
        k = []
        ell = 1
        for m in range(1, 50):
            if m==1:
                pass
            else:
                ell = ell / k[-1]
#             print(ell)
            sub = np.amin(n * p(ell/n) + α(ell, n))
            if sub < c(ell):
                k.append(np.argmin(n * p(ell/n) + α(ell, n)) + 2)
                
            else:
                return m, k
            
            
   
            

# Example

if __name__ == '__main__':

    pc = Chain_n(α=lambda ell, n: 0.01 * n**1.1)
    pc.plot_prices()
    print(pc.level())
