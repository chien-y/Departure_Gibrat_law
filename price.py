"""
Compute the equilibrium prices for the price function

p(ell) = δ min{κ p(ell/κ ) + α, c(ell)}
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fminbound


class Production_Chain:

    def __init__(self,
                 n=5000,
                 τ=0.01,
                 α=0.01,
                 κ=2,
                 c=lambda x: np.exp(x) - 1,
                 q=1,
                 proportional_assembly=False):

        self.n = n
        self._τ = τ
        self._δ = 1 / (1 - τ)
        self._c = c
        self._κ = κ

        if proportional_assembly == True:
            self._α = lambda x: α * x**q
        else:
            self._α = α

        self.proportional_assembly = proportional_assembly
        self.update()

    def update(self):
        # increase the precision for ell < 0.01
        a = np.linspace(0, 0.01, num=(self.n-300), endpoint=False)
        b = np.linspace(0.01, 1, num=300)
        self.grid = np.concatenate((a, b))
        self.p = self.compute_prices()
        self.p_func = lambda x: np.interp(x, self.grid, self.p)

    def get_c(self):
        return self._c

    def set_c(self, c):
        self._c = c
        self.update()

    def get_κ(self):
        return self._κ

    def set_κ(self, κ):
        self._κ = κ
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
    κ = property(get_κ, set_κ)
    τ = property(get_τ, set_τ)
    α = property(get_α, set_α)

    def solve_min(self, p, ell):
        """
        Solve for the minumum and minimizers when at stage ell, given p.

        p is a function.
        """
        δ, α, c, κ, n = self._δ, self._α, self._c, self.κ, self.n


        if self.proportional_assembly == True:
            Tp_ell = δ * np.minimum(κ * p(ell / κ) + α(ell), c(ell))
        else:
            Tp_ell = δ * np.minimum(κ * p(ell / κ) + α, c(ell))


        if  δ * c(ell) == Tp_ell:
            home_production = 1
            return Tp_ell, home_production
        else:
            home_production = 0
            return Tp_ell, home_production

    def apply_T(self, current_p):

        δ, α, c, κ, n = self._δ, self.α, self._c, self._κ, self.n
        p = lambda x: np.interp(x, self.grid, current_p)
        new_p = np.empty(n)

        for i, ell in enumerate(self.grid):
            current_function_min, home_production = self.solve_min(p, ell)
            new_p[i] = current_function_min

        return new_p

    def compute_prices(self, tol=1e-10, verbose=False):
        """
        Iterate with T.
        The initial condition is p=c.
        """

        δ, α, c, κ, n = self._δ, self._α, self._c, self._κ, self.n
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

    def plot_prices(self, plottype='-', label="$p(\ell)$", cost_label="$\delta c(\ell)$", plot_cost=True):
        
        if plot_cost:
            plt.plot(self.grid, self._δ * self._c(self.grid), plottype, label=cost_label, alpha=0.6)
            
        plt.plot(self.grid, self.p_func(self.grid), plottype, label=label, alpha=0.6)
        
        plt.xlabel("$\ell$",fontsize=12)
        plt.ylabel("$price$",fontsize=12)
        
    def level(self,):
        "retrun number of levels"
        δ, α, c, κ, n = self._δ, self._α, self._c, self._κ, self.n
        p = self.p_func
        for m in range(50):
            if self.proportional_assembly:
                if κ * p(κ**(-m-1)) + α(κ**(-m)) >=  c(κ**(-m)):
                    return m + 1 # level index starts from 1
    
            else:
                if κ * p(κ**(-m-1)) + α >=  c(κ**(-m)):
                    return m + 1 # level index starts from 1
    
    def tilde_ell(self):
        δ, κ, α, c = self._δ, self._α, self._c, self._κ
        p = self.p_func
        n = np.array([i for i in range(2, 50)])
        
        if self.proportional_assembly:
            f = lambda ell: κ * p(ell/κ) + α(ell) - c(ell)
        else:
            f = lambda ell: κ * p(ell/κ) + α - c(ell)
        root = brentq(f, 0, 1)
        
        return root


# Example

if __name__ == '__main__':

    production = Production_Chain()
    production.plot_prices(label="$p(\ell)$", cost_label="$e^{\ell}-1$")

    print(production.level())
# production.c = lambda x: x + x**2
# production.plot_prices(label="$p(\ell): \ell+\ell^2$", cost_label="$\ell+\ell^2$")

# production.α = 0.2
# production.plot_prices(label="κ=10")
    plt.legend()
    plt.show()
