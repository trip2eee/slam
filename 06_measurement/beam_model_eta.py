import sympy as sp
import numpy as np
import scipy

scipy.special.erf()


pi = sp.pi
s_hit = sp.symbols('s_hit')
x, z = sp.symbols('x z')

N = 1/sp.sqrt(2*pi*s_hit**2) * sp.exp(-1/2*(x-z)**2 / s_hit**2)

print(N)
dN = sp.integrate(N, (x, 0, z))
print(dN)

dN = sp.integrate(N, x)
print(dN)

