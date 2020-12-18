# Li-Air Battery Discharge parameters with varying volume fraction

# Input Values
C_o = 0.3 #mol/L
ep = 0.1
A_surf = 2.57e-5 #m^2
A = 4.90e-12 #m^2
H_an = -16 #kJ/mol
U_o = 1.2 #V
T = 400 #K
T_o = 298 #K
S_rxn = 59 #J/mol*K
n = 4 #moles
F = 96485 #C/eq
R = 8.314 #J/mol*K
gamma_k = 1
v_k = 1
RTinv = 1/R/T
i_o= 1.0 #mA/cm^2
beta =0.5
eta = 2.3 #V

F = 96485 #C/eq
R = 8.3145 #J/mol*K

# Initialize
import numpy as np

U_an_0 = 0 #V
U_ca_0 = 0 #V

DC_0 = np.array([U_an_0, U_ca_0])

# Model
from scipy.integrate import solve_ivp
import numpy as np

time_span = np.array([0,160]) #s

solution = solve_ivp(dX_k_dt, y: residual(t, U), DC_0,)

# Function
import numpy as np
from math import exp

def residual(t, DC, pars):

    RTinv = 1/R/pars.T
    dDC_dt = np.zeros_like(DC)


    U = U_o+(T-T_o)*(S_rxn/(n*F))-(R*T/(n*F))*ln(a_k^v_k)
    a_k = gamma_k*(C_k/C_o)
    i_far =i_o*(exp(-n*F*beta*eta*RTinv) -exp(n*F*(1-beta)*eta*RTinv))

    d(X_k)/d(t) = (1/C_o)*(1/ep)*(A_surf/A)*(1/H_an)*s_k
    s_k = v_k/(n*F)*i_far
    C_k = X_k*C_o
    q_rxn = k_rxn*C_k



    ddC_dt[1] = i_dl_ca*pars.C_dl_ca_inv

    return dDC_dt

# Plotting Graphs
    from matplotlib import pyplot as plt
    for var in solution.y:
        plt.plot(solution.t,var)

    plt.show()
