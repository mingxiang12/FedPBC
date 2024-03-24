from config import time_varying, algorithm
from varying_main import varying_dynamics
from varying_pbc import varying_dynamics_pbc
from static_main import static_dynamics
from static_pbc import static_dynamics_pbc


if __name__ == '__main__':
    if time_varying:
        if algorithm != 'fedpbc':
            varying_dynamics()
        else:
            varying_dynamics_pbc()
    else:
        if algorithm != 'fedpbc':
            static_dynamics()
        else:
            static_dynamics_pbc()
                
