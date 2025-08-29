from __future__ import print_function, division
import numpy as np

# CALL tspn.py
def update(func, y_init, t_len, clamp_template, params, tstep, ccOrVc=True, g_syn_template=None):
    # ccOrVc = False for voltage clamp; True for current clamp
    num_vars = len(y_init)
    y = np.zeros((t_len, num_vars))
    y[0] = y_init

    # Exponential Euler
    for i in range(0, t_len):
        if i <= t_len - 2:
            if g_syn_template is not None:
                y[i+1] = func(y[i], clamp_template[i], params, tstep, ccOrVc, g_syn_template[i])
            else:
                y[i + 1] = func(y[i], clamp_template[i], params, tstep, ccOrVc)
    print('Trace computed...')
    return y

