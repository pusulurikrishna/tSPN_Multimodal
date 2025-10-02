from __future__ import print_function, division

#from matplotlib.mathtext import VCentered
from numpy import exp
from numpy import power
from numpy import log, log10
import numpy as np

def step(y, clampValue, gmax, dt, ccOrVc=True, Gsyn = 0):

    ## initialization
    V = y[0]             # Somatic membrane voltage (mV)
    CaS = y[1]           # somatic [Ca2+]
    m = y[2]             # Na activation
    h = y[3]             # Na inactivation
    n = y[4]             # K activation
    mA = y[5]            # A activation
    hA = y[6]            # A inactivation
    mh = y[7]            # h activation
    mM = y[8]            # M activation
    mCaL = y[9]          # CaL activation
    hCaL = y[10]         # CaL inactivation
    s = y[11]            # Na slow inactivation
    mKCa = y[12]         # KCa activation
    mh_inf_prev = y[21]

    hCaL2 = y[22]    #Ca slow inactivation
    I_nearElectrode = y[23] # I near electrode in the first compartment including capacitive artifact and the current coming from the second compartment;
    V_nearElectrode = y[24] # V near electrode in the first compartment;
    Nai = y[25] # [Na] inside
    Ki = y[26] # [K] inside

    gBetweenCompartments = gmax[0]
    GNa = gmax[1]           # nS; maximum conductance of INa
    GK = gmax[2]
    GCaL = gmax[3]
    GM = gmax[4]
    GKCa = gmax[5]
    GA = gmax[6]
    Gh = gmax[7]
    Gleak = gmax[8]
    C = gmax[9]
    #Gsyn = 0
    NaShift = gmax[10]
    gElectrode = gmax[11];
    CElectrode = gmax[12];
    T = gmax[13]; # Temperature
    Pump_P = gmax[14];
    Pump_Am = gmax[15];
    ELeakMultiplier = gmax[16];

    # T = 296 #K    #Dr. Hochman says this is close to the room temperature value of Celia's experiments
    defaultT = 296  # K
    #E_Na = 60                # mV; reverse potential of INa
    #E_Na = 70                # celia's data for 60mV VC also sees Na current
    #E_K = -90
    #E_K = -80                # TODO: listen to meeting recording with Celia and Dr. Prinz Dec 03 2020
    E_h = -31.6 * (T/defaultT)

    E_syn = 0
    E_syn = 10
    #E_Ca = 120
    #"""
    R = 8314 # Jkg-1mole-1K-1
    zCa = 2
    CaO = 2.0 # mM

    F = 96500 #C mole-1
    E_Ca = (R*T/(zCa*F)) * log(CaO/CaS) #- 78.7

    Nao = 145; #mM
    E_Na = (R*T/F) * log(Nao/Nai);
        # double Nao = 145; // [Na_i]_init 10mM, [Na_o] 145mM gives E_Na ~ 68.2 mV
        #                   // [Na_i]_init 9.3171063545mM,  [Na_o] 145mM gives E_Na ~ 70 mV
    Ko = 4.1; #mM
    E_K = (R*T/F) * log(Ko/Ki);
        # double Ko = 4.1; // [K_i]_init 122.9 mM, [K_o] 4.1mM gives E_K ~ -86.7mV
        #                  // [K_i]_init 139.787094212 mM, [K_o] 4.1mM gives E_K ~ -90mV

    #E_leak = -55 * (T / defaultT)
    ## E_leak as a combination of E_Na and E_K
    ## At any voltage, gLeak(V - E_leak) = gNaLeak(V-ENa) + gKLeak(V-EK); At V = E_leak, this is 0, implies E_leak = (gNaL*ENa+gKL*Ek)/(gNaL+gKL); Since we know default ENa, EK, ELeak, we get ratio of gNaLeak/gKLeak; Also, at equilibrium same eqn. gives gNaLeak(Eleak-ENa) = gKLeak(EK-Eleak), which gives the ratio gNaLeak/gKLeak = (-90 + 55)/(-55-70) = 35/125 = 0.28; Assuming these conductances remains the same even as ion concentrations change;
    #E_leak = (gNaL*ENa+gKL*Ek)/(gNaL+gKL) = (0.28*E_Na + E_K)/1.28
    E_leak = (0.28 * E_Na + E_K) / 1.28; ## This equals -55 if E_Na = 70mV and E_K = -90mV
    E_leak = (1. * E_Na + E_K) / 1.28;  ## Adjusting this to match RMP of the cell.
    E_leak = (0.7 * E_Na + E_K) / 1.28;  ## Adjusting this to match RMP of the cell.
    E_leak = (ELeakMultiplier * E_Na + E_K) / 1.28;
                # 'T' part of E_Na and E_K, so not needed here again.
                ## Now these currents should also be part of ion changes for Na/K
                ## At V = 0, gLeak*Eleak = gNaLeak*ENa + gKLeak * EK ==> gLeak = gNaLeak + gKLeak; This will fit the eqn. at any voltages; gLeak = 1.28 gKLeak or gKLeak = 0.78125 * gLeak; gNaLeak = 0.21875 * gLeak;

    Q10 = 4 #Q10 for adjusting activation and inactivation rates, typical range for ion channels is 2.4 - 4, see https://link.springer.com/referenceworkentry/10.1007%2F978-1-4614-7320-6_236-1
    Q10tauFactor = np.power(Q10, (defaultT-T)/10) #factor to multiply activation and inactivation time constants to adjust for temperature dependence of gating dynamics

    Q10Max = 1. #1.2 #Q10 for maximal conductance
            # The rates of gating often increase with temperature coefficients Q10 of 2.4 to 4 as the temperature is raised ;
            # Unlike gating, the conductance of an open channel can be relatively temperature-insensitive, with a Q10 of only 1.2-1.5
    Q10MaxtauFactor = np.power(Q10Max, (defaultT - T) / 10)

    #"""
    # TODO Reversal potentials of Na, K, Ca need to be updated based on concentrations which change due to Na, K, CaL currents as well as pumps
        # TODO and temperature as well
        # TODO check ion conc. and reversal potentials with celia's data
    # TODO For Ca reversal potential see  8.1. A-and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristicsjn.1994.71.6.2338
    # TODO also include temperature dependency for reversal potentials and activation/inactivation time constants
        # TODO make a temperature-explicit version of the code by including temperature in GHK and Nernst potential equations and including Q10s for activation and inactivation time constants. That way we can have a global T parameter that we can change and all the dynamics will be automatically adjusted.
        # Q10s - https://www.jneurosci.org/content/jneuro/34/14/4963.full.pdf   
    # TODO Pump: Acurrents/8.1. A-and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristicsjn.1994.71.6.2338

    f = 0.01                   # percent of free to bound Ca2+
    alpha = 0.002              # uM/pA; convertion factor from current to concentration
    kCaS = 0.024               # /ms; Ca2+ removal rate, kCaS is proportional to  1/tau_removal; 0.008 - 0.025

    SCa = 1                    # uM; half-saturation of [Ca2+]; 25uM in Ermentrount book, 0.2uM in Kurian et al. 2011
    #SCa = 25

    tauKCa_0 = 50              # ms
    #tau_hA_scale = 100         # scaling factor for tau_hA
    tau_hA_scale = 10         # scaling factor for tau_hA --- seems to be the updated value; code pointed in 2019 tSPN paper incorrect w.r.t figures
    tau_hA_scale = 10 * 10
    tau_mA_scale = 5

    ## update dydt

    """
    # Sodium current (pA), Wheeler & Horn 2004 or Yamada et al., 1989
    # TODO Na V-ramp shouldn't have two peaks
    xFactor = 1
    alpha_m = xFactor*0.36 * (V + 33 - NaShift) / (1 - exp(-(V + 33 - NaShift) / 3)) if V != -33 else 0.36;
    beta_m = xFactor*- 0.4 * (V + 42 - NaShift) / (1 - exp((V + 42 - NaShift) / 20)) if V != -42 else 0.4;
    m_inf = alpha_m / (alpha_m + beta_m)
    tau_m = 2 / (alpha_m + beta_m)
    m_next = m_inf + (m - m_inf) * exp(-dt / tau_m) if dt < tau_m else m_inf

    hTauFactor = 1
    alpha_h = hTauFactor * - 0.1 * (V + 55) / (1 - exp((V + 55) / 6)) if V!=-55 else 0.1;
    beta_h = hTauFactor * 4.5 / (1 + exp(-V / 10))
    h_inf = alpha_h / (alpha_h + beta_h)
    tau_h = 2 / (alpha_h + beta_h)
    h_next = h_inf + (h - h_inf) * exp(-dt / tau_h) if dt < tau_h else h_inf

    alpha_s = 0.0077 / (1 + exp((V - 18) / 9))  # Miles et al., 2005
    beta_s = 0.0077 / (1 + exp((18 - V) / 9))
    tau_s = 129.2
    s_inf = alpha_s / (alpha_s + beta_s)
    s_next = s_inf + (s - s_inf) * exp(-dt / tau_s) if dt < tau_s else s_inf

    gNa = GNa * power(m_next, 2) * h_next
    I_Na = gNa * (V - E_Na)
    """

    """
    # Original - Sodium current (pA), Based on 8.1. A-and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristicsjn.1994.71.6.2338
    m_inf = 1/(1+exp((V+41.35)/-4.75))  #original
    tau_m = 0.75 * exp(-(0.0635)*(0.0635)*(V+40.35)*(V+40.35)) + 0.12
    m_next = m_inf + (m - m_inf) * exp(-dt / tau_m) if dt < tau_m else m_inf

    h_inf = 1/(1+exp((V+62.)/4.5))
    tau_h = 6.5*exp(-(0.0295)*(0.0295)*(V+75.00)*(V+75.00))+0.55
    h_next = h_inf + (h - h_inf) * exp(-dt / tau_h) if dt < tau_h else h_inf

    s_inf = 1/(1+exp((V+40.)/1.5))
    tau_s = 25./(1+exp((V-20.)/4.5)) + 0.01
    s_next = s_inf + (s - s_inf) * exp(-dt / tau_s) if dt < tau_s else s_inf

    gNa = GNa * power(m_next, 3) * h_next * s_next
    I_Na_nearNaChannels = 0.4*gNa * (V- E_Na)
    """

    #"""
    # Sodium current (pA), Based on 8.1. A-and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristicsjn.1994.71.6.2338
    m_inf = 1/(1+exp((V +41.35)/-4.75))  #original
    m_inf = 1/(1+exp((V +41.35 -NaShift)/-4.75))  # V offset (threshold) controls delay in onset of inward current as well as the shape of the delay curve vs VClamp (since it controls the amount of time needed for Vclamp to reach threshold) (in conjunction with the slope if needed) -- seems to be controlling the first and last Vclamp delays.. something else seems to control curve slope
            # Larger the +ve NaShift value, greater the threshold voltage, needing more gNa to reach it..

    tau_m = 0.75 * exp(-(0.0635)*(0.0635)*(V+40.35)*(V+40.35)) + 0.12
    tau_m = Q10tauFactor * 0.75 * exp(-(0.0635) * (0.0635) * (V + 40.35) * (V + 40.35)) + 0.12
    tau_m = Q10tauFactor * 0.75 * exp(-(0.0635) * (0.0635) * (V + 40.35 - NaShift) * (V + 40.35 - NaShift)) + 0.12

    #tau_m = Q10tauFactor * 0.75 * exp(-0.05*(0.0635) * (0.0635) * (V + 40.35 -7.4 -NaShift) * (V + 40.35 -7.4 -NaShift)) + 0.12
    #tau_m = Q10tauFactor*(0.4*0.75 * exp(-0.05*(0.0635)*(0.0635)*(V+40.35+20)*(V+40.35+20)) + 0.3) # first term seems to control amplitude of secondary inward pulses but not timing
            # tau_m modified vs original, only thing that's affected seems to be slope and values of max/sus FI curves .. all others in CC, VC, VR seem to not be affected much..
            # original tau_m makes slope and values of sus/max FI curves very less, whereas modification seems to make slope higher..  Adding NaShift to s_inf eqn. Also seems to reduce the slope of sus/maxFI curves.
            # original taum range for -50mv to 0mV is 0.8 to 0.1, while modified version is 0.55 to 0.4; compared modified, orignal model reduced sus fI curves values, slope largely and max curve slightly.. so reducing taum increasing susFI values n slope largely..
                    # taum modifications - three changes -- first term 0.4 multiple makes values smaller, second constant term increased from 0.12 to 0.3 raises baseline, and exp 0.05 multiple inside makes curve much slower..  only first term 0.4 multiple seems to keep overall behavior it fairly similar to original; using only exp multiple 0.05 seems to replicate the higher slopes of susFI curves.. so the main modification seems to be this multiple inside exp.. adding NaShift as well since that's also there for m_inf..
                    # For the modified model, we had to try many things to bring down fI curve slope (main thing affecting is capacitance).. but the original is doing this already..
    #tau_m = Q10tauFactor*0.4*0.75 * exp(-5*(0.0635)*(0.0635)*(V+40.35+20)*(V+40.35+20)) + 0.3 # first term
                    #Tau_m doesn't look very effective always in manipulating sus/maxFI curve slope; Only sometimes;

    m_next = m_inf + (m - m_inf) * exp(-dt / tau_m) if dt < tau_m else m_inf

    h_inf = 1/(1+exp((V+62.)/4.5))
    #h_inf = 1 / (1 + exp((V + 62. - NaShift) / 4.5))
        # NaShift for Na inactivation might help get rid of depolarization block at the beginning of CC steps without having to change input resistance
    tau_h = 6.5*exp(-(0.0295)*(0.0295)*(V+75.00)*(V+75.00))+0.55
    tau_h = Q10tauFactor * 6.5 * exp(-(0.0295) * (0.0295) * (V + 75.00) * (V + 75.00)) + 0.55
    tau_h = Q10tauFactor*(3.6*6.5*exp(-(0.0295)*(0.0295)*(V+75.00)*(V+75.00))+0.4)  # tau_h first term multiplicative factors needed to change inward peak curve shape to linear as in data, from somewhat upward curving near the bottom..
            # plot and compare tau_m vs tau_h -- is this biologically acceptable
    tau_h = Q10tauFactor * (5 * 6.5 * exp(-(0.0295) * (0.0295) * (V + 75.00) * (V + 75.00)) + 0.4)
    #tau_h = Q10tauFactor * (5 * 6.5 * exp(-(0.0295) * (0.0295) * (V + 75.00 - NaShift) * (V + 75.00 -NaShift)) + 0.4)

    h_next = h_inf + (h - h_inf) * exp(-dt / tau_h) if dt < tau_h else h_inf

    s_inf = 1/(1+exp((V+40.)/1.5))
            #  Adding NaShift to s_inf eqn. also seems to reduce the slope of sus/maxFI curves.
    tau_s = Q10tauFactor*(25./(1+exp((V-20.)/4.5)) + 0.01)
    s_next = s_inf + (s - s_inf) * exp(-dt / tau_s) if dt < tau_s else s_inf

    #including s seems to fit inward current amplitudes slightly better
    gNa = Q10MaxtauFactor * GNa * power(m_next, 3) * h_next * s_next
    #gNa = GNa * power(m_next, 3) * h_next #* s_next
    I_Na = gNa * (V - E_Na)

    #"""

    """
    # Sodium current Nav1.7 from https://senselab.med.yale.edu/modeldb/ShowModel?model=264591&file=/DRG_Codes/XPP/DRG.ode#tabs-2
    # Based on Using Bifurcation Theory for Exploring Pain - https://pubs.acs.org/doi/pdf/10.1021/acs.iecr.9b04495

    # Functions for Nav1.7

    am = 15.5 / (1 + exp(-(V - 5) / 12.08))
    bm = 35.2 / (1 + exp((V + 72.7) / 16.7))
    tau_m = 1 / (am + bm)
    m_inf = tau_m * am
    m_next = m_inf + (m - m_inf) * exp(-dt / tau_m) if dt < tau_m else m_inf

    ah = 0.38685 / (1 + exp((V + 122.35) / 15.29))
    bh = -0.00283 + 2.00283 / (1 + exp(-(V + 5.5266) / 12.70195))
    tau_h = 1 / (ah + bh)
    h_inf = tau_h * ah
    h_next = h_inf + (h - h_inf) * exp(-dt / tau_h) if dt < tau_h else h_inf

    a_s = 0.00003 + 0.00092 / (1 + exp((V + 93.9) / 16.6))
    bs = 132.05 - 132.05 / (1 + exp((V - 384.9) / 28.5))
    tau_s = 1 / (a_s + bs)
    s_inf = tau_s * a_s
    s_next = s_inf + (s - s_inf) * exp(-dt / tau_s) if dt < tau_s else s_inf


    gNa = GNa * power(m_next, 3) * h_next * s_next
    I_Na_nearNaChannels = gNa * (V - E_Na)
    """

    #"""
    # Potassium current (pA), Wheeler & Horn 2004 or Yamada et al., 1989
    alpha_n_20 = 0.0047 * (V - 8) / (1 - exp(-(V - 8) / 12)) if V!=8 else 0.0047;
    beta_n_20 = exp(-(V + 127) / 30)
    n_inf = alpha_n_20 / (alpha_n_20 + beta_n_20)
    alpha_n = 0.0047 * (V + 12) / (1 - exp(-(V + 12) / 12)) if V!=-12 else 0.0047;
    beta_n = exp(-(V + 147) / 30)
    tau_n = Q10tauFactor*1 / (alpha_n + beta_n) #Original
    #tau_n = 10 / (alpha_n + beta_n)  ## This seems to control the spike duration but also changing the shape of the spike
        # Since Na model is based on rat nodose ganglion - may be its better to use K model also from the same system
    n_next = n_inf + (n - n_inf) * exp(-dt / tau_n) if dt < tau_n else n_inf
    
    gK = Q10MaxtauFactor * GK * power(n_next, 4)
    I_K = gK * (V - E_K)
    #"""

    """
    #Potassium current, Based on 8.1. A-and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristicsjn.1994.71.6.2338
        # The voltage ramp results using this model show the K-current peak is much wider than in experiments
    n_inf = 1/(1.+exp((V+14.62)/-18.38))
    alpha_n = 0.001265*(V+14.273)/(1.-exp((V+14.273)/-10.0))
    beta_n = 0.125 * exp(-(V + 55) / 2.5)
    tau_n = 1 + 1 / (alpha_n + beta_n)
    tau_n = Q10tauFactor*(1 + 1 / (alpha_n + beta_n))

    n_next = n_inf + (n - n_inf) * exp(-dt / tau_n) if dt < tau_n else n_inf
    gK = GK * n_next
    I_K = Q10MaxtauFactor * gK * (V - E_K)
    """

    """ # Calcium current (pA), L-type, Bhalla & Bower, 1993
    alpha_mCaL = 7.5 / (1 + exp((13 - V) / 7))
    beta_mCaL = 1.65 / (1 + exp((V - 14) / 4))
    mCaL_inf = alpha_mCaL / (alpha_mCaL + beta_mCaL)
    tau_mCaL = 1 / (alpha_mCaL + beta_mCaL)
    mCaL_next = mCaL_inf + (mCaL - mCaL_inf) * exp(-dt / tau_mCaL) if dt < tau_mCaL else mCaL_inf

    alpha_hCaL = 0.0068 / (1 + exp((V + 30) / 12))
    beta_hCaL = 0.06 / (1 + exp(-V / 11))
    hCaL_inf = alpha_hCaL / (alpha_hCaL + beta_hCaL)
    tau_hCaL = 1 / (alpha_hCaL + beta_hCaL)
    hCaL_next = hCaL_inf + (hCaL - hCaL_inf) * exp(-dt / tau_hCaL) if dt < tau_hCaL else hCaL_inf
    
    gCaL = GCaL * mCaL_next * hCaL_next
    I_CaL = gCaL * (V - E_Ca)

    hCaL2_next = hCaL2
    """

    #""" # Calcium current - long lasting - Based on 8.1. A-and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristicsjn.1994.71.6.2338
    mCaL_inf = 1/(1+exp((V+20.)/-4.5))
    tau_mCaL = Q10tauFactor*(3.25 * exp(-(0.042)*(0.042)*(V+31.)*(V+31.)) + 0.395)
    mCaL_next = mCaL_inf + (mCaL - mCaL_inf) * exp(-dt / tau_mCaL) if dt < tau_mCaL else mCaL_inf

    hCaL_inf = 1/(1+exp((V+20.)/25.))
    tau_hCaL = Q10tauFactor*(33.5*exp(-(0.0395)*(0.0395)*(V+30.)*(V+30.))+5.)
    hCaL_next = hCaL_inf + (hCaL - hCaL_inf) * exp(-dt / tau_hCaL) if dt < tau_hCaL else hCaL_inf

    hCaL2_inf = 0.2/(1+exp((V+5.)/-10.)) + 1/(1+exp((V+40.)/10.))
    tau_hCaL2 = Q10tauFactor*(225.*exp(-(0.0275)*(0.0275)*(V+40.)*(V+40.)) + 75.)
    hCaL2_next = hCaL2_inf + (hCaL2 - hCaL2_inf) * exp(-dt / tau_hCaL2) if dt < tau_hCaL2 else hCaL2_inf

    gCaL = Q10MaxtauFactor * GCaL * mCaL_next * (0.55*hCaL_next + 0.45*hCaL2_next)
    I_CaL = gCaL * (V - E_Ca)
    #"""

    # M current (pA), Wheeler & Horn, 2004
    mM_inf = 1 / (1 + exp(-(V + 35) / 10))
    tau_mM = Q10tauFactor*2000 / (3.3 * (exp((V + 35) / 40) + exp(-(V + 35) / 20)))
    mM_next = mM_inf + (mM - mM_inf) * exp(-dt / tau_mM) if dt < tau_mM else mM_inf
    gM = Q10MaxtauFactor * GM * power(mM_next, 2)
    I_M = gM * (V - E_K)

    #"""# Somatic KCa current (pA), Ermentrout & Terman 2010
    mKCa_inf = CaS ** 2 / (CaS ** 2 + SCa ** 2)
    tau_mKCa = Q10tauFactor*tauKCa_0 / (1 + (CaS / SCa) ** 2)
    mKCa_next = mKCa_inf + (mKCa - mKCa_inf) * exp(-dt / tau_mKCa) if dt < tau_mKCa else mKCa_inf
    gKCa = Q10MaxtauFactor * GKCa * power(mKCa_next, 1)
    I_KCa = gKCa * (V - E_K)
        #"""

    """# Somatic KCa current (pA) - Based on 8.1. A-and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristicsjn.1994.71.6.2338
    alpha_mKCa = 750. * CaS * exp((V-10.)/12.)
    beta_mKCa = 0.05 * exp((V-10.)/-60.)
    tau_mKCa = 4.5/(alpha_mKCa+beta_mKCa)
    mKCa_inf = alpha_mKCa/(alpha_mKCa+beta_mKCa)
    mKCa_next = mKCa_inf + (mKCa - mKCa_inf) * exp(-dt / tau_mKCa) if dt < tau_mKCa else mKCa_inf
    gKCa = GKCa * power(mKCa_next, 1)
    I_KCa = gKCa * (V - E_K)
    """

    """# KCa Current (large conductance BKCa based on Furlan RNA exp.) from 1. 2018-A biophysically detailed computational model-journal.pcbi.1006293
    pCa = log10(CaS * 0.001)
    sf = 33.88 * exp(-(((pCa+5.42)/1.85)**2))
    mKCa_inf = 1/(1+exp((-43.4 * pCa - 203 -V)/sf))
    tau_mKCa = 5.55 * exp(V/42.91) + 0.75 - 0.12 * V
    mKCa_next = mKCa_inf + (mKCa - mKCa_inf) * exp(-dt / tau_mKCa) if dt < tau_mKCa else mKCa_inf
    gKCa = GKCa * mKCa_next
    I_KCa = GKCa * (V - E_K)
    """

    """ #BKCa channel from https://senselab.med.yale.edu/ModelDB/ShowModel?model=111877&file=/IKCa/Ikca_Mocz.ode#tabs-2
    d1 = 0.84; d2 = 1.0; k1 = 0.18; k2 = 0.011; bbar = 0.28; abar = 0.48; celsius = 20; fara=96.485;
    gkbar = 0.01; #cai = 0.1; ko=5.4; ki=140;
    alpha_mKCa = abar / (1 + k1 * exp(-2 * d1 * fara * V / 8.313424 / (273.15 + celsius)) / CaS)
    beta_mKCa = bbar / (1 + CaS / (k2 * exp(-2 * d2 * fara * V / 8.313424 / (273.15 + celsius))))
    tau_mKCa = 1 / (alpha_mKCa + beta_mKCa)
    mKCa_inf = alpha_mKCa * tau_mKCa
    mKCa_next = mKCa_inf + (mKCa - mKCa_inf) * exp(-dt / tau_mKCa) if dt < tau_mKCa else mKCa_inf
    gKCa = GKCa * mKCa_next
    I_KCa = GKCa * (V - E_K)
    """
    #For both the above BKCa channel models, the current just mimics the VR protocol

    """ #KCa sympathetic ganglion cells https://senselab.med.yale.edu/modeldb/ShowModel?model=183949&file=/Discharge_hysteresis/kca2.mod#tabs-2
    """

    """ #KCa L-type Ca1.2 from dentate granule cells GC model https://senselab.med.yale.edu/modeldb/ShowModel?model=231818&file=/BeiningEtAl2017nrn/lib_mech/BK.mod#tabs-2
    """

    """ #Original
    # A-type potassium current (pA), Rush & Rinzel, 1995
    mA_inf = (0.0761 * exp((V + 94.22) / 31.84) / (1 + exp((V + 1.17) / 28.93))) ** (1/3)
    tau_mA = 0.3632 + 1.158 / (1 + exp((V + 55.96) / 20.12))
    mA_next = mA_inf + (mA - mA_inf) * exp(-dt / tau_mA) if dt < tau_mA else mA_inf

    hA_inf = (1 / (1 + exp(0.069 * (V + 53.3)))) ** 4
    tau_hA = (0.124 + 2.678 / (1 + exp((V + 50) / 16.027))) * tau_hA_scale
    hA_next = hA_inf + (hA - hA_inf) * exp(-dt / tau_hA) if dt < tau_hA else hA_inf

    gA = GA * power(mA_next, 3) * hA_next
    I_A = gA * (V - E_K)
    """

    """#Modified
    # A-type potassium current (pA), Rush & Rinzel, 1995
    #mA_inf = (0.0761 * exp((V + 94.22) / 31.84) / (1 + exp((V + 1.17) / 28.93))) ** (1/3) #Original
    #mA_inf = (0.0761 * exp((V + 94.22) / 20) / (1 + exp((V + 1.17) / 28.93))) ** (1/3) #This version doesn't plateau at higher V-clamp values
    mA_inf = (0.03 * exp(2.5*(V + 0.4*94.22 + 0.4*50 ) / 31.84) / (1 + exp(2.5*(V + 0.4*1.17 - 0.4*30 + 0.4*50) / 28.93))) ** (1 / 3) # This version plateaus (depressed) at higher V-clamp valuesl

    tau_mA = (0.3632 + 1.158 / (1 + exp((V + 55.96) / 20.12))) * tau_mA_scale
    mA_next = mA_inf + (mA - mA_inf) * exp(-dt / tau_mA) if dt < tau_mA else mA_inf

    hA_inf = (1 / (1 + exp(0.069 * (V + 53.3)))) ** 4
    tau_hA = (0.124 + 2.678 / (1 + exp((V + 50) / 16.027))) * tau_hA_scale
    hA_next = hA_inf + (hA - hA_inf) * exp(-dt / tau_hA) if dt < tau_hA else hA_inf

    gA = GA * power(mA_next, 3) * hA_next
    I_A = gA * (V - E_K)
    """

    """#A-current mh model
    #https://elifesciences.org/articles/26517
    #https://github.com/MarcelBeining/Dentate-Granule-Cell-Model/blob/master/lib_mech/Kv34.mod

    #mA_inf = 1 / (1 + exp(((V - (10)) / (-5))) + exp(((V - (45)) / (16))))
    mA_inf = 1/(1+exp(((V -14)/(-9.7))))
    scale_a = 4
    am = scale_a * (1. / 16.) * exp(0.1 * 0.5 * (V - 38))
    bm = scale_a * (1. / 16.) * exp(-0.1 * 0.5 * (V + 45))
    tau_mA = 1 / (am + bm)
    mA_next = mA_inf + (mA - mA_inf) * exp(-dt / tau_mA) if dt < tau_mA else mA_inf

    hA_inf = 0.1 + (1 - 0.1) / (1 + exp(((V - (-29.7)) / (12.2))))
    tau_hA = 250 / (1 + exp(((V - (-10)) / (17)))) + 8
    hA_next = hA_inf + (hA - hA_inf) * exp(-dt / tau_hA) if dt < tau_hA else hA_inf

    gA = GA * 2. * mA_next * hA_next
    I_A = gA * (V - E_K)
    
    """

    #""" # A-current m^3*h model 2021-Excitation properties of computational models of unmyelinated peripheral-jn.00315.2020, 7.1. 2020-Examining Sodium and Potassium Channel Conductances Involved in Hyperexcitability of Chemotherapy-Induced Peripheral Neuropathy: A Mathematical and Cell Culture-Based Study-Examining_Sodium_and_Potassium_Channel_Conductance, 7.2. 2020-Using Bifurcation Theory for Exploring Pain-acs.iecr.9b04495
    # Based on 8.1. A-and C-type rat nodose sensory neurons: model interpretations of dynamic discharge characteristicsjn.1994.71.6.2338
            #These are neurons in nodose ganglion - therefore a kind of tSPNs?
    mA_inf = 1./(1+exp((V+28.)/-28.))   #original
    #mA_inf = 1. / (1 + exp((V + 10 + 28.) / -28.)) #modified
    mA_inf = 1. / (1 + exp((V + 28.) / -18.))  # modified
    mA_inf = 1. / (1 + exp((V + 10 + 28.) / -18.))  # modified +10 after junction potential adjustment, same in below equations

    tau_mA = 5. * exp(-(0.022)*(0.022)*(V+65.)*(V+65.)) + 2.5 #original
    tau_mA = 0.5 * exp(-(0.022)*(0.022)*(V+65.)*(V+65.)) + 2 #modified
    tau_mA = 1. * exp(-(0.022)*(0.022)*(V+65.)*(V+65.)) + 1.1 #modified
    tau_mA = Q10tauFactor*(1. * exp(-(0.022)*(0.022)*(V+10+65.)*(V+10+65.)) + 1.1) #modified
    mA_next = mA_inf + (mA - mA_inf) * exp(-dt / tau_mA) if dt < tau_mA else mA_inf

    hA_inf = 1./(1.+exp((V+58.)/7.))
    hA_inf = 1./(1.+exp((V+10+58.)/7.))
    #tau_hA = 100. * exp(-(0.035)*(0.035)*(V+30.)*(V+30.)) + 10.5 #original
    #tau_hA = 30. * exp(-(0.035)*(0.035)*(V+30.)*(V+30.)) + 10.5 #modified
    tau_hA = 12. * exp(-(0.035) * (0.035) * (V + 30.) * (V + 30.)) + 9  # modified
    tau_hA = Q10tauFactor*(12. * exp(-(0.035) * (0.035) * (V+10 + 30.) * (V+10 + 30.)) + 9)  # modified
    hA_next = hA_inf + (hA - hA_inf) * exp(-dt / tau_hA) if dt < tau_hA else hA_inf

    gA = Q10MaxtauFactor * GA * mA_next * mA_next * mA_next * hA_next
    I_A = gA *  (V - E_K)
    #"""

    """ # A current mh model 7.2. 2020-Using Bifurcation Theory for Exploring Pain-acs.iecr.9b04495, 7.2.1. 2020-Using Bifurcation Theory for Exploring Pain-Suppl, 7.2.2. 2011 - Physiological interactions between Nav1.7 and Nav1.8 sodium channels: a computer simulation study -jn.00100.2011
    # Based on 7.2.3. 1996 - Characterization of Six Voltage-Gated K+ Currents in Adult Rat Sensory Neurons -jn.1996.75.6.2629

    mA_inf = (1/(1+exp(-(V + 5.4)/16.4)))**4
    tau_mA = (0.25 + 10.04 * exp(-(((V+24.67)**2)/(2*34.8**2))))
    mA_next = mA_inf + (mA - mA_inf) * exp(-dt / tau_mA) if dt < tau_mA else mA_inf

    hA_inf = 1 / (1 + exp((V + 49.9) / 4.6))
    tau_hA =  (20 + 50 * exp(- ((V + 40) ** 2)/(2 * 40 ** 2)))
    if tau_hA < 5.: tau_hA = 5.
    hA_next = hA_inf + (hA - hA_inf) * exp(-dt / tau_hA) if dt < tau_hA else hA_inf

    gA = GA * mA_next * hA_next
    I_A = gA * (V - E_K)
    """


    # Ih (pA), Based on Kullmann et al., 2016
    mh_inf = 1 / (1 + exp((V + 87.6) / 11.7))
    tau_mh_activ = Q10tauFactor*(53.5 + 67.7 * exp(-(V + 120) / 22.4))
    tau_mh_deactiv = Q10tauFactor*(40.9 - 0.45 * V)
    tau_mh = tau_mh_activ if mh_inf > mh_inf_prev else tau_mh_deactiv
    mh_next = mh_inf + (mh - mh_inf) * exp(-dt / tau_mh)
    gh = Q10MaxtauFactor * Gh * mh_next
    I_h = gh * (V - E_h)

    # Leak current (pA)
    I_leak = Gleak * (V - E_leak)

    # Synaptic current (pA)
    I_syn = Gsyn * (V - E_syn)

    # Somatic calcium concentration (uM), Kurian et al., 2011 & Methods of Neuronal Modeling, p. 490. 12.24
    CaS_next = CaS * exp(-f * kCaS * dt / Q10tauFactor) - alpha / kCaS * I_CaL * (1 - exp(-f * kCaS * dt / Q10tauFactor))

    # pump current - cubic depndncy on [Na+] - 2.1. 2018-Biophysical models reveal the relative importance of transporter proteins and impermeant anions in chloride homeostasis-elife-39575-v3
        # this formula is mainly for v calculated from ion conc.; also includes cell volume; both of which are ignored for now.
        # check these parmtrs at equilibrium/holding potential first so cncntrtns rmyn fixed.. then see behavior under bigger currnt inputs..
    #P = 0; Am=0;   # removes any affect of pump; #P = 1000; Am = 0.001; #Am controls the change in conc. due to ion channels; P controls change in conc. due to pump
    # at holding potential, make sure it keeps [Na+], [K+] at default internal levels
    Jp = Pump_P * power(Nai/Nao,3);
    Nai_next = Nai + dt * (-Pump_Am/F) * ((gNa + (ELeakMultiplier/1.28)*Gleak)*(V-E_Na) + 3*Jp) / Q10tauFactor;
            # d(nai) / dt = (-Am / F) * (INa_all + 3Jp)
            #             = () * (gNa*(V-  (R*T/F) * log(Nao/Nai)) + 3*Pump_P*power(Nai/Nao,3));
            #                   Too complex to solve
            # General form mh_next = mh + (mh - mh_inf) * (exp(-dt / tau_mh)-1)
            # if d(mh/dt) = k; what are mh_inf and tau_mh? both infinity in this case, what's mh_next? just mh+k*dt?   so d(mh/dt) = (mh_inf-mh)/tau = k ==> mh_inf = k*tau = infiity
            # ==> mh_next = mh + limit(x->0) (k/x)(1-e^(-dt*x) --> L-Hospital rule derivatives top bottom -->
            # ==> mh_next = mh + k*dt -- same as regular euler.. so it should be okay to use regular euler here?
    Ki_next = Ki + dt * (-Pump_Am / F) * ((gK + gA + gM + gKCa + (1/1.28)*Gleak) * (V - E_K) - 2 * Jp) / Q10tauFactor;
    I_pump = -(-3 * Jp + 2 * Jp);     # Since outgoing current is +ve
            # Not exponential Euler -- how to change to exp. Euler from regular Euler in this case?
                    # See above with Nai limits and L'Hospital rule -- so

    ## update voltage
    g_inf = gNa + gCaL + gK + gA + gM + gKCa + gh + Gleak + Gsyn
    if ccOrVc: #current clamp
        currentClamp = clampValue[0]
        V_nearElectrode_next = V_nearElectrode
    else: #voltage clamp
        currentClamp = 0

    #print(currentClamp)
    #V_inf = (int(currentClamp) + gNa * E_Na + gCaL * E_Ca + (gK + gA + gM + gKCa) * E_K + gh * E_h + Gleak * E_leak + Gsyn * E_syn) / g_inf
    V_inf = (currentClamp-I_pump + gNa * E_Na + gCaL * E_Ca + (gK + gA + gM + gKCa) * E_K + gh * E_h + Gleak * E_leak + Gsyn * E_syn) / g_inf
    tau_tspn = C / g_inf

    #""" # Space clamp with two-compartment model separating electrode and the cell
    V_next = V
    if ccOrVc:  # current clamp
        V_next = V_inf + (V - V_inf) * exp(-dt / tau_tspn)
    else:  # voltage clamp/ramp
        #V_nearElectrode = clampValue  # voltageClamp

        #dV/dt = gBetweenCompartments * (V_nearElectrode - V)/C   +  (V_inf - V)/tau_tspn
        tau_VClampDiff = C/gBetweenCompartments; #tau_tspn = C/g_inf
        #tau_VClampDiff = C/(gBetweenCompartments-9); #tau_tspn = C/g_inf ## the offset to gBetweenCompartments seems to control the shape of the inward current onset delay curve vs VClamp.. possibly accounting for the current coming from electrode leaking to the cell axon, dendrites etc? So, we should use gBetweenCompartments only for computing initial artifact and replace it with gBetweenCompartments-9 everywhere else..

        V_inf_updatedWithElectrodeInput = (tau_tspn * V_nearElectrode + tau_VClampDiff * V_inf) / (tau_VClampDiff + tau_tspn)
        tau_tspn_updatedWithElectrodeInput = ((tau_VClampDiff * tau_tspn) / (tau_VClampDiff + tau_tspn))
        V_next = V_inf_updatedWithElectrodeInput + (V - V_inf_updatedWithElectrodeInput) * exp(-dt / tau_tspn_updatedWithElectrodeInput) if dt < tau_tspn_updatedWithElectrodeInput else V_inf_updatedWithElectrodeInput

        I_nearElectrode = - gBetweenCompartments * (V_next - V_nearElectrode) # /C_electrode coresponding to electrode
                #TODO rising phase of electrode artifact -- basically capacitive current to charge electrode will be added to I_nearElectrode
            #TODO see pClamp user guide - https://mdc.custhelp.com/euf/assets/content/pCLAMP10_User_Guide.pdf
        # TODO See also meeting notes with Dr. Prinz - screenshots for the electrode circuit from 25 May 2021
        #gElectrode=0.8*600.*2.8; CElectrode = 0.8*1./6.; #tau_Electrode = CElectrode/gElectrode;  #Matching tau gives estimate for CE/gE and then shape gives approx. of each of CE, gE? Not exactly as gBetweenComprtmnts also plays a role
        #dVElectrode/dt = gElectrode(clampValue - VElectrode)/CElectrode  #assuming outgoing current to second comprtmnt doesn't play a role during rising phase
        #V_nearElectrode_next = clampValue + (V_nearElectrode - clampValue) * exp(-dt / tau_Electrode) if dt < tau_Electrode else clampValue;
        #dVElectrode/dt = (gElectrode(clampValue - VElectrode) + gBetweenCompartments( V_next - VElectrode))/CElectrode  #including outgoing current to second comprtmnt
                    #   = (gElectrode*clampValue+gBetweenCompartments*V_next - (gElectrode+gBetweenCompartments)*VElectrode)/CElectrode

        #V_nearElectrode_inf = (gElectrode*clampValue+gBetweenCompartments*V_next)/(gElectrode+gBetweenCompartments)  #Since VClamp injected current cancels what's going thru gBetweenCompartments, this wouldn't play a role in V_nearElectrode_inf
        #tau_nearElectrode = CElectrode/(gElectrode+gBetweenCompartments)
        V_nearElectrode_inf = clampValue
        tau_nearElectrode = CElectrode/gElectrode
        V_nearElectrode_next = V_nearElectrode_inf + (V_nearElectrode - V_nearElectrode_inf) * exp(-dt / tau_nearElectrode) if dt < tau_nearElectrode else V_nearElectrode_inf;

        #""" # End - Space clamp with two-compartment model separating electrode and the cell
    if ccOrVc:
        x24 = g_inf  #Using this variable to plot input resistance in case of CC
    else:
        x24 = V_nearElectrode_next #Using this to plot V near electrode to model its charging during VC
    y_next = [V_next, CaS_next, m_next, h_next, n_next, mA_next, hA_next, mh_next, mM_next, mCaL_next, hCaL_next, s_next, mKCa_next, I_Na, I_K,
          I_CaL, I_M, I_KCa, I_A, I_h, I_leak, mh_inf, hCaL2_next, I_nearElectrode, x24, Nai_next, Ki_next]

    #print(y_next)
    return y_next