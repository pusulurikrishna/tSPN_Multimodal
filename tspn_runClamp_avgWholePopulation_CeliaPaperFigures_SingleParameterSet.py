import tspn_ClampProtocol_Wrapper
import numpy as np
import matplotlib.pyplot as plt
from EPSPsPoisson import *

modelStepSize = 0.01; modelPlotSamplingRate = 25;
modelStepSize = 0.25; modelPlotSamplingRate = 1;

stepSize = modelStepSize; stepSizeAbf = 0.1  # ms   #check data resolution #After TTX
holdingCurrentOrHoldingVoltageIfAutoHold = 0+8 - 20; #adjusting for junction potential? // holdingCurrent
savePrefix = ''

#    gBetweenCompartments, GNa, GK, GCaL, GM, GKCa, GA, Gh, Gleak, C, NaShift, gElectrode, CElectrode,
        #    T, Pump_P, Pump_Am, ELeakMultiplier
# Passive properties and GMax changes matched to whole cell population
G = [34, 11954250, 82.8, 8.0, 2, 3, 43.7, 0.23, 0.115, 81, 26.3, 13.44, 0.7, 296, 0, 0.0, 0.7];
G_Destination = [34, 2*11954250, 1.25*82.8, 8.0, 2*2, 3, 1.5*43.7, 1.5*0.23, 1.5*0.115, 92, 26.3, 13.44, 0.7, 296, 0, 0.0, 0.7];
G_GLeakChangesOnly = [34, 11954250, 82.8, 8.0, 2, 3, 43.7, 0.23, 3*0.115, 81, 26.3, 13.44, 0.7, 296, 0, 0.0, 0.7];
G_GNaGAChangesOnly = [34, 3*11954250, 82.8, 8.0, 2, 3, 3*43.7, 0.23, 0.115, 81, 26.3, 13.44, 0.7, 296, 0, 0.0, 0.7];

## Blue to red curves as we go from G to G_Destination
## All the below parameters tuned to avg NPY+ cells.

##############################################################################################################
# Figure: IR changes rheobase - gLeak changes - Current clamp
# IR is dependent mainly on gA, GLeak and gH - not only gLeak but for demonstration of IR changes using Gleak
savePrefix = 'CeliaPaperOutput_'+str(modelStepSize)+'/NPY_IRChangesWith_gLeak_';
# Create cell
t = tspn_ClampProtocol_Wrapper.tspn(stepSize, stepSizeAbf, -70, G, autoHold=True, G_Destination=G_GLeakChangesOnly, noOfGsToPlot=2)
rowsToPlot = [0, 1]; rowHeights = [5, 2];
t.stepSize = modelStepSize; plotSamplingRate = modelPlotSamplingRate;
t.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=3, IStepIncrease=6, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix, plotSamplingRate=plotSamplingRate, IStepAdd=-1, plotFICurve=False)
###########################
# Input resistance changes with gLeak gA
t.autoHold = True; t.holdingCurrentOrHoldingVoltageIfAutoHold = -68
t.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=3, IStepIncrease=-10, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix+"IRChangesAt-68mV_asInData_", plotSamplingRate=plotSamplingRate, plotFICurve=False, plotAbf=True)
###########################
##############################################################################################################

##############################################################################################################
# Figure: AP changes with gNa gA - Current clamp
# While all of gNa, gK, gA, gCaL, gLeak as well as gM, gKCa, gh affect rheobase, if you look at the current steps where you only see one spike at rheobase, then the shape should be affected by gNa and gA only.
savePrefix = 'CeliaPaperOutput_'+str(modelStepSize)+'/NPY_APChangesWith_gNa_gA_';
# Create cell
t = tspn_ClampProtocol_Wrapper.tspn(stepSize, stepSizeAbf, -70, G, autoHold=True, G_Destination=G_GNaGAChangesOnly, noOfGsToPlot=2)
rowsToPlot = [0, 1]; rowHeights = [5, 2];
t.stepSize = modelStepSize; plotSamplingRate = modelPlotSamplingRate;
t.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=3, IStepIncrease=5, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix, plotSamplingRate=plotSamplingRate, plotFICurve=True, IStepAdd=2)
##############################################################################################################

##############################################################################################################
# The following CC, VR, EPSP, IR CC, RMP CC are for chronic vs sham parameters
savePrefix = 'CeliaPaperOutput_'+str(modelStepSize)+'/NPY_chronicVsShamChanges_';
# Create cell
t = tspn_ClampProtocol_Wrapper.tspn(stepSize, stepSizeAbf, -70, G, autoHold=True, G_Destination=G_Destination, noOfGsToPlot=2)
t.stepSize = modelStepSize; plotSamplingRate = modelPlotSamplingRate;
###########################
# Current clamp
rowsToPlot = [0, 1]; rowHeights = [5, 2];
t.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=13, IStepIncrease=10, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix, plotSamplingRate=plotSamplingRate)
###########################
###########################
# Voltage clamp - ramp
abfLocation = "CeliaData/L39 (8-21-2018)-20210426T193143Z-001/L39 (8-21-2018)/18821018.abf" #Before TTX
rowsToPlot = [0, 1, 3, 4, 5, 6, 7, 8, 9]; rowHeights = [3,5,1,1,1,1,1,1,1];
vClampStepsStartingValues=np.array([-80, -80, -90, 10, -80])-t.junctionPotentialAdjustment; vClampStepsEndingValues=np.array([-80, -90, 10, -80, -80])-t.junctionPotentialAdjustment;
t.stepSizeAbf = 0.1  # ms   #check data resolution #After TTX
t.voltageClampRamp(vClampStepsDurations=np.array([2000, 1000, 4000, 4000, 2000]), vClampStepsStartingValues=vClampStepsStartingValues,vClampStepsEndingValues=vClampStepsEndingValues, rowsToPlot = rowsToPlot, rowHeights = rowHeights, plotFirstCompartmentV=False, plotOverlappingCurrents=False, plotAbf=True,abfLocation=abfLocation, abfOffset=660 + 2000 - 1172.57, plotGDestination=True, plotSamplingRate=plotSamplingRate, savePrefix=savePrefix)
###########################
###########################
# Current clamp with EPSPs
rowsToPlot = [0, 1]; rowHeights = [5, 2];
#TODO: Autocompute holding current to maintain cell at -60mVw
doEpscSimulations = True; doGSynSimulations = True;
doEpscSimulations = True; doGSynSimulations = False;
# Chronic cell
tChronic = tspn_ClampProtocol_Wrapper.tspn(stepSize, stepSize, -60, G, autoHold=True, G_Destination=G_Destination, noOfGsToPlot=1)
# Sham cell
tSham = tspn_ClampProtocol_Wrapper.tspn(stepSize, stepSize, -60, G_Destination, autoHold=True, G_Destination=G_Destination, noOfGsToPlot=1)
tChronic.autoHold = False; tSham.autoHold = False
tChronic.holdingCurrentOrHoldingVoltageIfAutoHold = 0; tSham.holdingCurrentOrHoldingVoltageIfAutoHold = 0
tChronic.stepSize = modelStepSize; tSham.stepSize = modelStepSize
plotSamplingRate = modelPlotSamplingRate;
    # Comment above two line to use holding potential of -60mV
# Modeling injected currents and Gsyns separately for matching with EPSCs in data
rates = [2,4,5,6,10,20];
rates = [2,4,5,6,10];
rates = [4, 6];
timeEPSCSyns = []; epscSimulations = []; timeEPSCSynsHA = []; epscSimulationsHA = [];
timeGSyns = []; gSynSimulations = []; timeGSynsHA = []; gSynSimulationsHA = []; timeGSynsHA2 = []; gSynSimulationsHA2 = [];
for rate in rates:
    #epsc simulations
    if doEpscSimulations:
        # timeEPSCSyn, epscSimulation = model_epsc_ipsc(rate=rate, duration=20000, amplitude=20 * 2, tau_rise=2, tau_decay=10,
        #                                               inward=False, step_size=modelStepSize)
        # timeEPSCSyns.append(timeEPSCSyn); epscSimulations.append(epscSimulation)
        timeEPSCSynHA, epscSimulationHA = model_epsc_ipsc(rate=rate, duration=20000, amplitude=20 * 2 * 2, tau_rise=2,
                                                          tau_decay=10, inward=False, step_size=modelStepSize)
        timeEPSCSynsHA.append(timeEPSCSynHA); epscSimulationsHA.append(epscSimulationHA)
    # gSyn simulations
    if doGSynSimulations:
        timeGSyn, gSynSimulation = model_epsc_ipsc(rate=rate, duration=20000, amplitude=0.6, tau_rise=2, tau_decay=10,
                                                 inward=False, step_size=stepSize)
        timeGSyns.append(timeGSyn); gSynSimulations.append(gSynSimulation);
        timeGSynHA, gSynSimulationHA = model_epsc_ipsc(rate=rate, duration=20000, amplitude=1.2, tau_rise=2, tau_decay=10,
                                                     inward=False, step_size=modelStepSize)
        timeGSynsHA.append(timeGSynHA); gSynSimulationsHA.append(gSynSimulationHA);
        timeGSynHA2, gSynSimulationHA2 = model_epsc_ipsc(rate=rate, duration=20000, amplitude=2.5, tau_rise=2, tau_decay=10,
                                                       inward=False, step_size=modelStepSize)
        timeGSynsHA2.append(timeGSynHA2); gSynSimulationsHA2.append(gSynSimulationHA2);
for i,rate in enumerate(rates):
    if doEpscSimulations:
        # Lower amplitudes EPSCs
        # tSham.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix='Output/EPSPModeling/epscSimulations/sham_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=epscSimulations[i], plotFICurve=False, plotAbf=False)
        # tChronic.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix='Output/EPSPModeling/epscSimulations/chronic_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=epscSimulations[i], plotFICurve=False, plotAbf=False)
        # Higher amplitudes EPSCs
        tSham.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix+'epscSimulations_sham_HA_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=epscSimulationsHA[i], plotFICurve=False, plotAbf=False)
        tChronic.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix+'epscSimulations_chronic_HA_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=epscSimulationsHA[i], plotFICurve=False, plotAbf=False)
    if doGSynSimulations:
        # Lower amplitude gSyns
        tSham.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix='Output/EPSPModeling/gSynSimulations/sham_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=0, plotFICurve=False, plotAbf=False, g_syn_template=gSynSimulations[i])
        tChronic.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix='Output/EPSPModeling/gSynSimulations/chronic_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=0, plotFICurve=False, plotAbf=False, g_syn_template=gSynSimulations[i])
        # Higher amplitude gSyns
        tSham.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix='Output/EPSPModeling/gSynSimulations/sham_HA_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=0, plotFICurve=False, plotAbf=False, g_syn_template=gSynSimulationsHA[i])
        tChronic.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix='Output/EPSPModeling/gSynSimulations/chronic_HA_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=0, plotFICurve=False, plotAbf=False, g_syn_template=gSynSimulationsHA[i])
        # Even higher amplitude gSyns
        tSham.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix='Output/EPSPModeling/gSynSimulations/sham_HA2_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=0, plotFICurve=False, plotAbf=False, g_syn_template=gSynSimulationsHA2[i])
        tChronic.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=1, IStepIncrease=0, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix='Output/EPSPModeling/gSynSimulations/chronic_HA2_'+str(rate)+'Hz_', plotSamplingRate=plotSamplingRate, epscToAddToCC=0, plotFICurve=False, plotAbf=False, g_syn_template=gSynSimulationsHA2[i])
# With current step fI curves
# timeEPSC, epsc = model_epsc_ipsc(rate=4, duration=200000, amplitude=80, tau_rise=2, tau_decay=100, inward=False, step_size=modelStepSize)
# t.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=13, IStepIncrease=10, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix, plotSamplingRate=plotSamplingRate, epscToAddToCC=epsc)
###########################
###########################
# Input resistance measure at RMP or at -70mV?  seems to not change between both voltages (~-70mV or RMP) for chronic and sham (2200 vs 1800)
rowsToPlot = [0, 1]; rowHeights = [5, 2];
t.stepSize = modelStepSize; plotSamplingRate = modelPlotSamplingRate;
t.autoHold = True; t.holdingCurrentOrHoldingVoltageIfAutoHold = -68
t.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=3, IStepIncrease=-10, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix+"IRChangesAt-68mV_asInData_", plotSamplingRate=plotSamplingRate, plotFICurve=False, plotAbf=True)
###########################
###########################
# Current clamp to 0pA to measure RMP - Celia's RMP for cell is about -50mV; Avg is about -60mV
rowsToPlot = [0, 1]; rowHeights = [5, 2];
t.stepSize = modelStepSize; plotSamplingRate = modelPlotSamplingRate;
t.holdingCurrentOrHoldingVoltageIfAutoHold = 0;
t.autoHold=False;
t.currentClampStep(startOfFirstIStep=5000, durationOfIStep=3000, durationPostIStep=12000, durationLast=0 * 50000, numberOfISteps=3, IStepIncrease=-10, rowsToPlot=rowsToPlot, rowHeights=rowHeights, saveData=True, savePrefix=savePrefix+"RMP_IR_tau_", plotSamplingRate=plotSamplingRate, plotFICurve=False, plotAbf=True)
###########################
###############################################################################################################

plt.show()
