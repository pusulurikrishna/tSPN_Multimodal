from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import ode_solver
import tspn_ClampProtocol_Integrator as tspn_Clamp
import pyabf
import pyabf.filter
import matplotlib.lines as mlines
import csv

class tspn(object):
        #Celia's recorindgs holding voltage seems to be -60mV for CC; For VC, starts at -80mV before step/ramp
        #2019 tSPN paper holding voltage is -70mV
    def __init__(self, stepSize, stepSizeAbf, holdingCurrentOrHoldingVoltageIfAutoHold, G, G_Destination=None, noOfGsToPlot=1, autoHold=False, NaShift=0):
        self.stepSize = stepSize;
        self.stepSizeAbf = stepSizeAbf;
        self.holdingCurrentOrHoldingVoltageIfAutoHold = holdingCurrentOrHoldingVoltageIfAutoHold;
        self.autoHold=autoHold;
        self.G = np.asarray(G); # gBetweenCompartments, GNa, GK, GCaL, GM, GKCa, GA, Gh, Gleak, C, NaShift, gElectrode, CElectrode, T, Pump_P, Pump_Am
        self.G_Destination = np.asarray(G_Destination) if G_Destination != None else self.G;
        self.noOfGsToPlot=noOfGsToPlot;
        self.junctionPotentialAdjustment = 10
        self.colors = plt.cm.rainbow(np.linspace(0, 1, self.noOfGsToPlot))

        #Function to compute cumulative current clamp steps
    def computeCurrentClampSteps(self, startOfFirstIStep, durationOfIStep, durationPostIStep, durationLast, numberOfISteps, IStepIncrease, IstepAdd = 0):
        t_total = startOfFirstIStep + (
                    durationOfIStep + durationPostIStep) * numberOfISteps + durationLast  # ms; total simulation time
        t_len = int(t_total / self.stepSize)
        t_template = np.arange(0, t_total, self.stepSize)

        # compute current clamp (cc) template (efficient)
        cc_template = np.zeros((int(startOfFirstIStep / self.stepSize), 1))
        for j in range(0, numberOfISteps, 1):
            cc_I = IStepIncrease * (j + 1) * np.ones((int(durationOfIStep / self.stepSize), 1)) + IstepAdd
            cc_noI = np.zeros((int(durationPostIStep / self.stepSize), 1))
            cc_template = np.concatenate((cc_template, cc_I, cc_noI))
        cc_last = np.zeros((int(durationLast / self.stepSize), 1))
        cc_template = np.concatenate((cc_template, cc_last))
        #cc_template = cc_template + self.holdingCurrentOrHoldingVoltageIfAutoHold

        return cc_template, t_template, t_len


    #Function to compute voltage clamp/ramp steps
    def computeVoltageClampSteps(self, vClampStepsDurations, vClampStepsStartingValues, vClampStepsEndingValues):
        t_total = np.sum(vClampStepsDurations)  # ms; total simulation time
        t_len = int(t_total / self.stepSize)
        t_template = np.arange(0, t_total, self.stepSize)

        # compute voltage clamp (vc) template
        vc_template = np.zeros(0)
        numberOfVClampSteps = len(vClampStepsDurations)
        for j in range(numberOfVClampSteps):
            vc_duration = vClampStepsDurations[j];
            vc_len = vc_duration / self.stepSize;
            vc_startingV = vClampStepsStartingValues[j]
            vc_endingV = vClampStepsEndingValues[j]
            vc_step = np.linspace(vc_startingV, vc_endingV, num=int(vc_len), endpoint=False)
            vc_template = np.concatenate((vc_template, vc_step))

        return vc_template, t_template, t_len

        #If rowsToPlot and respective heights not specified, all 10 rows are plotted with equal height of 1;
    def createAxes_CC(self, figtitle='', rowsToPlot=None, rowHeights=None):
        nRows = 10;  # Voltage, currentClamp/tatalCurrentInVoltageClamp, 8 different currents
        nCols = 1;  # traces
        if rowsToPlot is None:
            rowsToPlot=range(nRows)
            rowHeights = np.ones(nRows)
        else:
            if rowHeights is None:
                rowHeights = np.ones(len(rowsToPlot))
            nRows = sum(rowHeights)

        # fig = plt.figure(figsize=(20, 4*nRows))
        fig = plt.figure(figsize=(10, 5))

        #fig.suptitle(figtitle, fontsize=16)
        gs1 = gridspec.GridSpec(nRows, nCols)
        gs1.update(wspace=0.03, hspace=0.2)  # set the spacing between axes.
        axLabels = ['V (mV)', 'I_ext (pA)', 'I_Na (pA)', 'I_K (pA)', 'I_CaL (pA)', 'I_M (pA)', 'I_KCa (pA)', 'I_A (pA)', 'I_h (pA)', 'I_leak (pA)' ]
        axList = []
        currentRowLocation = 0
        for i in range(len(rowsToPlot)):
            nextRowLocation = currentRowLocation+int(rowHeights[i])
            ax = plt.subplot(gs1[currentRowLocation:nextRowLocation, :]);
            #ax.text(0.01, 0.9, axLabels[rowsToPlot[i]], ha='left', va = 'top', transform = ax.transAxes)
            axList.append(ax)
            currentRowLocation=nextRowLocation
        axList[0].set_title(figtitle)
        axList[0].set_xticks([])
        axList[0].set_yticks([-140, -80, -60, -20, 0, 40])
        axList[1].set_xlabel('Time (s)')
        axList[1].set_ylabel('Current (pA)')
        axList[0].set_ylabel('V (mV)')
        axList[0].tick_params(axis='y', labelrotation=90)

        axList[1].xaxis.set_major_locator(plt.MaxNLocator(4))
        axList[1].yaxis.set_major_locator(plt.MaxNLocator(4))
        axList[1].tick_params(axis='y', labelrotation=90)

        return axList, fig

    def plotTraces_CC(self, y, axList, t_template, ccOrVc=True, cc_template=None, skipInitial=0, rowsToPlot=None, plotOverlappingCurrents=False, plotFirstCompartmentV=False, plotColor='blue', plotSamplingRate = 1, g_syn_template=None):
        axId = 0
        if rowsToPlot is None or 0 in rowsToPlot:
            axList[axId].plot(t_template[int(skipInitial/self.stepSize):][::plotSamplingRate], y[int(skipInitial/self.stepSize):,0][::plotSamplingRate], lw='1.5', color=plotColor); #Voltage traces
            #if plotFirstCompartmentV:
                #axList[axId].plot(t_template[int(skipInitial/self.stepSize):], y[int(skipInitial/self.stepSize):,24], color='k', lw=0.5); #V_nearElectrode
            axList[axId].set_xlim(t_template[int(skipInitial/self.stepSize)], t_template[-1])   #setting xlimits manually so it doesn't extend when plotting overlapping exp. data
            axId+=1
        totalCurrent=np.zeros(len(t_template))
        if rowsToPlot is None or 1 in rowsToPlot:
            axList[axId].set_xlim(t_template[int(skipInitial / self.stepSize)], t_template[-1])
            #total current plotted towards the end after computing the sum below
            axId+=1
        for i in range(8):
            current = y[:, 13 + i] # currents INa, IK, ICaL, IM, IKCa, IA, Ih, Ileak
            if rowsToPlot is None or i+2 in rowsToPlot:
                axList[axId].set_xlim(t_template[int(skipInitial / self.stepSize)], t_template[-1])
                axList[axId].plot(t_template[int(skipInitial/self.stepSize):][::plotSamplingRate], current[int(skipInitial/self.stepSize):][::plotSamplingRate], color=plotColor)
                axId += 1
            if plotOverlappingCurrents:
                    axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate], current[int(skipInitial / self.stepSize):][::plotSamplingRate], color=plotColor)
                        #plotting all currents together so we can track which ones are important
            totalCurrent+= current
        #totalCurrent+= y[:, 23] #IArtifact
        totalCurrent_nearElectrode = y[:, 23]; totalCurrent = totalCurrent_nearElectrode;

        if rowsToPlot is None or 1 in rowsToPlot:
            if ccOrVc:
                #axList[1].plot(t_template[int(skipInitial/self.stepSize):], cc_template[int(skipInitial/self.stepSize):]-self.holdingCurrentOrHoldingVoltageIfAutoHold); # current clamp
                axList[1].plot(t_template[int(skipInitial/self.stepSize):][::plotSamplingRate], cc_template[int(skipInitial/self.stepSize):][::plotSamplingRate], color=plotColor); # current clamp
                if g_syn_template is not None:
                    axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                                   g_syn_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                                   color='purple');  # gsyn
                    I_syn = (0 - y[int(skipInitial / self.stepSize):, 0][::plotSamplingRate])* (g_syn_template[int(skipInitial / self.stepSize):][::plotSamplingRate])
                    axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                                   I_syn,
                                   color='green');  # current clamp
            else:
                axList[1].plot(t_template[int(skipInitial/self.stepSize):][::plotSamplingRate], totalCurrent[int(skipInitial/self.stepSize):][::plotSamplingRate], lw=1.5, color=plotColor) # tatalCurrentInVoltageClamp
                if plotOverlappingCurrents:
                    axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate], totalCurrent[int(skipInitial / self.stepSize):][::plotSamplingRate], color=plotColor)#, color='#1f77b4')  # tatalCurrentInVoltageClamp
            """
            if ccOrVc:
                inputResistance = 1000/y[:, 24]  ; #MOhm
                axList[1].plot(t_template[int(skipInitial / self.stepSize):], inputResistance[int(skipInitial / self.stepSize):], color='red')  # tatalCurrentInVoltageClamp
            """
        #sp.plot(t_template, y[:,18]*0.25, color='g') #IA
        return totalCurrent

    def initialValues(self):
        # initialization
        y0 = []
        y0.append(-65.)  # initial membrane potential; mV
        y0.append(0.001)  # intracellular [Ca2+]; mM
        y0.append(0.0000422117)  # m
        y0.append(0.9917)  # h
        y0.append(0.00264776)  # n
        y0.append(0.5873)  # mA
        y0.append(0.1269)  # hA
        y0.append(0.0517)  # mh
        y0.append(0.000025)  # mM
        y0.append(7.6e-5)  # mCaL
        y0.append(0.94)  # hCaL
        y0.append(0.4)  # s
        y0.append(0.000025)  # mKCa
        y0.append(0)  # INa
        y0.append(0)  # IK
        y0.append(0)  # ICaL
        y0.append(0)  # IM
        y0.append(0)  # IKCa
        y0.append(0)  # IA
        y0.append(0)  # Ih
        y0.append(0)  # Ileak
        y0.append(0)  # mh_inf
        y0.append(0.1)  #hCaL2
        y0.append(0) #Iartifact
        y0.append(0) # VClampDiff (for Na) after space clamp exponential increase/decrease
        y0.append(9.3171063545) # Nai
        y0.append(139.787094212) #Ki
        return y0

    def computeSpikes(self, voltageTrace, t_template, spikeAboveV):
            spikeTimingIndices = [];
            i = 0;
            for i in range(1, len(voltageTrace), 1):
                if voltageTrace[i] >= spikeAboveV and voltageTrace[i - 1] < spikeAboveV:
                    spikeTimingIndices.append(i)
            # time[spikeTimingIndicies] will give spike times
            return spikeTimingIndices

    def plotMaxFrequencies(self, axes, t_template, CC_template, voltageTrace, spikeAboveV, CCStepInterval=5000,
                               color='blue', axFICurvesSus=None, axfigFirstSpikes=None):
            print("\tPlotting fI curve")
            ## measuring initial frequency for each CC step
            maxFrequencies = []; susFrequencies = [];
            CC_steps = []; sus_CC_steps = [];
            firstSpikesTiming= []; firstSpikesVoltage = [];
            i = 0;
            j = 0;
            spikeTimingIndices = self.computeSpikes(voltageTrace, t_template, spikeAboveV)
            # print(spikeTimingIndices)
            # print(t_template[spikeTimingIndices])
            for i in range(1, len(spikeTimingIndices)):
                if t_template[int(spikeTimingIndices[i])] - t_template[int(spikeTimingIndices[i - 1])] > CCStepInterval:
                    # New CC step; j counts number of spikes per CC step while i counts total spikes
                    if(j>=3):
                        #For the previous CC step there were atleast four spikes and 3 ISIs, which we will use for susFI
                        susISI =  ((t_template[int(spikeTimingIndices[i-1])] - t_template[
                            int(spikeTimingIndices[i - 2])]) +
                                   (t_template[int(spikeTimingIndices[i - 2])] - t_template[
                                       int(spikeTimingIndices[i - 3])]) +
                                   (t_template[int(spikeTimingIndices[i - 3])] - t_template[
                                       int(spikeTimingIndices[i - 4])]))/3.
                        susFrequencies.append(1000. / susISI);
                        sus_CC_steps.append(CC_template[int(spikeTimingIndices[i-2])])
                            # Using i-1 here giving some CC steps as 0 for some reason
                    j = 0;
                else:
                    j = j + 1
                    if (j == 1):
                        maxFrequencies.append(1000. / (t_template[int(spikeTimingIndices[i])] - t_template[
                            int(spikeTimingIndices[i - 1])]))
                        CC_steps.append(CC_template[int(spikeTimingIndices[i])])
            # print(maxFrequencies)
            axes.plot(CC_steps, maxFrequencies, '-o', color=color)
            axes.set_title('MaxFI curve')
            if axFICurvesSus is not None:
                axFICurvesSus.plot(sus_CC_steps, susFrequencies, '-o', color=color)
                axFICurvesSus.set_title('SusFI curve (mean of last 3 ISIs')

            if axfigFirstSpikes is not None:
                nStepsToPlotSpike = int(25/self.stepSize);  # 25 ms before spike peak and 50 ms after
                firstSpikesTiming = (t_template[spikeTimingIndices[0]-nStepsToPlotSpike:spikeTimingIndices[0]+2*nStepsToPlotSpike])-t_template[spikeTimingIndices[0]-nStepsToPlotSpike];
                firstSpikesVoltage = voltageTrace[spikeTimingIndices[0]-nStepsToPlotSpike:spikeTimingIndices[0]+2*nStepsToPlotSpike];
                axfigFirstSpikes.plot(firstSpikesTiming, firstSpikesVoltage , '-', color=color);
                axfigFirstSpikes.set_title('First spike (ms vs. mV)')

            return CC_steps, maxFrequencies, sus_CC_steps, susFrequencies, firstSpikesTiming, firstSpikesVoltage

    def currentClampAutoHold(self, G_present, timeToIntegrateForRMP=10000, maxAttempts=1000, holdVPrecisionInMV=1):

        cc_template, t_template, t_len = self.computeCurrentClampSteps(startOfFirstIStep=timeToIntegrateForRMP, durationOfIStep=0, durationPostIStep=0, durationLast=0, numberOfISteps=0, IStepIncrease=0);
        print("Binary search for holding current: \n")

        holdingCurrent = 0; holdingCurrentHigherBound = 200; holdingCurrentLowerBound = -100
        holdingVoltage = self.holdingCurrentOrHoldingVoltageIfAutoHold;
        attemptNumber = 0;

        y_current = ode_solver.update(tspn_Clamp.step, self.initialValues(), t_len, cc_template + holdingCurrent, G_present, self.stepSize, ccOrVc=True)
        lastVoltage = y_current[-1][0]
        print("\t AttemptNumber: {}, vHoldTarget: {}, vHold (Eq. potential): {}, Current: {}, LowerBound: {}, HigherBound: {};".format(attemptNumber, holdingVoltage, lastVoltage, holdingCurrent, holdingCurrentLowerBound,holdingCurrentHigherBound))
        attemptNumber = attemptNumber+1;

        while attemptNumber < maxAttempts and (lastVoltage - holdingVoltage > holdVPrecisionInMV or lastVoltage - holdingVoltage < -holdVPrecisionInMV):
            if lastVoltage - holdingVoltage > holdVPrecisionInMV:
                if holdingCurrent < holdingCurrentHigherBound:
                    holdingCurrentHigherBound = holdingCurrent
                holdingCurrent = (holdingCurrentHigherBound + holdingCurrentLowerBound) / 2
            elif lastVoltage - holdingVoltage < -holdVPrecisionInMV:
                if holdingCurrent > holdingCurrentLowerBound:
                    holdingCurrentLowerBound = holdingCurrent
                holdingCurrent = (holdingCurrentLowerBound + holdingCurrentHigherBound) / 2
            y_current = ode_solver.update(tspn_Clamp.step, self.initialValues(), t_len, cc_template + holdingCurrent,
                                          G_present, self.stepSize, ccOrVc=True)
            lastVoltage = y_current[-1][0]
            print("\t AttemptNumber: {}, vHoldTarget: {}, vHold (Eq. potential): {}, Current: {}, LowerBound: {}, HigherBound: {};".format(attemptNumber, holdingVoltage, lastVoltage, holdingCurrent, holdingCurrentLowerBound,holdingCurrentHigherBound))
            if holdingCurrentLowerBound == holdingCurrentHigherBound:
                print("\t Current bounds insufficient to find suitable holding current")
                attemptNumber = maxAttempts
            attemptNumber = attemptNumber+1

        if attemptNumber < maxAttempts:
            print("\tFound holding current: {}; Computing trajectory with holding current.\n".format(holdingCurrent))
        else:
            print("\tCouldn't find holding current in {} attempts. Computing trajectory with no holding current. \n".format(maxAttempts))
            holdingCurrent = 0

        return holdingCurrent;

    def currentClampStep(self, startOfFirstIStep = 5000, durationOfIStep=3000, durationPostIStep=5000, durationLast=0*50000, numberOfISteps = 13, IStepIncrease = 10, rowsToPlot=None, rowHeights=None, plotFICurve=True, saveData=False, savePrefix="", plotSamplingRate=1, epscToAddToCC=0, g_syn_template=None, plotAbf=True, IStepAdd = 0):
        #Cumulative current clamp steps
        cc_template, t_template, t_len = self.computeCurrentClampSteps(startOfFirstIStep, durationOfIStep, durationPostIStep, durationLast, numberOfISteps, IStepIncrease, IStepAdd)
        #ccOrVc = False for voltage clamp; True for current clamp
        axList, fig = self.createAxes_CC('Current clamp - cumulative steps', rowsToPlot=rowsToPlot, rowHeights=rowHeights)

        #print(cc_template.shape, epscToAddToCC.shape)
        cc_template[:,0] = cc_template[:,0] + epscToAddToCC;
        #print(cc_template.shape)
        csvFile, csvWriter, csvTableTimeVs = None, None, None
        csvFileFI, csvWriterFI = None, None
        csvFileFISus, csvWriterFISus = None, None
        csvFileFirstSpikes, csvWriterFirstSpikes = None, None

        if saveData:
            csvFile = open(savePrefix+'CellTuned_CC.csv', 'w')
            csvWriter = csv.writer(csvFile, delimiter=',')
            csvHeader = ['time', 'CCStep']
            for i in range(self.noOfGsToPlot):
                csvHeader.append('Model_V_'+str(i));
            csvWriter.writerow(csvHeader)
            csvTableTimeVs = np.zeros([t_len, 2+self.noOfGsToPlot], dtype='float')
            csvTableTimeVs[:, 0] = np.squeeze(t_template)
            csvTableTimeVs[:, 1] = np.squeeze(cc_template)

        if(plotFICurve):
            figFI = plt.figure(figsize=(5, 5))
            axFICurves = plt.subplot()
            figFISus = plt.figure(figsize=(5, 5))
            axFICurvesSus = plt.subplot()
            figFirstSpikes = plt.figure(figsize=(5, 5))
            axfigFirstSpikes = plt.subplot()

            if saveData:
                csvFileFI = open(savePrefix+'CellTuned_CC_maxFICurve.csv', 'w')
                csvWriterFI = csv.writer(csvFileFI, delimiter=',')
                csvWriterFI.writerow(['CC_Step', 'Max_Frequency', 'Curve_Type'])

                csvFileFISus = open(savePrefix+'CellTuned_CC_susFICurve.csv', 'w')
                csvWriterFISus = csv.writer(csvFileFISus, delimiter=',')
                csvWriterFISus.writerow(['CC_Step', 'Sus_Frequency', 'Curve_Type'])

                csvFileFirstSpikes = open(savePrefix+'CellTuned_CC_firstSpikes.csv', 'w')
                csvWriterFirstSpikes = csv.writer(csvFileFirstSpikes, delimiter=',')
                csvWriterFirstSpikes.writerow(['Time(ms)', 'Voltage(mV)', 'Curve_Type'])

        for i in range(self.noOfGsToPlot):

            print("Computing current clamp for G_present version "+str(i+1)+" out of "+str(self.noOfGsToPlot)+"\n");

            G_present = self.G if self.noOfGsToPlot==1 else self.G + (self.G_Destination - self.G) * i / (self.noOfGsToPlot-1);

            if(self.autoHold):
                holdingCurrent = self.currentClampAutoHold(G_present);
                y = ode_solver.update(tspn_Clamp.step, self.initialValues(), t_len, cc_template+holdingCurrent, G_present, self.stepSize, ccOrVc = True, g_syn_template=g_syn_template)
            else:
                y = ode_solver.update(tspn_Clamp.step, self.initialValues(), t_len, cc_template+self.holdingCurrentOrHoldingVoltageIfAutoHold, G_present, self.stepSize, ccOrVc = True, g_syn_template=g_syn_template)

            self.plotTraces_CC(y, axList, t_template/1000, True, cc_template, skipInitial=1000, rowsToPlot=rowsToPlot, plotColor=self.colors[i], plotSamplingRate=plotSamplingRate, g_syn_template=g_syn_template)

            if saveData:
                csvTableTimeVs[:, i+2] = y[:,0]

            if (plotFICurve):
                CC_steps, maxFrequencies, sus_CC_steps, susFrequencies, firstSpikesTiming, firstSpikesVoltage = self.plotMaxFrequencies(axFICurves, t_template, cc_template, y[:, 0], spikeAboveV=0, CCStepInterval=durationPostIStep, color=self.colors[i], axFICurvesSus=axFICurvesSus, axfigFirstSpikes=axfigFirstSpikes)
                if saveData:
                    for CC_step, maxFrequency in zip(CC_steps, maxFrequencies):
                        csvWriterFI.writerow([CC_step[0], maxFrequency, 'Model ' + str(i)]);
                    for sus_CC_step, susFrequency in zip(sus_CC_steps, susFrequencies):
                        csvWriterFISus.writerow([sus_CC_step[0], susFrequency, 'Model ' + str(i)]);
                    for firstSpikesTiming1, firstSpikesVoltage1 in zip(firstSpikesTiming, firstSpikesVoltage):
                        csvWriterFirstSpikes.writerow([firstSpikesTiming1, firstSpikesVoltage1, 'Model ' + str(i)]);

        if plotAbf:
            abfLocation = "CeliaData/L39 (8-21-2018)-20210426T193143Z-001/L39 (8-21-2018)/18821013.abf"

            abf = pyabf.ABF(abfLocation)
            sweepCShift = -5;
            isSequential = True; timeScaleFactor = 1000;
            timeShift = -49.23+5
            abf_t_template, abf_cc_template, abf_voltageTrace = [], [], []
            for sweepId in range(abf.sweepCount):
                abf.setSweep(sweepId)
                axList[0].plot(abf.sweepX + timeShift, abf.sweepY -self.junctionPotentialAdjustment -120 , color='gray', zorder=-1)
                axList[0].plot(abf.sweepX + timeShift, abf.sweepY -self.junctionPotentialAdjustment , color='gray', zorder=-1)
                axList[1].plot(abf.sweepX + timeShift, abf.sweepC + sweepCShift, color='gray', zorder=-1)
                abf_t_template.extend(timeScaleFactor * (abf.sweepX + timeShift));
                abf_cc_template.extend(abf.sweepC);
                abf_voltageTrace.extend(abf.sweepY - 10)
                if (isSequential):
                    timeShift += abf.sweepX[-1]

        axList[0].plot(0, 2, '.', color='C0', linestyle='-', zorder=-1, label='model', alpha=1)
        axList[0].plot(0, 2, '.', color='gray', linestyle='-', zorder=-1, label='data-120mV', alpha=1)
        axList[0].legend(loc='upper left')
        axList[0].plot(0, 2, '.', color='white', linestyle='-', zorder=-1, alpha=1)

        plt.figure(fig)
        plt.savefig(savePrefix+'CellTuned_CC.png', dpi=300, bbox_inches='tight')

        if(plotFICurve):
            CC_steps, maxFrequencies, sus_CC_steps, susFrequencies, firstSpikesTiming, firstSpikesVoltage = self.plotMaxFrequencies(axFICurves, abf_t_template, abf_cc_template, abf_voltageTrace, spikeAboveV=0, CCStepInterval=5000, color='gray', axFICurvesSus=axFICurvesSus)                      #axfigFirstSpikes=axfigFirstSpikes)
            if saveData:
                for CC_step, maxFrequency in zip(CC_steps, maxFrequencies):
                    csvWriterFI.writerow([CC_step, maxFrequency, 'Exp. data'])
                for sus_CC_step, susFrequency in zip(sus_CC_steps, susFrequencies):
                    csvWriterFISus.writerow([sus_CC_step, susFrequency, 'Exp. data']);
                csvFileFI.close()
                csvFileFISus.close()
                csvFileFirstSpikes.close()
            plt.figure(figFI)
            plt.savefig(savePrefix+'CellTuned_CC_maxFICurve.png', dpi=300, bbox_inches='tight')
            plt.figure(figFISus)
            plt.savefig(savePrefix+'CellTuned_CC_susFICurve.png', dpi=300, bbox_inches='tight')
            plt.figure(figFirstSpikes)
            plt.savefig(savePrefix+'CellTuned_CC_firstSpikes.png', dpi=300, bbox_inches='tight')

        if saveData:
            for tIndex in range(0, len(csvTableTimeVs[:,0]), plotSamplingRate):
                csvWriter.writerow(csvTableTimeVs[tIndex,:])
            csvFile.close()

##############################################################################################################

    # If rowsToPlot and respective heights not specified, all 10 rows are plotted with equal height of 1;
    def createAxesVC(self, figtitle='', rowsToPlot=None, rowHeights=None):
        nRows = 10;  # Voltage, currentClamp/tatalCurrentInVoltageClamp, 8 different currents
        nCols = 1;  # traces
        if rowsToPlot is None:
            rowsToPlot = range(nRows)
            rowHeights = np.ones(nRows)
        else:
            if rowHeights is None:
                rowHeights = np.ones(len(rowsToPlot))
            nRows = sum(rowHeights)

        # fig = plt.figure(figsize=(20, 4*nRows))
        fig = plt.figure(figsize=(10, 5))

        # fig.suptitle(figtitle, fontsize=16)
        gs1 = gridspec.GridSpec(nRows, nCols)
        gs1.update(wspace=0.03, hspace=0.2)  # set the spacing between axes.
        axLabels = ['V (mV)', 'I_ext (pA)', 'I_Na (pA)', 'I_K (pA)', 'I_CaL (pA)', 'I_M (pA)', 'I_KCa (pA)',
                    'I_A (pA)', 'I_h (pA)', 'I_leak (pA)']
        axList = []
        currentRowLocation = 0
        for i in range(len(rowsToPlot)):
            nextRowLocation = currentRowLocation + int(rowHeights[i])
            ax = plt.subplot(gs1[currentRowLocation:nextRowLocation, :]);
            # ax.text(0.01, 0.9, axLabels[rowsToPlot[i]], ha='left', va = 'top', transform = ax.transAxes)
            axList.append(ax)
            currentRowLocation = nextRowLocation
        axList[0].set_title(figtitle)
        axList[0].set_xticks([])
        axList[0].set_yticks([-80, -20, 50])
        axList[1].set_xlabel('Time (ms)')
        axList[1].set_ylabel('Current (pA)')
        axList[0].set_ylabel('VC step (mV)')
        axList[0].tick_params(axis='y', labelrotation=90)

        axList[1].xaxis.set_major_locator(plt.MaxNLocator(4))
        # axList[1].yaxis.set_major_locator(plt.MaxNLocator(4))
        axList[1].set_yticks([-2000, 0, 2000])
        axList[1].tick_params(axis='y', labelrotation=90)
        #axList[1].set_ylim([-3500, 2700])

        return axList, fig

    def plotTracesVC(self, y, axList, t_template, ccOrVc=True, cc_template=None, skipInitial=0, rowsToPlot=None,
                   plotOverlappingCurrents=False, plotFirstCompartmentV=False, plotSamplingRate=1):
        axId = 0
        if rowsToPlot is None or 0 in rowsToPlot:
            axList[axId].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                              y[int(skipInitial / self.stepSize):, 0][::plotSamplingRate], lw='1.5');  # Voltage traces
            # if plotFirstCompartmentV:
            # axList[axId].plot(t_template[int(skipInitial/self.stepSize):], y[int(skipInitial/self.stepSize):,24], color='k', lw=0.5); #V_nearElectrode
            axList[axId].set_xlim(t_template[int(skipInitial / self.stepSize)], t_template[
                -1])  # setting xlimits manually so it doesn't extend when plotting overlapping exp. data
            axId += 1
        totalCurrent = np.zeros(len(t_template))
        if rowsToPlot is None or 1 in rowsToPlot:
            axList[axId].set_xlim(t_template[int(skipInitial / self.stepSize)], t_template[-1])
            # total current plotted towards the end after computing the sum below
            axId += 1
        for i in range(8):
            current = y[:, 13 + i]  # currents INa, IK, ICaL, IM, IKCa, IA, Ih, Ileak
            if rowsToPlot is None or i + 2 in rowsToPlot:
                axList[axId].set_xlim(t_template[int(skipInitial / self.stepSize)], t_template[-1])
                axList[axId].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                                  current[int(skipInitial / self.stepSize):][::plotSamplingRate])
                axId += 1
            if plotOverlappingCurrents:
                axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                               current[int(skipInitial / self.stepSize):][::plotSamplingRate])
                # plotting all currents together so we can track which ones are important
            totalCurrent += current
        # totalCurrent+= y[:, 23] #IArtifact
        totalCurrent_nearElectrode = y[:, 23];
        totalCurrent = totalCurrent_nearElectrode;

        if rowsToPlot is None or 1 in rowsToPlot:
            if ccOrVc:
                axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                               cc_template[int(skipInitial / self.stepSize):][::plotSamplingRate] - self.iHold);  # current clamp
            else:
                axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                               totalCurrent[int(skipInitial / self.stepSize):][::plotSamplingRate],
                               lw=1.5)  # tatalCurrentInVoltageClamp
                if plotOverlappingCurrents:
                    axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                                   totalCurrent[int(skipInitial / self.stepSize):][::plotSamplingRate],
                                   color='#1f77b4')  # tatalCurrentInVoltageClamp
            """
            if ccOrVc:
                inputResistance = 1000/y[:, 24]  ; #MOhm
                axList[1].plot(t_template[int(skipInitial / self.stepSize):], inputResistance[int(skipInitial / self.stepSize):], color='red')  # tatalCurrentInVoltageClamp
            """
        # sp.plot(t_template, y[:,18]*0.25, color='g') #IA
        return totalCurrent

    def plotPeaks(self, multiplevClamps, t_template, currentList, currentAxis, startTime, endTime, whichStepSize, isMaxElseMin=True,
              bidirectionalSplitIndex=-1, color='g', vClampVsCurrentPeakAxis=None, vClampVsTimeDelayAxis=None, isAbf=False, plotFromIndex = 0):
        peakValues = []; peakTimings = [];
        direction1 = 1 if isMaxElseMin else -1;
        for i in range(len(currentList)):
            direction2 = 1 if (i >= bidirectionalSplitIndex) else -1;
            current = currentList[i]
            peakValue = direction1 * direction2 * max(direction1 * direction2 * current[int(startTime / whichStepSize):int(endTime / whichStepSize)])
            peakIndex = np.argmax(direction1 * direction2 * current[int(startTime / whichStepSize):int(endTime / whichStepSize)])
            peakTiming = t_template[int(startTime / whichStepSize) + peakIndex]

            peakValues.append(peakValue); peakTimings.append(peakTiming)
            if isAbf:
                currentAxis.plot(peakTimings[plotFromIndex:], peakValues[plotFromIndex:] , 'D', color=color, markersize=3)
            else:
                currentAxis.plot(peakTimings[plotFromIndex:], peakValues[plotFromIndex:], 'o', color=color, markersize=3)

        if vClampVsCurrentPeakAxis is not None:
            if isAbf:
                vClampVsCurrentPeakAxis.plot(multiplevClamps[plotFromIndex:], peakValues[plotFromIndex:], 'D', color=color, linestyle='--', zorder=-1)
            else:
                vClampVsCurrentPeakAxis.plot(multiplevClamps[plotFromIndex:], peakValues[plotFromIndex:], 'o', color=color, linestyle = '-')
        if vClampVsTimeDelayAxis is not None:
            if isAbf:
                vClampVsTimeDelayAxis.plot(multiplevClamps[plotFromIndex:], np.array(peakTimings[plotFromIndex:])- 1010, 'D', color=color, linestyle='--', zorder=-1)
            else:
                vClampVsTimeDelayAxis.plot(multiplevClamps[plotFromIndex:], np.array(peakTimings[plotFromIndex:])- 1010, 'o', color=color, linestyle='-')

    def voltageClampStep(self, multiplevClamps=np.arange(-120,10,10.), vClampStepsDurations = np.array([1000+10, 70, 130]), vClampStepsStartingValues = np.array([-80, -80, -80]), vClampStepsEndingValues = np.array([-80, -80, -80]), plotFirstCompartmentV=False, rowsToPlot=None, rowHeights=None, skipInitial = 1000, plotPeaks=False, plotAbf=False, abfLocation="", abfOffset=660, abfRisingNoOfSteps=2, plotGDestination=False, inwardStart=0.5, inwardEnd=20, inwardPlotFromIndex=10, outwaredStart=6, outwardEnd=15, maxVPeaks=0, plotSamplingRate=1, savePrefix=""):

        axList, fig = self.createAxesVC('Voltage clamp steps', rowsToPlot=rowsToPlot, rowHeights=rowHeights)
        t_template = []; totalCurrentList = [];
        t_templateAbf = []; totalCurrentListAbf = [];
        stepStartTime = vClampStepsDurations[0]; stepEndTime = stepStartTime + vClampStepsDurations[1];

        if plotPeaks:
            vClampVsCurrentPeakFig, vClampVsCurrentPeakAxis = plt.subplots(1);
            vClampVsCurrentPeakAxis.set_xlabel("VClamp (mV)"); vClampVsCurrentPeakAxis.set_ylabel("Current peak (pA)")
            vClampVsCurrentPeakAxis.set_xticks([-130,-100,-50, 0, 50, 70])
            vClampVsCurrentPeakAxis.set_xlim([-135,maxVPeaks+5])
            vClampVsCurrentPeakAxis.set_yticks([-2000,0,2000])
            #vClampVsCurrentPeakAxis.yaxis.set_major_locator(plt.MaxNLocator(4))
            vClampVsCurrentPeakAxis.tick_params(axis='y', labelrotation=90)

            vClampVsTimeDelayFig, vClampVsTimeDelayAxis = plt.subplots(1);
            vClampVsTimeDelayAxis.set_xlabel("VClamp (mV)"); vClampVsTimeDelayAxis.set_ylabel("Inward peak delay (ms)")
            vClampVsTimeDelayAxis.set_xticks([-50,0,50, 70])
            vClampVsTimeDelayAxis.set_xlim([-55,75])
            vClampVsTimeDelayAxis.yaxis.set_major_locator(plt.MaxNLocator(4))

        if plotAbf:
            abf = pyabf.ABF(abfLocation)
            print("Loaded Abf with sweepCount:", abf.sweepCount)
            assert abf.sweepCount == len(multiplevClamps)
            abfSweepXUpdated = abf.sweepX * 1000 + abfOffset   #For VC steps - adjusting data vs model timing
            #abfOffset1 = abfOffset - (abfRisingNoOfSteps+1)*self.stepSizeAbf + (self.stepSize - self.stepSizeAbf)
            abfOffset1 = abfOffset  # With electrode rising model
            t_templateAbf = abf.sweepX * 1000 + abfOffset1  ##For VC current output - adjusting data vs model timing
                            #The offset is to be applied only while plotting abf alongside model and not to be used during computations of peaks etc. since the current data array is not affected by the offsetting time axis

        for sweepId in range(len(multiplevClamps)):
            vClamp = multiplevClamps[sweepId]
            vClampStepsStartingValues[1]=vClamp; vClampStepsEndingValues[1]=vClamp;
            vc_template, t_template, t_len = self.computeVoltageClampSteps(vClampStepsDurations, vClampStepsStartingValues, vClampStepsEndingValues)
            y = ode_solver.update(tspn_Clamp.step, self.initialValues(), t_len, vc_template, self.G, self.stepSize, ccOrVc = False)
            totalCurrentList.append(self.plotTracesVC(y, axList, t_template, ccOrVc=False, skipInitial=skipInitial, rowsToPlot=rowsToPlot, plotFirstCompartmentV=plotFirstCompartmentV, plotSamplingRate=plotSamplingRate))
            if plotGDestination:
                        # For VC, only plotting total current (in black) for GDestination for comparision of inward/outward peaks
                y1 = ode_solver.update(tspn_Clamp.step, self.initialValues(), t_len, vc_template, self.G_Destination, self.stepSize, ccOrVc=False)
                axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                               y1[int(skipInitial / self.stepSize):, 23][::plotSamplingRate],
                               lw=1., color='tab:brown', marker=',', zorder=-2)  # tatalCurrentInVoltageClamp

            if plotAbf:
                #pyabf.filter.gaussian(abf, 0.1)
                abf.setSweep(sweepId)
                axList[0].plot(abfSweepXUpdated, abf.sweepC - 80 -self.junctionPotentialAdjustment, color='gray', lw=0.5, zorder=-1)
                axList[1].plot(t_templateAbf , abf.sweepY, color='gray', lw=0.5, zorder=-1)  #There seems to be one timeStep ms delay between current output and VC step?
                totalCurrentListAbf.append(abf.sweepY)

        if plotPeaks:

            # Plot peak artifacts
            #self.plotPeaks(multiplevClamps, t_template, totalCurrentList, axList[1], stepStartTime-0.1, stepStartTime+1., self.stepSize, isMaxElseMin=True, bidirectionalSplitIndex=5, color='g', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis)
            #if plotAbf:
                #self.plotPeaks(multiplevClamps, t_templateAbf, totalCurrentListAbf, axList[1], stepStartTime-0.1-abfOffset1, stepStartTime+1.-abfOffset1, self.stepSizeAbf, isMaxElseMin=True, bidirectionalSplitIndex=5, color='lime', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis, isAbf=True)
                #print(t_template[int(stepStartTime/self.stepSize):int((stepStartTime+1)/self.stepSize)], t_templateAbf[int(stepStartTime/self.stepSizeAbf):int((stepStartTime+1)/self.stepSizeAbf)])

            # Plot peak artifacts discharging
            #self.plotPeaks(multiplevClamps, t_template, totalCurrentList, axList[1], stepEndTime-0.1, stepEndTime+1., self.stepSize, isMaxElseMin=False, bidirectionalSplitIndex=5, color='darkgoldenrod', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis)
            #if plotAbf:
                #self.plotPeaks(multiplevClamps, t_templateAbf, totalCurrentListAbf, axList[1], stepEndTime-0.1-abfOffset1, stepEndTime+1.-abfOffset1, self.stepSizeAbf, isMaxElseMin=False, bidirectionalSplitIndex=5, color='goldenrod', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis, isAbf=True)


            # Inward current peaks
            self.plotPeaks(multiplevClamps, t_template, totalCurrentList, axList[1], stepStartTime+inwardStart, stepStartTime+inwardEnd, self.stepSize, isMaxElseMin=False, bidirectionalSplitIndex=-1, color='brown', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis, vClampVsTimeDelayAxis=vClampVsTimeDelayAxis, plotFromIndex=inwardPlotFromIndex)
            if plotAbf:
                self.plotPeaks(multiplevClamps, t_templateAbf, totalCurrentListAbf, axList[1], stepStartTime+inwardStart-abfOffset1, stepStartTime+inwardEnd-abfOffset1, self.stepSizeAbf, isMaxElseMin=False, bidirectionalSplitIndex=-1, color='indianred', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis, vClampVsTimeDelayAxis=vClampVsTimeDelayAxis, isAbf=True, plotFromIndex=inwardPlotFromIndex)

            # Outward current peaks
            self.plotPeaks(multiplevClamps, t_template, totalCurrentList, axList[1], stepStartTime+outwaredStart, stepStartTime+outwardEnd, self.stepSize, isMaxElseMin=True, bidirectionalSplitIndex=-1, color='purple', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis)
            if plotAbf:
                self.plotPeaks(multiplevClamps, t_templateAbf, totalCurrentListAbf, axList[1], stepStartTime+outwaredStart-abfOffset1, stepStartTime+outwardEnd-abfOffset1, self.stepSizeAbf, isMaxElseMin=True, bidirectionalSplitIndex=-1, color='m', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis, isAbf=True)

            # Long lasting current
            self.plotPeaks(multiplevClamps, t_template, totalCurrentList, axList[1], stepEndTime-1., stepEndTime-1.+2*self.stepSize, self.stepSize, isMaxElseMin=True, bidirectionalSplitIndex=-1, color='dodgerblue', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis)
            if plotAbf:
                self.plotPeaks(multiplevClamps, t_templateAbf, totalCurrentListAbf, axList[1], stepEndTime-1.-abfOffset1, stepEndTime-1.+2*self.stepSizeAbf-abfOffset1, self.stepSizeAbf, isMaxElseMin=True, bidirectionalSplitIndex=-1, color='deepskyblue', vClampVsCurrentPeakAxis=vClampVsCurrentPeakAxis, isAbf=True)

        if plotPeaks:
            vClampVsCurrentPeakAxis.plot(-50, 0, 'o', color='g', linestyle='-', zorder=-1, label='Electrode charging (model)', alpha=1)
            vClampVsCurrentPeakAxis.plot(-50, 0, 'D', color='lime', linestyle='--', zorder=-1, label='Electrode charging (data)', alpha=1)
            #vClampVsCurrentPeakAxis.plot(-50, 0, 'o', color='darkgoldenrod', linestyle='-', zorder=-1, label='Electrode discharging (model)', alpha=1)
            #vClampVsCurrentPeakAxis.plot(-50, 0, 'D', color='goldenrod', linestyle='--', zorder=-1, label='Electrode discharging (data)', alpha=1)
            vClampVsCurrentPeakAxis.plot(-50, 0, 'o', color='brown', linestyle='-', zorder=-1, label='Inward Na+ (model)', alpha=1)
            vClampVsCurrentPeakAxis.plot(-50, 0, 'D', color='indianred', linestyle='--', zorder=-1, label='Inward Na+ (data)', alpha=1)

            vClampVsCurrentPeakAxis.plot(-50, 0, 'o', color='purple', linestyle='-', zorder=-1, label='Outward A-type K+ (model)', alpha=1)
            vClampVsCurrentPeakAxis.plot(-50, 0, 'D', color='m', linestyle='--', zorder=-1, label='Outward A-type K+ (data)', alpha=1)

            vClampVsCurrentPeakAxis.plot(-50,0, 'o', color='dodgerblue', linestyle='-', zorder=-1, label='Long lasting current (model)', alpha=1)
            vClampVsCurrentPeakAxis.plot(-50, 0, 'D', color='deepskyblue', linestyle='-', zorder=-1, label='Long lasting current (data)', alpha=1)

            #vClampVsCurrentPeakAxis.legend(loc='upper left')
            vClampVsCurrentPeakAxis.plot(-50, 0, 'D', color='white', linestyle='--', zorder=-1, label='Electrode artifact (data)', alpha=1)

            #vClampVsTimeDelayAxis.legend(loc='upper right')
            vClampVsTimeDelayAxis.plot(0, 2, 'D', color='white', linestyle='-', zorder=-1, alpha=1)

            vClampVsCurrentPeakFig.savefig(savePrefix+'CellTuned_NaVC-Peaks.png', dpi=300, bbox_inches='tight')
            vClampVsTimeDelayFig.savefig(savePrefix+'CellTuned_NaVC-NaDelays.png', dpi=300, bbox_inches='tight')

        # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#sphx-glr-tutorials-intermediate-legend-guide-py
        gray_line = mlines.Line2D([], [], color='gray', marker='.',
                                  markersize=3, label='data')
        blue_line = mlines.Line2D([], [], color='blue', marker='.',
                                  markersize=3, label='model (all colors)')
        axList[1].legend(handles=[gray_line, blue_line], loc='upper right')
        fig.savefig(savePrefix+'CellTuned_NaVC-Traces.png', dpi=300, bbox_inches='tight')

##############################################################################################################

        #If rowsToPlot and respective heights not specified, all 10 rows are plotted with equal height of 1;
    def createAxes_VR(self, figtitle='', rowsToPlot=None, rowHeights=None):
        nRows = 10;  # Voltage, currentClamp/tatalCurrentInVoltageClamp, 8 different currents
        nCols = 1;  # traces
        if rowsToPlot is None:
            rowsToPlot=range(nRows)
            rowHeights = np.ones(nRows)
        else:
            if rowHeights is None:
                rowHeights = np.ones(len(rowsToPlot))
            nRows = sum(rowHeights)

        # fig = plt.figure(figsize=(20, 4*nRows))
        fig = plt.figure(figsize=(10, 10))

        #fig.suptitle(figtitle, fontsize=16)
        gs1 = gridspec.GridSpec(nRows, nCols)
        gs1.update(wspace=0.03, hspace=0.2)  # set the spacing between axes.
        axLabels = ['VC ramp (mV)', 'Current (pA)', 'I_Na', 'I_K', 'I_CaL', 'I_M', 'I_KCa', 'I_A', 'I_h', 'I_leak' ]
        axList = []
        currentRowLocation = 0
        for i in range(len(rowsToPlot)):
            nextRowLocation = currentRowLocation+int(rowHeights[i])
            ax = plt.subplot(gs1[currentRowLocation:nextRowLocation, :]);
            ax.set_ylabel(axLabels[rowsToPlot[i]])
            #ax.text(0.01, 0.9, axLabels[rowsToPlot[i]], ha='left', va = 'top', transform = ax.transAxes)
            axList.append(ax)
            currentRowLocation=nextRowLocation
        axList[0].set_title(figtitle)
        axList[0].set_xticks([])
        axList[0].set_yticks([-80, -20, 50])
        axList[1].set_xlabel('Time (ms)')
        #axList[1].set_ylabel('Current (pA)')
        #axList[0].set_ylabel('VC step (mV)')
        axList[0].tick_params(axis='y', labelrotation=90)

        axList[1].xaxis.set_major_locator(plt.MaxNLocator(4))
        axList[1].yaxis.set_major_locator(plt.MaxNLocator(4))
        axList[1].tick_params(axis='y', labelrotation=90)

        return axList, fig

    def plotTraces_VR(self, y, axList, t_template, ccOrVc=True, cc_template=None, skipInitial=0, rowsToPlot=None, plotOverlappingCurrents=False, plotFirstCompartmentV=False, plotSamplingRate=1):
        axId = 0
        if rowsToPlot is None or 0 in rowsToPlot:
            axList[axId].plot(t_template[int(skipInitial/self.stepSize):][::plotSamplingRate], y[int(skipInitial/self.stepSize):,0][::plotSamplingRate], lw='1.5'); #Voltage traces
            #if plotFirstCompartmentV:
                #axList[axId].plot(t_template[int(skipInitial/self.stepSize):], y[int(skipInitial/self.stepSize):,24], color='k', lw=0.5); #V_nearElectrode
            axList[axId].set_xlim(t_template[int(skipInitial/self.stepSize)], t_template[-1])   #setting xlimits manually so it doesn't extend when plotting overlapping exp. data
            axId+=1
        totalCurrent=np.zeros(len(t_template))
        if rowsToPlot is None or 1 in rowsToPlot:
            axList[axId].set_xlim(t_template[int(skipInitial / self.stepSize)], t_template[-1])
            #total current plotted towards the end after computing the sum below
            axId+=1
        for i in range(8):
            current = y[:, 13 + i] # currents INa, IK, ICaL, IM, IKCa, IA, Ih, Ileak
            if rowsToPlot is None or i+2 in rowsToPlot:
                axList[axId].set_xlim(t_template[int(skipInitial / self.stepSize)], t_template[-1])
                axList[axId].plot(t_template[int(skipInitial/self.stepSize):][::plotSamplingRate], current[int(skipInitial/self.stepSize):][::plotSamplingRate])
                axId += 1
            if plotOverlappingCurrents:
                    axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate], current[int(skipInitial / self.stepSize):][::plotSamplingRate])
                        #plotting all currents together so we can track which ones are important
            totalCurrent+= current
        #totalCurrent+= y[:, 23] #IArtifact
        totalCurrent_nearElectrode = y[:, 23]; totalCurrent = totalCurrent_nearElectrode;

        if rowsToPlot is None or 1 in rowsToPlot:
            if ccOrVc:
                axList[1].plot(t_template[int(skipInitial/self.stepSize):][::plotSamplingRate], cc_template[int(skipInitial/self.stepSize):][::plotSamplingRate]-self.iHold); # current clamp
            else:
                axList[1].plot(t_template[int(skipInitial/self.stepSize):][::plotSamplingRate], totalCurrent[int(skipInitial/self.stepSize):][::plotSamplingRate], lw=1.5) # tatalCurrentInVoltageClamp
                if plotOverlappingCurrents:
                    axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate], totalCurrent[int(skipInitial / self.stepSize):][::plotSamplingRate], color='#1f77b4')  # tatalCurrentInVoltageClamp
            """
            if ccOrVc:
                inputResistance = 1000/y[:, 24]  ; #MOhm
                axList[1].plot(t_template[int(skipInitial / self.stepSize):], inputResistance[int(skipInitial / self.stepSize):], color='red')  # tatalCurrentInVoltageClamp
            """
        #sp.plot(t_template, y[:,18]*0.25, color='g') #IA
        return totalCurrent

    def voltageClampRamp(self, vClampStepsDurations=np.array([2000, 1000, 4000, 4000, 2000]),
                         vClampStepsStartingValues=np.array([-80, -80, -90, 10, -80]),
                         vClampStepsEndingValues=np.array([-80, -90, 10, -80, -80]), rowsToPlot=None, rowHeights=None, plotFirstCompartmentV=False, plotOverlappingCurrents=False, plotAbf=False, abfLocation="", abfOffset=660, plotGDestination=False, plotSamplingRate=1, savePrefix=""):
        vc_template, t_template, t_len = self.computeVoltageClampSteps(vClampStepsDurations,
                                                                       vClampStepsStartingValues,
                                                                       vClampStepsEndingValues)
        axList, fig = self.createAxes_VR('Voltage ramp', rowsToPlot=rowsToPlot, rowHeights=rowHeights)
        y = ode_solver.update(tspn_Clamp.step, self.initialValues(), t_len, vc_template, self.G, self.stepSize, ccOrVc=False)
        self.plotTraces_VR(y, axList, t_template, rowsToPlot=rowsToPlot, ccOrVc=False, skipInitial=1000, plotOverlappingCurrents=plotOverlappingCurrents, plotFirstCompartmentV=plotFirstCompartmentV, plotSamplingRate=plotSamplingRate)

        if plotGDestination:
            y1 = ode_solver.update(tspn_Clamp.step, self.initialValues(), t_len, vc_template, self.G_Destination, self.stepSize,
                                  ccOrVc=False)
            skipInitial=1000
            axList[1].plot(t_template[int(skipInitial / self.stepSize):][::plotSamplingRate],
                           y1[int(skipInitial / self.stepSize):, 23][::plotSamplingRate], lw=0.5, color='tab:brown')
        if plotAbf:
            # VRamp superimpose plot Abf data with model
            abf = pyabf.ABF(abfLocation)
            print("No. of sweeps: ", abf.sweepCount)
            # for sweepId in range(abf.sweepCount):
            for sweepId in range(1):
                abf.setSweep(sweepId)
                abf.protocol
                axList[0].plot(abf.sweepX * 1000 + abfOffset, abf.sweepC - 80 -self.junctionPotentialAdjustment + 5, color='gray', lw=0.5) #-80 to adjust recording offset for liquid junction potential; +5 to separate from model while displaying
                axList[1].plot(abf.sweepX * 1000 + abfOffset, abf.sweepY, color='gray', lw=1.0)

        axList[1].plot(1000, 2, '.', color='C0', linestyle='-', zorder=-1, label='model', alpha=1)
        axList[1].plot(1000, 2, '.', color='gray', linestyle='-', zorder=-1, label='data-120mV', alpha=1)
        axList[1].legend(loc='upper left')
        axList[1].plot(1000, 2, '.', color='white', linestyle='-', zorder=-1, alpha=1)

        plt.savefig(savePrefix+'ellCTuned_VR.png', dpi=300, bbox_inches='tight')
