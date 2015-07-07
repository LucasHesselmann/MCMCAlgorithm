#! /usr/bin/env python3.4

import sys
import os
import numpy
import pylab
import random
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from matplotlib.colors import LogNorm


def multiDimDiagramm(string, numberOfSamples, directoryName, resultName):
  """
  Plot the convergence time for several dimensions.
  
  @param: string - 'RWM' or 'MALA' selects the algoritm type
  @param: numberOfSamples - The number of samples which  are generated
  @param: directoryName - Name of the directory for the analysis files generated for each dimension
  @param: resultName - Name of the diagramm showing the convergence time

  Example:  MCMC.multiDimDiagramm('RWM', 100000, 'Test01', 'ConvergenceTimeDiagramm')
  
  """
  result=[]
  acceptRate=[]
  autocor=[]
  colours=['r','g','b','k','y','c','m']
  
  # How many different dimensions? in [1, 6] #
  d = 7
  #------------------------------------------#

  k=0

  # Some preparation
  directory = '{0}_{1}_{2}'.format(directoryName, string, numberOfSamples)
  os.makedirs(directory) 
  
  pylab.figure()
  # Make diagramm for several dimensions
  #dimensions=[5, 10, 20, 30, 50, 100]
  dimensions=[1,2,3,4,5,8,10]
  if string in ['MALA']:
    x = 0.13
    y = 0.92
  else:
    x = 0.05
    y = 0.82
  xp = numpy.linspace(x, y, 100)
  p = []
  for dim in range(d):
    result=makeDiagramm(string, dimensions[dim], numberOfSamples, directoryName, resultName)
    acceptRate.append(result[0])
    autocor.append(result[1])
    p.append( numpy.poly1d(numpy.polyfit(acceptRate[dim], autocor[dim], 4)) )
    pylab.plot(acceptRate[dim], autocor[dim], 'm.', xp, p[dim](xp), '--', color=colours[dim], label='Convergence time')
    #pylab.plot(result[0], result[1], colours[k], label='Convergence time')
    pylab.ylabel('convergence time')
    pylab.xlabel('acceptance rate')
    pylab.grid(True)
    fileName1=os.path.join(directory, '{0}_{1}_Dim{2}_fit.png'.format(string, resultName, dimensions[dim]))
    pylab.savefig(fileName1) 
    pylab.clf()
    pylab.figure()
    pylab.plot(acceptRate[dim], autocor[dim], 'm.', color=colours[dim], label='Convergence time')
    #pylab.plot(result[0], result[1], colours[k], label='Convergence time')
    pylab.ylabel('convergence time')
    pylab.xlabel('acceptance rate')
    pylab.grid(True)
    fileName1=os.path.join(directory, '{0}_{1}_Dim{2}.png'.format(string, resultName, dimensions[dim]))
    pylab.savefig(fileName1) 
    pylab.clf()
    k+=1
  pylab.figure()
  for dim in range(d):    
    pylab.plot( xp, p[dim](xp), colours[dim], label=dimensions[dim] )
  #pylab.plot(acceptRate[0], autocor[0], colours[0], acceptRate[1], autocor[1], colours[1], acceptRate[2], autocor[2], colours[2], acceptRate[3], autocor[3], colours[3], label='Convergence time')
  pylab.ylabel('scaled autocorrelation time')
  pylab.xlabel('acceptance rate')
  pylab.legend()
  #pylab.ylim([10.0, 50.0])
  pylab.grid(True)
  fileName1=os.path.join(directory, '{0}_{1}.png'.format(string, resultName))
  pylab.savefig(fileName1) 
  pylab.clf()
  # Second plot
  pylab.figure()
  for dim in range(d):    
    pylab.plot( acceptRate[dim], autocor[dim], 'o', color=colours[dim], label=dimensions[dim] )
  pylab.ylabel('scaled autocorrelation time')
  pylab.xlabel('acceptance rate')
  pylab.legend()
  #pylab.ylim([10.0, 50.0])
  pylab.grid(True)
  newPrintName = fileName1.replace(".png", "_2.png")
  pylab.savefig(newPrintName) 
  pylab.clf()
  # Other markers
  pylab.figure()
  for dim in range(d):    
    pylab.plot( acceptRate[dim], autocor[dim], '>', color=colours[dim], label=dimensions[dim] )
  pylab.ylabel('scaled autocorrelation time')
  pylab.xlabel('acceptance rate')
  pylab.legend()
  #pylab.ylim([10.0, 50.0])
  pylab.grid(True)
  newPrintName = fileName1.replace(".png", "_3.png")
  pylab.savefig(newPrintName) 
  # Other markers
  pylab.figure()
  for dim in range(d):    
    pylab.plot( acceptRate[dim], autocor[dim], '.', color=colours[dim], label=dimensions[dim] )
  pylab.ylabel('scaled autocorrelation time')
  pylab.xlabel('acceptance rate')
  pylab.legend()
  #pylab.ylim([10.0, 50.0])
  pylab.grid(True)
  newPrintName = fileName1.replace(".png", "_4.png")
  pylab.savefig(newPrintName) 

  pylab.close('all')
  
  return 'Done'
                          

 
def makeDiagramm(string, dimension, numberOfSamples, directoryName, resultName):
  """
  Make a diagramm to show behavior of the algo depending on the scaling of the variance.
  
  @param: string - 'RWM' or 'MALA' selects the algoritm type
  @param: dimension - Dimension of the Markov chain
  @param: numberOfSamples - The number of samples which  are generated
  @param: directoryName - Name of the directory for the analysis files generated for each dimension
  @param: resultName - Name of the diagramm showing the convergence time

  Example:  MCMC.makeDiagramm('RWM', 5, 100000, 'Test01', 'ConvergenceTimeDiagramm')
  
  """
  result=[]
  autocor=[]
  acceptRate=[]
  steps=[]
  init=[]
  variances=[]
  # Number of repititions of simulations for each variance, we take the median of the acceptance rates and autocorrelation times. 
  repitition = 1
  tmpVec = numpy.array([0.0 for i in range(repitition)])
  tmpVec2 = numpy.array([0.0 for i in range(repitition)])
  k=1
  dimensionIter=range(dimension)
  # Simulations with a higher value as autocorrelation time are neglected.
  threshold = 15
  # Set order of convergence depending of algorithm type
  if string in ['MALA']:
    convOrder=1.0/3.0
  else:
    convOrder=1.0

  # Some preparation
  directory = '{0}_{1}_{2}_{3}'.format(directoryName, string, dimension, numberOfSamples)
  os.makedirs(directory) 
  # Initialise the corresponding algorithm with a burn-in of 100 steps (choose appropriate init value)
  algo = Algo(string, dimension, 100)
  # Initialize the init value
  init.append(random.gauss(0.0, 0.1))
  for dim in dimensionIter[1:]:
    init.append(random.gauss(-0.0, 0.2))
  # Generate the proposal variances and scale them according to the dimension
  if string in ['MALA']:
    if dimension == 1:
      helper=numpy.logspace(-0.2, 4.4, 26, True, 2.0) #MALA in 1D
    elif dimension == 2:
      helper=numpy.logspace(-0.2, 4.4, 26, True, 2.0) #MALA in 2D
    else:
      helper=numpy.logspace(0.2, 2.4, 25, True, 2.0) #MALA in ND
      # For changed Gaussian
      #helper=numpy.logspace(-5.0, 2.0, 32, True, 2.0) #MALA in ND
  elif string in ['RWM']:
    if dimension == 1:
      #helper=[0.08, 8.0, 160]
      helper=numpy.logspace(-0.8, 13, 26, True, 2.0) #RWM in 1D
    elif dimension == 2:
      helper=numpy.logspace(-0.8, 8, 26, True, 2.0) #RWM in 2D
    else:
      helper=numpy.logspace(-0.9, 4.5, 25, True, 2.0) #RWM in ND
      #For changed Gaussian
      #helper=numpy.logspace(-8.0, 3.5, 32, True, 2.0) #RWM in ND
  for var in helper:
    variances.append(var / dimension**(convOrder))
  # Simulate for each variance in variances.
  for var in variances:
    for i in range(repitition):
      result=algo.simulation(numberOfSamples, var, False, True, init)
      tmp = format(result[0], '.2f')
      if ((string in ['RWM'] and (result[0]>0.85 or result[0]<0.04)) or (string in ['MALA'] and (result[0]<0.14 or result[0]>0.95))):#0.98 0.15
        tmpVec2[i]=tmp
        tmpVec[i]=threshold+1
        continue
      fileName=os.path.join(directory, '{0}_{1}-{2}_{3}.png'.format(string, k, i, tmp))
      # For analysis: use sample mean or real mean=0 ---- Last two entries for graphics [need simple analysis of samples!]
      tmp2=algo.analyseData(result[1], [0.0], result[3], fileName, result[4], result[5])  
      tmpVec[i] = (tmp2 / dimension**(convOrder))
      tmpVec2[i] = (tmp)
      print('-----------------------------------------------------------')
      print('Round {0}'.format(i))
      print('Acceptance rate: {0}'.format(tmp))
      print('Integrated autocorrelation: {0}'.format(tmp2 / dimension**(convOrder)))
      print('-----------------------------------------------------------')
    # Take the median of acceptance rate and convergence time
    tmp2=numpy.median(tmpVec2)
    if tmp2 > 0.98:
      continue
    tmp=numpy.median(tmpVec)
    if tmp < threshold:
      acceptRate.append(tmp2)
      autocor.append(tmp)
      print('-----------------------------------------------------------')
      print('The median of loop {0}'.format(k))
      print('Acceptance rate: {0}'.format(tmp2))
      print('Integrated autocorrelation: {0}'.format(tmp))
      print('-----------------------------------------------------------')
      k+=1
    if string in ['MALA']:
      if tmp2<0.15:
        break
    if tmp2<0.05:
      break
    pylab.figure()
    pylab.plot(acceptRate, autocor, 'ro', label='Autocorrelation time')
    pylab.ylabel('scaled autocorrelation time')
    pylab.xlabel('acceptance rate')
    pylab.grid(True)
    fileName1=os.path.join(directory, '{0}_{1}_Dim{2}.png'.format(string, resultName, dimension))
    pylab.savefig(fileName1) 
    pylab.close('all')
  returnValue=[]
  returnValue.append(acceptRate)
  returnValue.append(autocor)
  return returnValue

def calculateSEM(string, dimension, numberOfSamples, directoryName, resultName):
  """
  Calculate the Standard Error of the Mean for an example.
  
  @param: string - 'RWM' or 'MALA' selects the algoritm type
  @param: dimension - Dimension of the Markov chain
  @param: numberOfSamples - The number of samples which  are generated
  @param: directoryName - Name of the directory for the analysis files generated for each dimension
  @param: resultName - Name of the diagramm showing the convergence time

  Example:  MCMC.calculateSEM('RWM', 5, 100000, 'Test01', 'ConvergenceTimeDiagramm')
  
  """
  result=[]
  rsme = 0.0
  steps=[]
  init=[]
  variances=[]
  # Number of repititions of simulations for each variance, we take the median of the acceptance rates and autocorrelation times. 
  repitition = 8
  tmpVec = numpy.array([0.0 for i in range(repitition)])
  tmpVec2 = numpy.array([0.0 for i in range(repitition)])
  k=1
  dimensionIter=range(dimension)
  # Simulations with a higher value as autocorrelation time are neglected.
  threshold = 400
  # Set order of convergence depending of algorithm type
  if string in ['MALA']:
    convOrder=1.0/3.0
  else:
    convOrder=1.0

  # Some preparation
  directory = '{0}_{1}_{2}_{3}'.format(directoryName, string, dimension, numberOfSamples)
  os.makedirs(directory) 
  # Initialise the corresponding algorithm with a burn-in of 100 steps (choose appropriate init value)
  algo = Algo(string, dimension, 100)
  # Initialize the init value
  init.append(random.gauss(0.0, 0.1))
  for dim in dimensionIter[1:]:
    init.append(random.gauss(-0.0, 0.1))
  # Generate the proposal variances and scale them according to the dimension
  if string in ['MALA']:
    # For changed Gaussian in dim=20
    helper=[0.201]
  elif string in ['RWM']:
    #For changed Gaussian in dim=20
    helper=[0.546]
  for var in helper:
    variances.append(var / dimension**(convOrder))
  # Simulate for each variance in variances.
  for var in variances:
    for i in range(repitition):
      result=algo.simulation(numberOfSamples, var, False, True, init)
      tmp = format(result[0], '.2f')
      if ((string in ['RWM'] and (result[0]>0.85 or result[0]<0.04)) or (string in ['MALA'] and (result[0]<0.14 or result[0]>0.95))):#0.98 0.15
        tmpVec2[i]=tmp
        tmpVec[i]=threshold+1
        continue
      fileName=os.path.join(directory, '{0}_{1}-{2}_{3}.png'.format(string, k, i, tmp))
      # For analysis: use sample mean or real mean=0 ---- Last two entries for graphics [need simple analysis of samples!]
      tmp2=algo.analyseData(result[1], [0.0], result[3], fileName, result[4], result[5])  
      tmpVec[i] = (tmp2)
      tmpVec2[i] = (tmp)
      print('-----------------------------------------------------------')
      print('Round {0}'.format(i))
      print('Acceptance rate: {0}'.format(tmp))
      print('Standard Error of the Mean: {0}'.format(tmp2 ))
      print('-----------------------------------------------------------')
    # Take the mean of acceptance rate and convergence time
    tmp2=numpy.mean(tmpVec2, dtype=numpy.float64)
    tmp=numpy.mean(tmpVec, dtype=numpy.float64)
    # Calculate root mean square error
    for i in range(repitition):
      tmpVec[i] += -tmp
    rsme = numpy.sqrt(numpy.mean(numpy.square(tmpVec)))
    print('-----------------------------------------------------------')
    print('The median of loop {0}'.format(k))
    print('Acceptance rate: {0}'.format(tmp2))
    print('Standard Error of the Mean: {0}'.format(tmp))
    print('Root Mean Squared Error: {0}'.format(rsme))
    print('-----------------------------------------------------------')
    k+=1
    if string in ['MALA']:
      if tmp2<0.15:
        break
    if tmp2<0.05:
      break
  return rsme                        
  
class Algo:
  """
  This class generates a MCMC method (RWM or MALA) and generates a sample of random variables according to the target distribution.
  """
  
  def __init__(self, Algo, dimension, BurnIn=None):
    # Set type of algorithm: RWM and MALA
    if Algo in ['RWM', 'MALA']:
      self.setAlgoType( Algo )
    else:
      raise RuntimeError('Only "RWM" and "MALA" are supported')
    # Set BurnIn
    self.setBurnIn( BurnIn )
    # Set dimension
    self.setDimension( dimension )

      
  def simulation(self, numberOfSamples, variance, analyticGradient=False, analyseFlag=True, initialPosition=[]):
    """
    Main simulation.
    """
    dimension = int(self._dimension)
    initialPosition
    algoType = str(self._algoType)
    counter = 0
    warmUp = 0
    samples = [[] for i in range(dimension) ]
    acceptRate = 0.
    acceptCounter = 0
    # BurnIn flag
    flag = False
    #Acceptance flag
    acceptance = False
    # Print the sample mean and sample covariance
    printMean=False
    # Set method to calculate the gradient for MALA
    if analyticGradient:
      #gradientMethod=self.evaluateGradientOfMultimodalGaussian
      gradientMethod=self.evaluateGradientOfGaussian
    else:
      gradientMethod=self.calculateGradient
    #  ---------- HERE YOU HAVE TO CHOOSE ----------
    # Set target distribution 
    #targetDistribution=self.evaluateGaussian
    #targetDistribution=self.evaluateMultimodalGaussian
    targetDistribution=self.evaluateChangedGaussian
    # ------------------- END ----------------------
    # Temporary samples
    x = []
    # Proposals
    y = numpy.zeros(dimension+1)
    # Mean of the proposals
    mean = [0.0 for i in range(dimension)]
    # Flag for a simple analysis of only one dimension and the prefered dimension
    analyseDim=0
    simpleAnalysis=True
    # Some helpers
    tmp=[]
    sampleMean=[0.0 for i in range(dimension)]
    covarianceMatrix=[[0.0 for i in range(dimension)] for i in range(dimension)]
    if simpleAnalysis is False:
      covarianceMatrixSum=[[0.0 for i in range(dimension)] for i in range(dimension)]
      sampleMeanSum=[0.0 for i in range(dimension)]
    else:
      covarianceMatrixSum=0.0
      sampleMeanSum=0.0
      allSampleMean = []
      allCovariance = []
    temp=0.0
    temp2=0.0
    temp3=0.0
       
    # Check dimensions and set initial sample
    if initialPosition and len(initialPosition) == dimension :
      #print('Start simulation with given initial value: {0}'.format(initialPosition))
      for dim in range(dimension):
        tmp = initialPosition[dim]
        x.append( tmp )
      # Last entry is for acceptance flag
      x.append(False)
    # If not initialize, set all entries to zero
    elif not initialPosition:
      #print('Start simulation with initial value zero')
      for dim in range(dimension):
        x.append(0.0)
      # Last entry is for acceptance flag
      x.append(False)
    else:
      raise RuntimeError('Dimension of initial value do not correspond to dimension of the MCMC method')
    
    # Repeat generating new samples
    print('Generate a sample of size: {0} and dimension: {1} with MCMC-type: {2}'.format(numberOfSamples, dimension, algoType))
    while counter < numberOfSamples+1:
      # Calculate the mean of your proposal
      if algoType in ['RWM']:
        for dim in range(dimension):
          mean[dim] = x[dim]
      elif algoType in ['MALA']:
        grad = gradientMethod(targetDistribution, x )
        for dim in range(dimension):
          mean[dim] = x[dim] + 0.5*variance*grad[dim]
        
      # Generate the proposal
      for dim in range(dimension):
        y[dim]=  self.generateGaussian(mean[dim], variance)
          
          
      # Accept or reject
      tmp = self.acceptanceStep(targetDistribution, gradientMethod, y, x, mean, variance)
      for dim in range(dimension):
        x[dim] = tmp[dim]
      acceptance=tmp[dimension]
      
      # Count steps for the burn-in
      if flag is False:
        warmUp += 1
        #print(warmUp)
      if warmUp == self._burnIn:
        flag = True
      # Reaching the burn-in, we start the counter and sample
      if flag:
        counter += 1
        #print(counter, end='\r')
          
      # Calculate acceptance rate
      if acceptance and flag:
        acceptCounter += 1
      if flag:
        acceptRate = float(acceptCounter) / float(counter)

      # Sample only  after the burn-in
      if flag:
        for dim in range(dimension):
          samples[dim].append( x[dim] )

      percentage = format(100*counter/numberOfSamples, '.1f')
      print('Processing: {0}%'.format(percentage), end='\r')

      # Calculation of sample mean and sample variance of all steps and in addition the mean and covariance for each iteration to plot them at the end
      if analyseFlag is True and counter >= 1:
        if simpleAnalysis is False:
          for dim in range(dimension):
            # Add the new coordinate to the existent sum 
            sampleMeanSum[dim] += x[dim]
            # Divide by the number of added samples
            if counter > 1:
              sampleMean[dim] = sampleMeanSum[dim] / (counter-1)
            elif counter == 1:
              sampleMean[dim] = sampleMeanSum[dim]
        else:
          sampleMeanSum += x[analyseDim]
          if counter>1:
            sampleMean[analyseDim] = sampleMeanSum / (counter -1)
            allSampleMean.append(sampleMean[analyseDim])
          else:
            sampleMean[analyseDim] = sampleMeanSum
            allSampleMean.append(sampleMean[analyseDim])
        if simpleAnalysis is False:
          # Use symmetry of covariance matrix (upper triangular matrix)
          for dim1 in range(dimension):
            for dim2 in range(dim1, dimension):
              # sampled covariance matrix
              covarianceMatrixSum[dim1][dim2]+=( x[dim1]-sampleMean[dim1] )*( x[dim2]-sampleMean[dim2] ) 
              # Divide by (numberOfSamples-1) for an unbiased estimate.
              if counter > 1:
                covarianceMatrix[dim1][dim2] = covarianceMatrixSum[dim1][dim2] / (counter -1)
              elif counter == 1:
                covarianceMatrix[dim1][dim2] = covarianceMatrixSum[dim1][dim2]
        else:
          covarianceMatrixSum += (x[analyseDim]- 0.0)**2#sampleMean[analyseDim])**2
          if counter>1:
            covarianceMatrix[0][0] = (counter-1)**-1 * covarianceMatrixSum
            allCovariance.append(covarianceMatrix[0][0])
          else:
            covarianceMatrix[0][0] = covarianceMatrixSum
            allCovariance.append(covarianceMatrix[0][0])
            
    print('Acceptance rate: {0}'.format(acceptRate))  
        
    if analyseFlag is True:
      #print('Sample variance: {0}'.format(covarianceMatrix))
      #print('Sample mean: {0}'.format(sampleMean))
      helperDim=analyseDim+1
      print('Sample mean of dimension {1}: {0}'.format(sampleMean[analyseDim], helperDim))
      print('Sample variance of dimension {1}: {0}'.format(covarianceMatrix[analyseDim][analyseDim], helperDim))


    returnValue=[]
    returnValue.append(acceptRate)
    returnValue.append(samples)
    returnValue.append(sampleMean)
    returnValue.append(covarianceMatrix)
    if analyseFlag is True:
      returnValue.append(allSampleMean)
      returnValue.append(allCovariance)
    return returnValue
    
  def analyseData(self, samples, mean, variance, printName, allSampleMean = [], allCovariance = []):
    """
    Here we analyse only the first component (for the sake of simplicity)
    """
          
    print('Analysing sampled data...')

    helper=numpy.shape(samples)
    dimension=helper[0]
    numberOfSamples=helper[1]
    # For parallelization (number of processors)
    procs = 2
    results = numpy.array([0.0 for i in range(procs)])
    # Analyse the first component!
    dim=0
    # Maximal number of lag_k autocorrelations
    maxS=int((numberOfSamples-1)/3)
    # lag_k autocorrelation
    autocor = [0.0 for i in range(maxS)]
    autocor[0]=variance[dim][dim]
    # sample frequency
    m=1
    # modified sample variance = autocor[0] + 2 sum_{i}(autocor[i])
    msvar=0.0
    # SEM = sqrt( msvar/numberOfSamples )
    sem=0.0
    # ACT = m * msvar / autocor[0]
    act=0.0
    # ESS = m * numberOfSamples / act
    ess=0.0
          
    temp=0.0

    flagSEM=True
    flagACT=True
    flagESS=True

    # Calculate lag_k for following k's
    evaluation = range( maxS )
    evaluation = evaluation[1:]
    evaluation2 = numpy.arange(numberOfSamples)
    for lag in evaluation:

      evaluation2 = evaluation2[:-1]
      # Do this expensive calculation parallel
      output = mp.Queue()
      morsel = numpy.array_split(evaluation2, procs)
      processes =  []
      for i in range(procs):
        processes.append( mp.Process(target = self.calculateACF, args = (samples[dim], mean[dim], lag, morsel[i], output, ) ) )
      for p in processes:
        p.start()
      for p in processes:
        p.join()
      results = [output.get() for p in processes]
      tmp = numpy.sum(results)
      autocor[lag] = (numberOfSamples-lag)**-1 * tmp
      # noise affects autocorrelation -> stop when near zero
      if (autocor[lag-1]+autocor[lag])<=0.01:
        maxS = lag
        break
      percentage = format(100*lag/maxS, '.2f')
      print('Processing: {0}%'.format(percentage), end='\r')
      
    # Calculate the modified sample variance
    evaluation = range( maxS-1 )
    evaluation = evaluation[1:]   
    msvar += autocor[0]
    # Plot Standard error of mean
    allSem = [0.0 for i in range(maxS-1)]
    allSem[0]= 1.0
    for lag in evaluation:
      msvar += 2*autocor[lag]
      # Calculate the autocovariance function by dividing by variance and multiplying a factor
      autocor[lag] = autocor[lag]/autocor[0]
      # Sample standard error of the Mean
      allSem[lag] = ( math.sqrt( abs(msvar)/ lag ) )
    # Standard Error of the Mean
    sem = math.sqrt(abs(msvar)/numberOfSamples)
    # AutoCorrelation Time
    act = m*msvar/autocor[0]
    # Effective Sample Size
    ess = m*numberOfSamples/act
    # Normalizing autocor[0] for plots
    autocor[0] = 1.0

    print('Modified sample variance: {0}'.format(msvar))   
    print('Standard Error of the Mean: {0}'.format(sem))   
    print('AutoCorrelation Time: {0}'.format(act))   
    print('Effective Sample Size: {0}'.format(ess))   

    #Print some results
    if False:
      if True:
        iterations=range(numberOfSamples)
        pylab.plot(iterations, allSampleMean, label='Sample mean')
        pylab.ylabel('Sample mean', fontsize=10)
        pylab.xlabel('Iterations', fontsize=10)
        pylab.ylim([-3.0, 3.0])
        pylab.grid(True)
        newPrintName = printName.replace(".png", "_mean.png")
        pylab.savefig(newPrintName)
        pylab.clf()  

        pylab.plot(iterations, allCovariance, label='Sample covariance')
        pylab.ylabel('Sample covariance', fontsize=10)
        pylab.xlabel('Iterations', fontsize=10)
        pylab.ylim([2.0, 8.0])
        pylab.grid(True)
        newPrintName = printName.replace(".png", "_cov.png")
        pylab.savefig(newPrintName)
        pylab.clf()      

      maxS = 41
      lag=range(maxS-1)
      #pylab.subplot(311)
      #pylab.suptitle('Analysis of the MCMC simulation')
      pylab.plot(lag, autocor[:maxS-1], 'r', label='Autocorrelation')
      pylab.ylabel('ACF', fontsize=10)
      pylab.xlabel('Iterations', fontsize=10)
      pylab.grid(True)
      newPrintName = printName.replace(".png", "_1.png")
      pylab.savefig(newPrintName)
      pylab.clf()

      pylab.plot(lag, allSem[:maxS-1], 'r', label='sem')
      pylab.ylabel('Standard error of the mean', fontsize=10)
      pylab.xlabel('Iterations', fontsize=10)
      pylab.grid(True)
      newPrintName = printName.replace(".png", "_sem.png")
      pylab.savefig(newPrintName)
      pylab.clf()

      iterations=range(numberOfSamples)
      #pylab.subplot(312)
      pylab.plot(iterations, samples[dim], label='First dimension of samples')
      pylab.ylabel('First dim of samples', fontsize=10)
      pylab.xlabel('Iterations', fontsize=10)
      #pylab.ylim([-6.5, 6.5])
      pylab.grid(True)
      newPrintName = printName.replace(".png", "_2.png")
      pylab.savefig(newPrintName)
      pylab.clf()
      
      #pylab.subplot(313)
      num_bins=100
      n, bins, patches=pylab.hist(samples[dim], num_bins, normed=1, facecolor='green', alpha=0.5, label='Histogram of the first dimension')
      # add a 'best fit' line
      #y = 1.0 * mlab.normpdf(bins, 0.0, 1) + 0.0 * mlab.normpdf(bins, 3.0, 1)
      y = 1.0 * mlab.normpdf(bins, 0.0, 1)
      #y = 0.5 * mlab.normpdf(bins, -2.0, 1.0) + 0.5 * mlab.normpdf(bins, 2.0, 1.0)
      plt.plot(bins, y, 'r--')
      pylab.xlabel('First dimension of samples', fontsize=10)
      pylab.ylabel('Relative frequency', fontsize=10)
      #pylab.xlim([-6.0, 6.0])
      pylab.grid(True)
      newPrintName = printName.replace(".png", "_3.png")
      pylab.savefig(newPrintName)

      #pylab.savefig(printName)
      pylab.clf()
      if True:
        newPrintName = printName.replace(".png", "Plot.png")
        self.scatterPlot3D(samples, newPrintName)
        newPrintName = printName.replace(".png", "Histo.png")
        self.Histogram3D(samples, newPrintName)
      newPrintName = printName.replace(".png", "_short.png")
      iterations=range(1000)
      pylab.plot(iterations, samples[dim][:1000], label='First dimension of samples')
      pylab.ylabel('First dim of samples', fontsize=10)
      #pylab.ylim([-6.5, 6.5])
      pylab.grid(True)
      pylab.savefig(newPrintName)
      pylab.close('all')
    if False:
      return act
    if True:
      return sem

  def calculateACF(self, samples, mean, lag, array, output):
    """
    A helper function to calculate the autocorrelation coefficient parallel
    """
    tmp=0.0
    for lag2 in array:
      tmp += (samples[lag2]-mean)*(samples[lag2+lag]-mean)
    output.put(tmp)
      
  def setAlgoType(self, Algo):
    self._algoType = Algo
  
  def setBurnIn(self, BurnIn):
    if BurnIn is not None:
      self._burnIn = BurnIn
    else:
      self._burnIn = 10000
      
  def setDimension(self, dimension):
    self._dimension = dimension
    
  def evaluateMultimodalGaussian(self, position=[]):
    """
    Implement the target distribution without normalization constants. 
    Here we have the multimodal example of Roberts and Rosenthal (no product measure)
    """
    m = 2.0
    tmp = []
    for dim in range( self._dimension ):
      tmp.append( position[dim]**2 )
    tmp= tmp[1:]
    return math.exp( - 0.5 * ( (position[0]-m)**2 + math.fsum(tmp) )) + math.exp( - 0.5 * ( (position[0]+m)**2 + math.fsum(tmp) ))
 
  def evaluateGradientOfMultimodalGaussian(self, evaluateMultiModalGaussian, position=[]):
    """
    Calculates the analytical gradient
    """
    m=2.0
    tmp=[]
    grad=[]
    interval=range( self._dimension )

    for dim in interval:
      tmp.append( position[dim]**2 )
    tmp=tmp[1:]
    grad.append( (-(position[0]-m)*math.exp( - 0.5 * ( (position[0]-m)**2 + math.fsum(tmp) )) -(position[0]+m)*math.exp( - 0.5 * ( (position[0]+m)**2 + math.fsum(tmp) )) )/evaluateMultiModalGaussian(position) )
    interval=interval[1:]
    for dim in interval:
      grad.append( (-position[dim]*math.exp( - 0.5 * ( (position[0]-m)**2 + math.fsum(tmp) )) - position[dim]*math.exp( - 0.5 * ( (position[0]+m)**2 + math.fsum(tmp) )) )/evaluateMultiModalGaussian(position) ) 
    
    return grad

  def evaluateGaussian(self, position=[]):
    """
    Simple multi-dimensional Gaussian
    """
    mean1=0.0
    mean2=-0.0
    variance1=1.0
    variance2=1.0
    value=1.0
    for dim in range(self._dimension):
      value *= variance1**(-0.5) * math.exp(-0.5*(position[dim]-mean1)**2 *variance1**-1) + variance2**(-0.5) * math.exp(-0.5*(position[dim]-mean2)**2 *variance2**-1)
    return value
    
  def evaluateGradientOfGaussian(self, evaluateGaussian, position=[]):
    """
    Calculates the analytic gradient of a Gaussian.
    """
    mean1=0.0
    mean2=-0.0
    variance1=1.0
    variance2=1.0
    
    tmp1=[]
    tmp2=[]
    grad=[]
    
    interval=range( self._dimension )

    for dim in interval:
      tmp1.append( (position[dim]-mean1)**2 )
      tmp2.append( (position[dim]-mean2)**2 )
    for dim in interval:
      grad.append( (-(position[dim]-mean1)*variance1**(-0.5)*math.exp( - 0.5 * variance1**(-1) * (  math.fsum(tmp1) )) + (position[0]-mean2)*variance2**(-0.5)*math.exp( - 0.5 * variance2**(-1) * (  math.fsum(tmp2) ) )) / evaluateGaussian(position) )
    return grad
      
  def evaluateChangedGaussian(self, position=[]):
    """
    Simple changed multi-dimensional Gaussian
    """
    sigma2 = 1.0
    s = 0.5

    value=1.0
    for dim in range(self._dimension):
      value *=  math.exp(-0.5*(position[dim])**2 *(sigma2**-1 + dim**(2*s)) )
    return value
    
     
      
  def calculateGradient(self, evaluateTargetDistribution, position):
    """
    Calculate the gradient of the logarithm of your target distribution by finite differences
    """
    if True:
      h = 1e-10
      grad = []
      shiftedpos1 = []
      shiftedpos2 = []
      for dim in range(self._dimension):
        shiftedpos1.append( position[dim] )
        shiftedpos2.append( position[dim] )
      for dim in range(self._dimension):
        shiftedpos1[dim] += h
        shiftedpos2[dim] -= h
        tmp1=evaluateTargetDistribution(shiftedpos1)
        # Check if value of the target distribution is not to small
        if tmp1 <= 0.0:
          tmp1=h
        tmp2=evaluateTargetDistribution(shiftedpos2)
        if tmp2 <= 0.0:
          tmp2=h
        grad.append(0.5 * h**-1 * ( math.log(tmp1)-math.log(tmp2) ))
        shiftedpos1[dim] -= h
        shiftedpos2[dim] += h
      return grad
        
  def generateGaussian(self, mean, variance):
    """
    Generates one-dimensional Gaussian random variable 
    """
    gauss=random.gauss(mean, math.sqrt(variance))
    return gauss
    
  def acceptanceStep(self, evaluateTargetDistribution, calculateGradient,  proposal=[], position=[], mean=[], variance=None):
    """
    Determines wether the proposal is accepted or rejected.
    """
    u = random.uniform(0,1)
    ratio = ( evaluateTargetDistribution(proposal) )/( evaluateTargetDistribution(position) )
    # Calculate the ratio of the transition kernels
    if self._algoType in ['MALA']:
      tmp2=0.0
      grad = calculateGradient( evaluateTargetDistribution, proposal )
      for dim in range(self._dimension):
        tmp = proposal[dim] + 0.5*variance*grad[dim]
        tmp2 += -(mean[dim]-proposal[dim])**2 + (tmp-position[dim])**2
      tmp = 0.5*variance**-1 * tmp2
      if ( tmp > 100. ):
        tmp = 100.
      elif ( tmp < -100. ):
        tmp = -100.
      ratio *= math.exp(-tmp)
    if u < ratio:
      proposal[self._dimension]=True
      return proposal
    else:
      position[self._dimension]=False
      return position

  def scatterPlot3D(self, samples, printName):
   """
   Plot samples in 3D as cloud.
   """
   vec = [ [ [],[],[] ] for i in range(5) ] 
   xs = []
   ys = []
   zs = []
   
   fig = plt.figure()
   ax = fig.add_subplot(111, projection='3d')

   # Control number and dimension of samples
   helper=numpy.shape(samples)
   dimension=helper[0]
   numberOfSamples=helper[1]
   if numberOfSamples < 5000:
     return 0
     
   # Sort your samples
   for it in range(numberOfSamples):
     if (it < 500):
       vec[0][0].append(samples[0][it])
       vec[0][1].append(samples[1][it])
       vec[0][2].append(samples[2][it])
     elif (it >= 500 and it < 1000):
       vec[1][0].append(samples[0][it])
       vec[1][1].append(samples[1][it])
       vec[1][2].append(samples[2][it])
     elif (it >= 1000 and it < 1500):
       vec[2][0].append(samples[0][it])
       vec[2][1].append(samples[1][it])
       vec[2][2].append(samples[2][it])
     elif (it >= 1500 and it < 2000):
       vec[3][0].append(samples[0][it])
       vec[3][1].append(samples[1][it])
       vec[3][2].append(samples[2][it])
     elif (it >= 2000 and it < 2500):
       vec[4][0].append(samples[0][it])
       vec[4][1].append(samples[1][it])
       vec[4][2].append(samples[2][it])
     else:
       break

   # Make scatterplots
   for c, sample in [('r', vec[0]), ('y', vec[1]), ('g', vec[2]), ('c', vec[3]), ('b', vec[4])]:
     xs = sample[0]
     ys = sample[1]
     zs = sample[2]
     ax.scatter(xs, ys, zs, c=c, marker='o')
     
   ax.set_xlabel('X ')
   ax.set_ylabel('Y ')
   ax.set_zlabel('Z ')
   ax.set_xlim([-6.0, 6.0])
   ax.set_ylim([-3.0, 3.0])
   ax.set_zlim([-3.0, 3.0])
   ax.view_init(23, 98)
   pylab.savefig(printName, dpi=200)
   #plt.show()
   pylab.clf()


  def Histogram3D(self, samples, printName):
   """
   Plot histogram of 2D samples.
   """
   fig=plt.figure()
   ax = fig.add_subplot(111)

   x = samples[0]
   y = samples[1]

   H, xedges, yedges, img = ax.hist2d(x, y, bins=50, norm=LogNorm())
   #extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
   #im = ax.imshow(H, cmap=plt.cm.jet)#, extent=extent)
   #fig.colorbar(im, ax=ax)
   ax.set_xlabel('X ')
   ax.set_ylabel('Y ')
   ax.set_xlim([-6.0, 6.0])
   ax.set_ylim([-4.0, 4.0])
   pylab.savefig(printName, dpi=200)
   #pylab.show()
   pylab.clf()


def TestVariance (string, dimension, variance, numberOfSamples, resultName):
  """
  Test one variance.
  """
  algo = Algo(string, dimension, 1)
  result = algo.simulation(numberOfSamples, variance, True, True, [2.0])
  tmp = algo.analyseData(result[1], [0.0], result[3], resultName)
   
	
#makeDiagramm('MALA', 10, 1000, 'GEANY01', 'ACT')
#algo=Algo('RWM', 3)
#samples=algo.simulation(10000, 2.2, analyticGradient=False, analyseFlag=False, returnSamples=True)
#algo.scatterPlot3D(samples[1])

#   fig=plt.figure()
#   ax = fig.add_subplot(111, projection='3d')
#  
#   x = samples[0]
#   y = samples[1]
#   hist, xedges, yedges = numpy.histogram2d(x, y, bins=12)
#   elements = (len(xedges) - 1) * (len(yedges) - 1)
#   xpos, ypos = numpy.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)
#   xpos = xpos.flatten()
#   ypos = ypos.flatten()
#   zpos = numpy.zeros(elements)
#   dx = 0.5 * numpy.ones_like(zpos)
#   dy = dx.copy()
#   dz = hist.flatten()
#   
#   ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
#   ax.view_init(28, 13)
#   pylab.savefig(printName, dpi=200)
#   #pylab.show()
#   pylab.clf()
