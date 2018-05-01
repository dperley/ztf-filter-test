# testfilter.py
# Written by Daniel Perley, 2018 April 30

# Simulates a SN light curve, converts it to the avro schema, and runs it
# through your filter 

# Change the first line below (or save your filter as filter.py) to test your filter.


from filter import compiledFunction
#from fasttransients import compiledFunction 
#from redshiftcompleteness import compiledFunction
#from sciencevalidation import compiledFunction
#from slsne import compiledFunction
import numpy
from numpy import exp, log10
import matplotlib.pyplot as plt
import warnings

def snfunc(dt, A, b, alpha, beta):
   # Analytic function to produce a SN light curve.
   tsub = dt*1.0
   tsub[dt <= 0] = 1. # avoid math errors in a lazy way
   f = A * (tsub/b)**beta / (exp((tsub/b)**alpha) - 1)
   f[dt <= 0] = 0.
   return f

def avro(fid, t, mag, magerr, limit, hostg=20., hostr=20., hosti=20., hostz=20., prev=0):
  # turn observation into an avro packet formated python dictionary
  # Most parameters are hard-coded to arbitrary, safe values.  Please report any spelling or typing mistakes.
  av = {'jd':2458120.+t, 'fid':fid, 'pid':0, 'diffmaglim':limit, 
                'pmdiffimfilename':'',
                'programpi':'', 'programid':'', 'candid':0, 'isdiffpos':'t', 'tblid':0, 'nid':0, 'rcid':0, 'field':0, 'xpos':1000., 'ypos':1000.,
                'ra':45.0, 'dec':45.0,
                'magpsf':mag, 'sigmapsf':magerr,  'chipsf':0., 'magap':mag, 'sigmagap':magerr, 
                'distnr':0, 'magnr':hostr, 'sigmagnr':0.0, 'chinr':0., 'sharpnr':0., 'sky':0., 'magdiff':0., 'fwhm':1., 'classtar':0., 
                'mindtoedge':100., 'magfromlim':limit-mag, 'seeratio':1.0, 'aimage':1., 'bimage':1., 'aimagerat':1., 'bimagerat':1., 'elong':1., 'nneg':0,
                'nbad':0, 'rb':1., 'ssdistnr':99999., 'ssmagnr':20., 'ssnamenr':'', 'sumrat':0.25, 'magapbig':mag, 'sigmagapbig':magerr,
                'ranr':45.0, 'decnr':45.0, 'scorr':5.}
  # If not a prevcandidate, information about the host and history is also included.
  if prev==0: 
    av.update({'sgmag1':hostg, 'srmag1':hostr, 'simag1':hosti, 'szmag1':hostz,
                'sgscore1':0.0, 'distpsnr1':0.,
                'ndethist':0, 'ncovhist':0, 'jdstarthist':0., 'jdendhist':0.,
                'tooflag':0, 'objectidps1':0, 
                'objectidps2':0, 'sgmag2':20, 'srmag2':20, 'simag2':20, 'szmag2':20, 'sgscore2':0.0, 'distpsnr2':0.,
                'objectidps3':0, 'sgmag3':20, 'srmag3':20, 'simag3':20, 'szmag3':20, 'sgscore3':0.0, 'distpsnr3':0.,
                'nmtchps':10, 'rfid':100, 
                'jdstartref':0., 'jdendref':0., 'nframesref':10 })
  return av


class lc:
   # Light curve object.
   t = 0.
   mag  = 0.
   magerr = 0.
   limit = 0.
   islimit = False
   fid = 0
   filter = ''

class galaxy:
   # Galaxy/counterpart object.
   g = 20.
   r = 20.
   i = 20.
   z = 20.

   def __init__(self, g=20., r=20., i=20., z=20.):
      self.g = g
      self.r = r
      self.i = i
      self.z = z


def createlc(dt, filterid=0, mag=-1, magerr=-1, islimit=-1, peakmag=18., peaktime=10., peakcolor=0., alpha=1.5, beta=2.0):
   # Simulate a multi-color light curve given an array of observation times and filters,
   # and the parameters of the light curve.
   # Can specify actual data values (mag, magerr, limit, as lists/arrays) or
   #     parameters for a simulated curve (peakmag, peaktime, peakcolor).

   # color is g-r mag.  (positive = redder)

   if type(filterid) is int:
      # Scalar filter ID (all points the same filter)
      if filterid == 0: 
         fid = numpy.array([1]*len(dt))
      else:
         fid = numpy.array([fid]*len(dt))
   else:
      # List filter ID: filters specified as a list by user
      fid = numpy.array(filterid)

   if type(mag) is int:
     mode = 'function'
     # numerically determine the function's peak time and flux and rescale to user specification
     alldt = numpy.arange(0.01,200,0.01)
     allf = snfunc(alldt, 1., peaktime, alpha, beta)
     b = peaktime * peaktime / ((alldt[allf == max(allf)])[0])
     allf = snfunc(alldt, 1., b, alpha, beta)
     A = 10.0**(-peakmag/2.5 + 9.56) / max(allf)

     warnings.filterwarnings("ignore") # ignore the logarithm of negative warnings

     # Simulate color evolution
     color = peakcolor-0.2*(exp(1-dt/peaktime)-1)
     mcolor = peakcolor-0.2*(exp(1-alldt/peaktime)-1)

     # Produce the functional/noiseless curves
     allfr = snfunc(alldt, A, b, alpha, beta)
     allfg = allfr*10.0**(-mcolor/2.5)

     # Produce the data (including noise)
     func = 5.+numpy.random.randn(len(dt))   # RMS flux noise of 5 uJy
     ferr = 0.1*func*numpy.random.randn(len(dt)) # 3 sigma is 21 mag at best
     fracerr = numpy.random.randn(len(dt))*0.03  # calibration etc. error, prop. to flux

     f  =  snfunc(dt, A, b, alpha, beta)*(1+fracerr) + ferr

     # Color adjustment
     f[fid == 1] *= 10.0**(-color[fid == 1]/2.5)
     f[fid == 3] /= 10.0**(-color[fid == 3]/2.5)

     # Convert from flux to magnitude
     mag    = 2.5*(9.56 - log10(f))
     magerr = ((-2.5*(9.56 - log10(f+func)) + mag)**2 + fracerr**2)**0.5
     allmagr = 2.5*(9.56 - log10(allfr)) 
     allmagg = 2.5*(9.56 - log10(allfg)) 
     limit =  2.5*(9.56 - log10(5.*func))
     islimit = (f/func) < 3.
     mag[islimit] = 99.
   else:
     mode = 'data'
     if type(magerr) is int: magerr = numpy.zeros(len(mag))+0.01
     if type(islimit) is int: islimit = [False]*len(mag)
     limit = [21.5]*len(mag)

   # Plotting
   cols = ['', 'green', 'red', 'yellow']
   dcols = ['']*len(fid)
   for i in range(len(fid)): dcols[i] = cols[fid[i]]

   #plt.errorbar(dt[isdetection], mag[isdetection], yerr=magerr[isdetection], fmt='none', ecolor=cols[fid[isdetection]])
   #plt.plot(dt[isdetection], mag[isdetection], 'o', color=cols[fid[isdetection]])
   #plt.plot(dt[islimit],limit[islimit], 'v', color=cols[fid[islimit]])
   if mode=='function':
     plt.plot(alldt, allmagr, color='red')
     plt.plot(alldt, allmagg, color='green')
   for i in range(len(dt)):
     if islimit[i]==False:
        plt.errorbar(dt[i], mag[i], yerr=magerr[i], fmt='none', ecolor=cols[fid[i]])
        plt.plot(dt[i], mag[i], 'o', color=cols[fid[i]])
     else:
        plt.plot(dt[i],limit[i], 'v', color=cols[fid[i]])
     
   plt.xlim((min(dt)-5, max(dt)+5))
   plt.ylim((21,min(18,peakmag-0.3)))
   plt.xlabel('t')
   plt.ylabel('mag')

   # Store as a list of light curve objects and return
   outlc = []
   for i in range(len(mag)):
      lci = lc()
      lci.t = dt[i]
      lci.mag = mag[i]
      lci.magerr = magerr[i]
      lci.limit = limit[i]
      lci.islimit = islimit[i]
      lci.fid = fid[i]
      if fid[i]==1: lci.filter = 'g'
      if fid[i]==2: lci.filter = 'r'
      if fid[i]==3: lci.filter = 'i'
      outlc.append(lci)

   outlc = sorted(outlc, key=lct)

   return outlc


def lct(lc):
  return lc.t


def createobs(lcurve, host):

   # Format a list of light curve data points into a list of avro packets.  
   # The most recent observation is the 'current' observation, all others are previous candidates stored in prv_candidates.

   prevobs = []
   for i in range(len(lcurve)):
      if i < len(lcurve)-1:
         addobs = avro(lcurve[i].fid, lcurve[i].t, lcurve[i].mag, lcurve[i].magerr, lcurve[i].limit, prev=1)
         prevobs.append(addobs)
      else:
         curobs = avro(lcurve[i].fid, lcurve[i].t, lcurve[i].mag, lcurve[i].magerr, lcurve[i].limit, host.g, host.r, host.i, host.z)
   obs = {'candidate':curobs, 'prv_candidates':prevobs}
   return obs



def main():

   # Modify this part to change the sampling.

   # simulate MSIP: 3 night cadence, 2 filters same-night, occasional weather losses
   # does not currently simulate pairs (i.e. for asteroid rejection)
   dtobs = numpy.array(range(-10,80,3)) 
   weather = numpy.random.random(len(dtobs))
   dtobs = dtobs[weather > 0.25]
   dt = numpy.append(dtobs, dtobs+0.1)
   fid = numpy.array([1]*len(dtobs) + [2]*len(dtobs)) # g and r both every night
   weather2 = numpy.random.random(len(dt))
   dt = dt[weather2 > 0.05]
   fid = fid[weather2 > 0.05]


   # simulate nightly
   #dt = 1.0*numpy.array(range(-10,100,1))
   #weather = numpy.random.random(len(dt))
   #dt = dt[weather > 0.25]
   #n = len(dt)
   #fid = numpy.array(range(n)) % 2 + 1 # alternate g(1) and r(2)
 
   # Create the light curve
   lc = createlc(dt, fid, peakmag=19., peaktime=20.) # alpha=1.5, beta=2.0 (shape parameters)
   # Below: a user-specified light curve
   #lc = createlc([5., 10., 15., 20], [1,1,1,1], [20., 19., 18.5, 18.2], [0.2, 0.2, 0.1, 0.1])

   host = galaxy(g=20., r=20., i=20., z=20.)

   passed = [False]*len(lc)

   # Try out the filter, and see which candidates result in pass as light curve builds up
   print 'Filter passes for these epochs:'
   for i in range(1,len(lc)):

      # Create the avro packet
      obs = createobs(lc[0:i], host) # createobs is rerun every time (suboptimal)

      # See if the target passes the filter
      (filteron, annotations) = compiledFunction(obs)
      passed[i] = filteron

      if passed[i]: plt.plot(lc[i].t, lc[i].mag, 'o', mfc='none',ms = 20)
      if passed[i]: print "  t =%7.2f" % lc[i].t, ': ', lc[i].filter, '=', "%.2f" %lc[i].mag, '+/-', "%.2f" %lc[i].magerr
      #print i, '/', len(lc),': ', lc[i].t, ' ', lc[i].mag, ' ', lc[i].magerr, ' ', passed[i]

   

   # Print out your annotations
   print 'Annotations:'
   for a in sorted(annotations.iterkeys()): print ' ', a, annotations[a]

   # Show the plot and prompt user to end
   plt.show(block=False)
 
   try:
      q = raw_input("Press Enter to continue... ")
   except:
      q = input("Press Enter to continue... ")

   return



main()
