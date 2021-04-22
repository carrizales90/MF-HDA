# Rescaled_CIL.py                   C. Carrizales-Velazquez 21 April 2021
#
#
#-------------------------------------------------------------------------------------------
# Rescaled_CIL: Calculate the rescaled confidence interval lenght of "generalized Higuchi length L(q,k)" for a negative q-moment (we recomend to use the minimum q-moment of the Multifractal Higuchi Dimension Analysis MF-HDA). 
#
# We calculate the confidence interval (CI) for a set whose percentile (p_r) have been removed. Here we set 0 <= p_R <= 20 , but you can modify this in line 62 of original version.
#
# At the end, we normalize each CI by the CI-value calculated with p_r = 0 in oder to get a stability reference in the linear fit of the log(L(q,k)) vs log(k)
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
#
#-------------------------------------------------------------------------------------------

# Usage:
#
#  $ python3 Rescaled_CIL.py First_10K_integrated_Ulysses_Book.dat -5 25 0 16 0.975 Ulysses
#
# where: 
#  ---> First_10K_integrated_Ulysses_Book.dat is the file to read (this contain the time-series to analyze)
#  ---> -5 is the q-moment used in order to analyze the instability of L(q,k). It must be a negative value
#  ---> 25 is the initial value of scale k used in the MF-HDA
#  ---> 0 indicates the maximum scale k used in the MF-HDA (IF IT IS ZERO THEN THIS PROGRAM TAKE N/10, where N=length of time-series)
#  ---> 16 is the number of bins used in the histogram for the expectation of Delta_X (for more details see section 2.2 of our publication: "Generalization of Higuchi’s fractal dimension for multifractal analysis of short time series")
#  ---> 0.975 is the significance level used in the T-Student distribution for the CI
#  ---> Ulysses is the name of outfiles


import matplotlib.pyplot as plt
import scipy as sp
from scipy import stats
import numpy as np
import seaborn as sns
import sys,os

# save the inputs
FILE = str(sys.argv[1]); q = float(sys.argv[2]); ini = int(sys.argv[3]); fin = int(sys.argv[4]); Nbins=int(sys.argv[5]); pi = float(sys.argv[6]); name = str(sys.argv[7])

##################################################### create a folder to save the results there
def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' + directory)

createFolder('Rescaled_CIL_'+name+'/')
####################################################

# reading data
data = np.loadtxt(FILE); x = data.transpose()
N = len(x)

# define the end of the scale k
if fin == 0:
	KF=N/10
else:
	KF=fin

# width of bins are calculated and, calculation for different p_r values star
wbin = 100/Nbins; PerF = 100; Percentile=[]; Confidence_Interval=[]
for PerI in np.arange(21): # here you can change the values of p_r for this stability study (this line 62 of original code)
	k = ini; M=[[],[]] # in array M we will save the values of k and CI

	while k <= KF:
		M[0].append(k);subN=[] # M[0] save the scale k
		
		preDx =[]	
		for pm in range(k):
			m = pm + 1
			pDx = abs(np.diff(x[m-1:N:k])); preDx.append(pDx[pDx != 0]) # calcualte the Delta_X for each scale k and for each m-value (for more details see chapter 2.1 of our paper). We also discard vales of Delta_X=0, this will produce incosistencies for negative q-moments
			
		Dx = np.concatenate(preDx) # we join all the m-subseries of Delta_X

		FI = np.percentile(Dx, PerI); FS = np.percentile(Dx, PerF); Dx = Dx[ Dx >= FI]; Dx = Dx[ Dx <= FS];	bin_edges = [] # we retire data according to p_r
		for i in np.arange(0,100+wbin, wbin):
			bin_edges.append(np.percentile(Dx, i)) # we set the bin_edges in order to get equi-probable bins

		BE = np.array(bin_edges); X=[]
		for i in range(len(BE)-1):
			X.append([]) # we create void-arrays for regrouping Delta_X according to its bin-belonging

		for i in range(len(BE)-1):
			for j in Dx:
				if i == 0: # just first bin should be a close interval
					if j >= BE[i] and j <= BE[i+1]:
						X[i].append(j) # we save Delta_X according to each bin-belonging
				else:
					if j > BE[i] and j <= BE[i+1]:
						X[i].append(j) # we save Delta_X according to each bin-belonging
		
		W = (np.diff(BE)); H = np.histogram(Dx, bins = BE, density=True); P = H[0] * W  # calculate the probability of each bin
		preL=[]; subNDX=[]
		for i in X:
			subNDX.append( (np.array(i)**q) / len(i)) # we save the q-moments of Delta_X
		
		for i,j in zip(subNDX, P):
			for a in i:
				preL.append(a * j) # calculate elements of expected Delta_X
		
		L=sum(preL) # calculate the expected Delta_X
		M[1].append((float(N-1)/k**2) *  L**(1.0/q) ) # calculate the L(q,k)

		k = int(k*np.sqrt(np.sqrt(2))) + 1  # the increment of scale k is logarithmic

	m, b = np.polyfit(np.log10(M[0]),np.log10(M[1]),1) # calculate the fit of log(L(q,k)) vs log(k)

	Y = []
	for i,j in zip(np.log10(M[0]),np.log10(M[1])):
		Y.append((j - m*i + b)**2) # calculate de estimated L(q,k)

	stdev = np.sqrt( (1/(len(M[0])-2))*sum(Y) ); one_minus_pi = 1 - pi; ppf_lookup = 1 - (one_minus_pi / 2); z_score = sp.stats.norm.ppf(ppf_lookup); interval = z_score * (stdev/np.mean(Y)); CI_sup = m + interval; CI_inf = m - interval # calculate CI (measured in terms of the estimated standard deviation)

	Percentile.append(PerI); Confidence_Interval.append(CI_sup - CI_inf) # save p_r and the corresponding CIL (see appendix B of our paper)

Renorm_CI = np.array(Confidence_Interval)/Confidence_Interval[0] # we normalize by CIL of p_r=0 (this is Î/Ì0 of figure 11 of our paper)
np.savetxt('Rescaled_CIL_'+name+'/R_CIL_'+name+'.txt',np.matrix([Percentile, Renorm_CI]).transpose(),fmt='%s') # save data in format .txt

sns.set_style('darkgrid')

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(Percentile, Renorm_CI)

ax.set_xlabel(r'Removed percentile, $p_r$', fontsize=18, style='italic')
ax.set_ylabel(r'Rescaled Confidence Interval, $\^{I}/\^{I}_{p_r=0}$', fontsize=18, style='italic')
ax.set_title(r'Results of '+name+' with L(q='+str(q)+',$k\in$['+str(ini)+'-'+str(int(M[0][len(M[0])-1]))+'])', fontsize=20, style='italic') # here is show the interval k used, as we increase k in a logarithm way, the last k scale always is <=fin (where fin is chosen in the inputs (line 37 of original version))
ax.grid(True, which="both")

fig.savefig('Rescaled_CIL_'+name+'/'+name+'.png') # save figure of Î/Ì0
plt.show() # also this figure is shown
