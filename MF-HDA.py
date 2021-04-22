# MF-HDA.py                   C. Carrizales-Velazquez 21 April 2021
#
#
#-------------------------------------------------------------------------------------------
# MF-HDA: Calculate the Multifractal Higuchi Dimension Analysis (MF-HDA) of our paper: "Generalization of Higuchi’s fractal dimension for multifractal analysis of short time series"
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY
#
#-------------------------------------------------------------------------------------------

# Usage:
#
#  $ python3 MF-HDA.py -5 5 0.25 First_10K_integrated_Ulysses_Book.dat 25 0 15 16 Ulysses
#
# where: 
#  ---> -5 is the minimum q-moment of L(q,k)
#  ---> 5 is the maximum q-moment of L(q,k)
#  ---> 0.25 is the separation between q-moments used in the MF-HDA
#  ---> First_10K_integrated_Ulysses_Book.dat is the file to read (this contain the time-series to analyze)
#  ---> 25 is the value initial scale k used in the MF-HDA
#  ---> 0 indicates the maximum scale k used in the MF-HDA (IF IT IS ZERO THEN THIS PROGRAM TAKE N/10, where N=length of time-series)
#  ---> 15 is p_r
#  ---> 16 is the number of bins used in the histogram for the expectation of Delta_X (for more details see section 2.2 of our publication: "Generalization of Higuchi’s fractal dimension for multifractal analysis of short time series")
#  ---> Ulysses is the name of outfiles
#
# these inputs can be changed as you want

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os, sys

# save the inputs
q_min = float(sys.argv[1]); q_max = float(sys.argv[2]); dq = float(sys.argv[3]); FILE = str(sys.argv[4]); ini = int(sys.argv[5]); fin = int(sys.argv[6]); PerI = int(sys.argv[7]); PerF = 100; Nbins=int(sys.argv[8]); name = str(sys.argv[9]);M=[];Q=np.arange(q_min-dq,q_max+2*dq,dq);NQ=len(Q)

##################################################### create a folder to save the results there
def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' + directory)

createFolder('MFH_P_'+str(PerI)+'-100_'+name+'_ki'+str(ini)+'_kf'+str(fin)+'_Nbins'+str(Nbins)+'/')
####################################################

for j in range(NQ+1): 
	M.append([]) # create the array for L(q,k)

# reading data
data = np.loadtxt(FILE); x = data.transpose()

N = len(x); k = ini # set the initial scale k

# define the end of the scale k
if fin == 0:
	KF=N/10
else:
	KF=fin

wbin = 100/Nbins # width of bins are calculated
while k <= KF:
	M[0].append(k);subN=[];L=[] # M[0] save the scale k
	
	preDx =[]	
	for pm in range(k):
		m = pm + 1
		pDx = abs(np.diff(x[m-1:N:k])); preDx.append(pDx[pDx != 0]) # calculate the Delta_X for each scale k and for each m-value (for more details see section 2.1 of our paper). We also discard vales of Delta_X=0, this will produce incosistencies for negative q-moments
		
	Dx = np.concatenate(preDx) # we join all the m-subseries of Delta_X
	FI = np.percentile(Dx, PerI); FS = np.percentile(Dx, PerF); Dx = Dx[ Dx >= FI]; Dx = Dx[ Dx <= FS];	bin_edges = [] # we remove data according to p_r
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
	for I in range(NQ):
		preL=[]; subNDX=[]
		if Q[I] == 0:
			for i in X:
				subNDX.append( (np.log(np.array(i))) / len(i)) # we save the q-moments of Delta_X (this line is for q=0)
		else:
			for i in X:
				subNDX.append( (np.array(i)**Q[I]) / len(i)) # we save the q-moments of Delta_X
		
		for i,j in zip(subNDX, P):
			for a in i:
				preL.append(a * j) # calculate elements of expected Delta_X

		L.append(sum(preL)) # calculate the expected Delta_X
	for i in range(NQ):
		if Q[i] == 0:
			M[i+1].append((float(N-1)/k**2) *  np.exp(L[i]) ) # calculate the L(q,k) for q=0
		else:
			M[i+1].append((float(N-1)/k**2) *  L[i]**(1.0/Q[i]) ) # calculate the L(q,k)

	k = int(k*np.sqrt(np.sqrt(2))) + 1 # the increment of scale k is logarithmic (note that k <= fin, where fin is an input (line 34 of original version))

np.savetxt('MFH_P_'+str(PerI)+'-'+str(PerF)+'_'+name+'_ki'+str(ini)+'_kf'+str(fin)+'_Nbins'+str(Nbins)+'/PF_'+name+'.txt',np.matrix(M).transpose(),fmt='%s') # save the L(q,k) in the folder created

MHT=[[],[],[]]; MHT[0]=Q # MHT will contain info of hölder and tau function
for i in range(NQ):
	h = 2 + np.polyfit(np.log10(M[0]),np.log10(M[i+1]),1)[0] # hölder is calculated
	MHT[1].append(h);MHT[2].append(Q[i]*h-1) # hölder and tau function are set in MHT

np.savetxt('MFH_P_'+str(PerI)+'-'+str(PerF)+'_'+name+'_ki'+str(ini)+'_kf'+str(fin)+'_Nbins'+str(Nbins)+'/h_tau-spectrum_'+name+'.txt',np.matrix(MHT).transpose(),fmt='%s') # hölder and tau are saved in the folder created

Maf=[[],[]]
for k in range(1,NQ-1):
	a=( (MHT[2][k+1]-MHT[2][k-1])/(2*dq) ) # calculate alpha
	Maf[0].append(a); Maf[1].append(MHT[0][k]*a-MHT[2][k]) # calculate multifractal spectrum f(alpha)

np.savetxt('MFH_P_'+str(PerI)+'-'+str(PerF)+'_'+name+'_ki'+str(ini)+'_kf'+str(fin)+'_Nbins'+str(Nbins)+'/MF-spectrum_'+name+'.txt',np.matrix(Maf).transpose(),fmt='%s') # multifractal spectrum is saved in the folder created

# finally we create the figure of the results as figures 3 and 4 of our paper
sns.set_style('darkgrid')

grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.45)
plt.figure(figsize=(9.5, 6))

plt.subplot(grid[0,1])
plt.plot(MHT[0], MHT[1])
plt.xlabel('q', fontsize=18, fontstyle='italic')
plt.ylabel('h(q)', fontsize=18, fontstyle='italic')
plt.grid(True, which="both")

plt.subplot(grid[1,1])
plt.plot(MHT[0], MHT[2])
plt.xlabel('q', fontsize=18, fontstyle='italic')
plt.ylabel(r'$\tau$(q)', fontsize=18, fontstyle='italic')
plt.grid(True, which="both")

plt.subplot(grid[2,1])
plt.plot(Maf[0], Maf[1])
plt.xlabel(r'$\alpha$', fontsize=18, fontstyle='italic')
plt.ylabel(r'f($\alpha$)', fontsize=18, fontstyle='italic')
plt.grid(True, which="both")

plt.subplot(grid[0:, 0])
for i in range(len(M)-1):
	plt.plot(M[0], M[i+1])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Scale k', fontsize=18, fontstyle='italic')
plt.ylabel('Generalized length L(q,k)', fontsize=18, fontstyle='italic')
plt.grid(True, which="both")

plt.suptitle(r'MF-HDA of '+name+' with $p_r=$'+str(PerI)+'', fontsize=20, style='italic') # title is created with the name of the inputs

plt.savefig('MFH_P_'+str(PerI)+'-'+str(PerF)+'_'+name+'_ki'+str(ini)+'_kf'+str(fin)+'_Nbins'+str(Nbins)+'/'+name+'.jpg') # save figure
plt.show() # also this figure is shown
