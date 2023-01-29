#!/usr/bin/env python
# coding: utf-8

# # **DAH PROJECT**

# 

# In[1]:


#Libraries used
import numpy as np
import matplotlib.pyplot as plt
import math
from iminuit import Minuit
from scipy import integrate
from scipy import stats
import pandas as pd
from scipy.optimize import curve_fit


# In[2]:


#Import data
f = open("ups-15.bin","r") 
datalist = np.fromfile(f,dtype=np.float32)


# In[3]:


#Creates the pandas data frame
nevent = int(len(datalist)/6)
xdata = np.split(datalist,nevent) 

cols = ["InvMass", "TransMomPair", "PseudoRapid", "MomPair", "TransMom1", "TransMom2"]
df = pd.DataFrame(xdata, columns = cols)


# In[4]:


# plot histogram of the data and save it into a .png
(n, bins, patches) = plt.hist(df['InvMass'], bins=1000, color = 'thistle', density = True)
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Candidates')
plt.title('Muon Pair Invariant Mass')
plt.savefig('WholeHist.png', dpi = 500);


# ## Cut the data (sideband substraction)

# #### Transverse momemntum muon pair

# In[5]:


#Cut the data depending on Transverse Momentum 
df = df[ (df['TransMomPair']>=0) & (df['TransMomPair']<=15) ]


# In[6]:


#s1 and s2 define sideband regions with just background 
#peak 1 is the location of the signal
peak1 = df[ (df['InvMass']>=9.3) & (df['InvMass']<=9.6) ]
s1 = df[ (df['InvMass']>=9.15) & (df['InvMass']<=9.3) ]
s2 = df[ (df['InvMass']>=9.6) & (df['InvMass']<=9.75) ]

#Create a 2D plot of Transverse Momentum against Invariant Mass
plt.hist2d(peak1['InvMass'], peak1['TransMomPair'], bins = 100, cmap = plt.cm.CMRmap)
plt.ylabel("Transverse Momentum of Muon Pair [GeV/c]")
plt.xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
plt.colorbar()
plt.savefig('2DHistTransMom.png', dpi = 500);


# In[7]:


obs = plt.hist(peak1['TransMomPair'], bins = 20, alpha = 0.5)
bck1 = plt.hist(s1['TransMomPair'], bins = 20, alpha = 0.5)
bck2 =plt.hist(s2['TransMomPair'], bins = 20, alpha = 0.5);

n = obs[0] - bck1[0] - bck2[0]


# In[8]:


#Using a sideband subtraction, plot signal, background, and the original distribution as function of transverse momentum
plt.plot(obs[1][1:], n, 'bx', label = 'signal')
plt.plot(obs[1][1:,], bck1[0]+bck2[0], 'rx', label = 'Bck')
plt.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
plt.xlabel('Muon Pair Transverse Momentum (GeV/c)')
plt.ylabel('Candidates')
plt.legend()
plt.savefig('SidebandsTransMom.png', dpi = 500);


# In[ ]:





# #### Pseudorapidity

# In[9]:


#Identical process as above however for Pseudorapidity
df = df[ (df['PseudoRapid']>=2) & (df['PseudoRapid']<=6) ]


# In[10]:


peak1 = df[ (df['InvMass']>=9.3) & (df['InvMass']<=9.6) ]
s1 = df[ (df['InvMass']>=9.15) & (df['InvMass']<=9.3) ]
s2 = df[ (df['InvMass']>=9.6) & (df['InvMass']<=9.75) ]

plt.hist2d(peak1['InvMass'], peak1['PseudoRapid'], bins = 100, cmap = plt.cm.CMRmap)
plt.ylabel("Pseudorapidity of Muon Pair")
plt.xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
plt.colorbar();


# In[11]:


obs = plt.hist(peak1['PseudoRapid'], bins = 20, alpha = 0.5)
bck1 = plt.hist(s1['PseudoRapid'], bins = 20, alpha = 0.5)
bck2 =plt.hist(s2['PseudoRapid'], bins = 20, alpha = 0.5);

n = obs[0] - bck1[0] - bck2[0]


# In[12]:


plt.plot(obs[1][1:], n, 'bx', label = 'signal')
plt.plot(obs[1][1:,], bck1[0]+bck2[0], 'rx', label = 'Bck')
plt.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
plt.xlabel('Pseudorapidity (GeV/c)')
plt.ylabel('Candidates')
plt.legend();


# In[ ]:





# #### Mom Pair

# In[13]:


#Identical process as above however for the total momentum of the muon pair
df = df[ (df['MomPair']>=0) & (df['MomPair']<=300) ]


# In[14]:


peak1 = df[ (df['InvMass']>=9.3) & (df['InvMass']<=9.6) ]
s1 = df[ (df['InvMass']>=9.15) & (df['InvMass']<=9.3) ]
s2 = df[ (df['InvMass']>=9.6) & (df['InvMass']<=9.75) ]

plt.hist2d(peak1['InvMass'], peak1['MomPair'], bins = 100, cmap = plt.cm.CMRmap)
plt.ylabel("(Total) Momentum of Muon Pair [GeV/c]")
plt.xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
plt.colorbar();


# In[15]:


obs = plt.hist(peak1['MomPair'], bins = 20, alpha = 0.5)
bck1 = plt.hist(s1['MomPair'], bins = 20, alpha = 0.5)
bck2 =plt.hist(s2['MomPair'], bins = 20, alpha = 0.5);

n = obs[0] - bck1[0] - bck2[0]


# In[16]:


plt.plot(obs[1][1:], n, 'bx', label = 'signal')
plt.plot(obs[1][1:,], bck1[0]+bck2[0], 'rx', label = 'Bck')
plt.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
plt.xlabel('(Total) Momentum of Muon Pair (GeV/c)')
plt.ylabel('Candidates')
plt.legend();


# In[ ]:





# #### Trans Mom 1 

# In[17]:


#Identical process as above however for the transverse momentum of the first muon
df = df[ (df['TransMom1']>=1) & (df['TransMom1']<=10) ]


# In[18]:


peak1 = df[ (df['InvMass']>=9.3) & (df['InvMass']<=9.6) ]
s1 = df[ (df['InvMass']>=9.15) & (df['InvMass']<=9.3) ]
s2 = df[ (df['InvMass']>=9.6) & (df['InvMass']<=9.75) ]

plt.hist2d(peak1['InvMass'], peak1['TransMom1'], bins = 100, cmap = plt.cm.CMRmap)
plt.ylabel("Transverse Momentum of First Muon [GeV/c]")
plt.xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
plt.colorbar();


# In[19]:


obs = plt.hist(peak1['TransMom1'], bins = 20, alpha = 0.5)
bck1 = plt.hist(s1['TransMom1'], bins = 20, alpha = 0.5)
bck2 =plt.hist(s2['TransMom1'], bins = 20, alpha = 0.5);

n = obs[0] - bck1[0] - bck2[0]


# In[20]:


plt.plot(obs[1][1:], n, 'bx', label = 'signal')
plt.plot(obs[1][1:,], bck1[0]+bck2[0], 'rx', label = 'Bck')
plt.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
plt.xlabel('First Muon Transverse Momentum (GeV/c)')
plt.ylabel('Candidates')
plt.legend();


# In[ ]:





# #### Trans Mom 2

# In[21]:


#Identical process as above however for the transverse momentum of the second muon
df = df[ (df['TransMom2']>=2) & (df['TransMom2']<=10) ]


# In[22]:


peak1 = df[ (df['InvMass']>=9.3) & (df['InvMass']<=9.6) ]
s1 = df[ (df['InvMass']>=9.15) & (df['InvMass']<=9.3) ]
s2 = df[ (df['InvMass']>=9.6) & (df['InvMass']<=9.75) ]

plt.hist2d(peak1['InvMass'], peak1['TransMom2'], bins = 100, cmap = plt.cm.CMRmap)
plt.ylabel("Transverse Momentum of Second Muon [GeV/c]")
plt.xlabel("Invariant Mass of Muon Pair [GeV/c^2]")
plt.colorbar();


# In[23]:


obs = plt.hist(peak1['TransMom2'], bins = 20, alpha = 0.5)
bck1 = plt.hist(s1['TransMom2'], bins = 20, alpha = 0.5)
bck2 =plt.hist(s2['TransMom2'], bins = 20, alpha = 0.5);

n = obs[0] - bck1[0] - bck2[0]


# In[24]:


plt.plot(obs[1][1:], n, 'bx', label = 'signal')
plt.plot(obs[1][1:,], bck1[0]+bck2[0], 'rx', label = 'Bck')
plt.plot(obs[1][1:], obs[0], 'gx', label = 'Data')
plt.xlabel('Second Muon Transverse Momentum (GeV/c)')
plt.ylabel('Candidates')
plt.legend();


# In[25]:


# plot mass histogram of cut data and save it into a .png
(n, bins, patches) = plt.hist(df['InvMass'], bins=1000, color = 'thistle', density = True)
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Candidates')
plt.title('Muon Pair Invariant Mass')
plt.savefig('WholeHistcut.png', dpi = 500);


# In[26]:


#Isolate the first peak
peak1 = df[ (df['InvMass']>=9) & (df['InvMass']<=9.7) ]


# ## Functions of normalized PDFs

# In[27]:


def pdf_comb(x, F, mu, sigma, tau):
    
    """
    Probability distribution function (pdf) of a gaussian and exponential 
    """
    
    a = 9
    b = 9.7
    
    gaussian = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-mu)**2 / (2*sigma**2)) 
    exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

    
    return F*(exponential) + (1-F)* ( gaussian )


# ## Likelihood functions

# In[28]:


# combined pdf likelihood
def likelihood(F, mu, sigma, tau):
    
    """
    Likelihood function of gaussian and exponential fit
    """
    
    return -np.sum(np.log(pdf_comb(peak1['InvMass'], F, mu, sigma, tau)))


# In[29]:


#determine optimal parameters for fitting of first peak
m_comb = Minuit(likelihood, F = 0.6, mu = 9.45, sigma = 0.3, tau = 1.6)

m_comb.migrad()  # run optimiser
m_comb.hesse()   # run covariance estimator

print(m_comb.values)  # print estimated values
print(m_comb.errors)  


# In[30]:


#Plot the mass histogram, fitted function, and signal distribution

n1, bins1, patches1 = plt.hist(peak1['InvMass'], bins = 1000, color = 'thistle', density = True, label = 'Mass Histogram');

y_fit = pdf_comb(bins1[:-1], *m_comb.values)

plt.plot(bins1[:-1], y_fit, color = "darkBlue", label = 'Fitted Function')
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Candidates')
plt.title('Muon Pair Invariant Mass for Y(1S) peak')

gaussian = (1-m_comb.values[0])*(1/(m_comb.values[2]*np.sqrt(2*np.pi))) * np.exp(-(bins1[:-1]-m_comb.values[1])**2 / (2*m_comb.values[2]**2))

plt.plot(bins1[:-1], gaussian, color = 'red', linestyle = '--', label = 'Signal Distribution')
plt.legend()
plt.savefig('Peak1.png', dpi = 500);


# In[31]:


#Plot the residuals

residuals = y_fit - n1
plt.plot(bins1[:-1], residuals, color = 'slateblue', marker = '.', linestyle = 'None')
plt.axhline(y=0, color='Grey', linestyle='-')
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Residuals')
plt.savefig('peak1res.png', dpi = 500);


# In[32]:


#Cut the data to isolate a region with just background

bgn = df[ (df['InvMass']>=10.5) ]


# In[33]:


def pdf_bgn(x, tau):
    
    """
    Pdf with just an exponential curve
    """
    
    a = float(bgn['InvMass'].min())
    b = float(bgn['InvMass'].max())
    
    
    return (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )


# In[34]:


# combined pdf likelihood
def likelihood_bgn(tau):
    
    """
    Likelihood function for the exponential fit 
    """
    
    return -np.sum(np.log(pdf_bgn(np.array(bgn["InvMass"]),tau)))


# In[35]:


#Determine optimal parameters for the expoential fit 

m_comb_bgn = Minuit(likelihood_bgn, tau = 1.76)

m_comb_bgn.migrad()  # run optimiser
m_comb_bgn.hesse()   # run covariance estimator

print(m_comb_bgn.values)  # print estimated values
print(m_comb_bgn.errors)  


# In[36]:


#Plot the background mass distribution as well as the fitted exponential curve

(n_bgn, bins_bgn, patches) = plt.hist(bgn['InvMass'], bins = 1000 ,color = "thistle", density = True, label = 'Mass Histogram')

y_bgn = pdf_bgn(bins_bgn[:-1], *m_comb_bgn.values)

plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Candidates')
plt.title('Muon Pair Invariant Mass (background)')
plt.plot(bins_bgn[:-1], y_bgn, color = 'darkblue', label = 'Fitted Background Exponential')
plt.savefig('bck.png', dpi = 500);


# # Full fit

# In[37]:


def pdf_tot2(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3):
    
    """
    pdf function describing the entire mass range of interest. Each of the three Upsilon peaks are described by a 
    gaussian and the background is described by an exponential curve
    
    Free parameters are the Upsilon peak means, and the first peak resolution. The two remaining peak resolutions are related
    to the original peak resolution
    """
    
    a = df['InvMass'].min()
    b = df['InvMass'].max()
    
    tau = m_comb_bgn.values[0]
    s_2 = s_1 * mu_2/mu_1
    s_3 = s_1 * mu_3/mu_1
    
    # one gaussian PDF per mass peak 
    g_1 = (1/(s_1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*s_1**2)) 
    g_2 = (1/(s_2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*s_2**2)) 
    g_3 = (1/(s_3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*s_3**2)) 
    
    # exponential pdf for background    
    exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

    
    return F1*(exponential) + (1 - F1 - F2 - F3)* g_1  + F2*g_2 + F3*g_3


# In[38]:


def pdf_tot2_for_plot(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3):
    
    """
    This function is used to provide the signal distribution so that it may be plotted
    """
    
    a = df['InvMass'].min()
    b = df['InvMass'].max()
    
    tau = m_comb_bgn.values[0]
    s_2 = s_1 * mu_2/mu_1
    s_3 = s_1 * mu_3/mu_1
    
    # one gaussian PDF per mass peak 
    g_1 = (1/(s_1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*s_1**2)) 
    g_2 = (1/(s_2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*s_2**2)) 
    g_3 = (1/(s_3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*s_3**2)) 
    
    # exponential pdf for background    
    exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

    
    return (1 - F1 - F2 - F3)* g_1  + F2*g_2 + F3*g_3


# In[39]:


# combined pdf likelihood
def likelihood_tot2(F1, F2, F3, s_1, mu_1, mu_2, mu_3):
    
    """
    Likelihood function for the triple gaussian fit 
    """
    
    return -np.sum(np.log(pdf_tot2(df["InvMass"], F1, F2, F3, s_1, mu_1, mu_2, mu_3)))


# In[40]:


#Determine the optimal parameters for the full fit

m_comb_tot2 = Minuit(likelihood_tot2, F1 = 0.6, F2 = 0.15, F3 = 0.15, s_1 = 0.05, mu_1 = 9.35, mu_2=10, mu_3=10.4)

m_comb_tot2.migrad()  # run optimiser
m_comb_tot2.hesse()   # run covariance estimator

print(m_comb_tot2.values)  # print estimated values
print(m_comb_tot2.errors)  


# In[41]:


# plot the mass histogram, fitted function, and signal distribution
(nfull, binsfull, patchesfull) = plt.hist(df['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Candidates')
plt.title('Triple Gaussian Fit')

x = binsfull[:-1]

y_tot2 = pdf_tot2(x, *m_comb_tot2.values)
plt.plot(x, y_tot2, color = 'darkblue', label = 'Fitted Function')

signl = pdf_tot2_for_plot(x, *m_comb_tot2.values)

plt.plot(x, signl, color = 'red', linestyle = '--', label = 'Signal Distribution')
plt.legend()
plt.savefig('FUllFitGaus.png', dpi = 500);


# In[42]:


#plot the residuals 

residuals = y_tot2 - nfull
plt.plot(binsfull[:-1], residuals, color = 'slateblue', marker = '.', linestyle = 'None')
plt.axhline(y=0, color='Grey', linestyle='-')
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Residuals')
plt.savefig('FullfitGausres.png', dpi = 500);


# # Monte Carlo Data to find crystall ball params

# In[43]:


#import the monte carlo simulated first peak distribution
f = open("mc.bin","r") 
datalist = np.fromfile(f,dtype=np.float32)


# In[44]:


#Creat data frame for the monte carlo data

nevent = int(len(datalist)/6)
xdata = np.split(datalist,nevent) 

cols = ["InvMass", "TransMomPair", "Rapid", "MomPair", "TransMom1", "TransMom2"]
df_mc = pd.DataFrame(xdata, columns = cols)


# In[45]:


mass_min = df_mc['InvMass'].min()
mass_max = df_mc['InvMass'].max()


# In[46]:


def pdf_cb(x, beta, loc, scale, sigma):
    
    """
    Pdf function describing the first Upsilon peak using a gaussian and crystal function that share a mean
    """
    
    m = 1.0001
    F = 0.75
    
    function = lambda x: stats.crystalball.pdf(x, beta, m, loc, scale)
    area = integrate.quad(function, mass_min, mass_max)[0]
    
    crystal_ball = stats.crystalball.pdf(x, beta, m, loc, scale) / area
    gaussian = (1/(sigma*np.sqrt(2*np.pi))) * np.exp(-(x-loc)**2 / (2*sigma**2))  

    
    return F*crystal_ball + (1-F)*gaussian


# In[47]:


def like_cb(beta, loc, scale, sigma):
    
    """
    Likelihood function for the monte carlo fit
    """
    
    
    return -np.sum(np.log(pdf_cb(df_mc["InvMass"], beta, loc, scale, sigma)))


# In[48]:


#Determine optimal parameter for the crystal ball + gaussian fit 

m_comb_mc = Minuit(like_cb, beta = 2, loc = 9.45, scale = 0.04, sigma = 0.02)

m_comb_mc.migrad()  # run optimiser
m_comb_mc.hesse()   # run covariance estimator

print(m_comb_mc.values)  # print estimated values
print(m_comb_mc.errors) 


# In[49]:


#Plot the mass histogram and fitted function

(n_mc, bins_mc, patches_mc) = plt.hist(df_mc['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Candidates')
plt.title('Muon Pair Invariant Mass')
plt.plot(bins_mc[:-1], pdf_cb(bins_mc[:-1], *m_comb_mc.values), color = 'darkblue', label = 'Fitted Function')
plt.legend()
plt.savefig('MC.png', dpi = 500);


# # Crystal Ball func

# In[50]:


def pdf_crys_all(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3, sigma1):
    
    """
    Pdf function to describe the entire mass distribution. Each of the three upsilon peaks are described by a gaussian and 
    crystal ball function that share a mean. As before, the background is described by the same exponential
    
    As before, the resolutions of the second and third peaks are related to the resolution of the first peak, which is kept 
    floating in the fit 
    """
    
    a = df['InvMass'].min()
    b = df['InvMass'].max()
    
    tau = m_comb_bgn.values[0]
    s_2 = s_1 * mu_2/mu_1
    s_3 = s_1 * mu_3/mu_1
    
    beta = m_comb_mc.values[0]
    m = 1.0001
    
    function1 = lambda x: stats.crystalball.pdf(x, beta, m, mu_1, s_1)
    function2 = lambda x: stats.crystalball.pdf(x, beta, m, mu_2, s_2)
    function3 = lambda x: stats.crystalball.pdf(x, beta, m, mu_3, s_3)
    
    area1 = integrate.quad(function1, a, b)[0]
    area2 = integrate.quad(function2, a, b)[0]
    area3 = integrate.quad(function3, a, b)[0]
    
    sigma2 = sigma1 * mu_2/mu_1
    sigma3 = sigma1 * mu_3/mu_1
    
    # one crystall ball PDF per mass peak 
    cb1 = stats.crystalball.pdf(x, beta, m, mu_1, s_1)/area1
    g1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*sigma1**2))  
    
    cb2 = stats.crystalball.pdf(x, beta, m, mu_2, s_2)/area2 
    g2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*sigma2**2))  
    
    cb3 = stats.crystalball.pdf(x, beta, m, mu_3, s_3)/area3
    g3 = (1/(sigma3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*sigma3**2))  
    
    
    # exponential pdf for background    
    exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

    
    return F1*(exponential) + F2*(0.75*cb1 + 0.25*g1) + (1 - F1 - F2 - F3)*(0.75*cb2 + 0.25*g2) + F3*(0.75*cb3 + 0.25*g3)


# In[51]:


def pdf_for_plot(x, F1, F2, F3, s_1, mu_1, mu_2, mu_3, sigma1):
    
    """
    Function used to obtain the signal distribution which is then plotted
    """
    
    a = df['InvMass'].min()
    b = df['InvMass'].max()
    
    tau = m_comb_bgn.values[0]
    s_2 = s_1 * mu_2/mu_1
    s_3 = s_1 * mu_3/mu_1
    
    beta = m_comb_mc.values[0]
    m = 1.0001
    
    function1 = lambda x: stats.crystalball.pdf(x, beta, m, mu_1, s_1)
    function2 = lambda x: stats.crystalball.pdf(x, beta, m, mu_2, s_2)
    function3 = lambda x: stats.crystalball.pdf(x, beta, m, mu_3, s_3)
    
    area1 = integrate.quad(function1, a, b)[0]
    area2 = integrate.quad(function2, a, b)[0]
    area3 = integrate.quad(function3, a, b)[0]
    
    sigma2 = sigma1 * mu_2/mu_1
    sigma3 = sigma1 * mu_3/mu_1
    
    # one crystall ball PDF per mass peak 
    cb1 = stats.crystalball.pdf(x, beta, m, mu_1, s_1)/area1
    g1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*sigma1**2))  
    
    cb2 = stats.crystalball.pdf(x, beta, m, mu_2, s_2)/area2 
    g2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*sigma2**2))  
    
    cb3 = stats.crystalball.pdf(x, beta, m, mu_3, s_3)/area3
    g3 = (1/(sigma3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*sigma3**2))  
    
    
    # exponential pdf for background    
    exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )

    
    return F2*(0.75*cb1 + 0.25*g1) + (1 - F1 - F2 - F3)*(0.75*cb2 + 0.25*g2) + F3*(0.75*cb3 + 0.25*g3)


# In[52]:


def like_cb(F1, F2, F3, s_1, mu_1, mu_2, mu_3, sigma1):
    
    """
    Likelihood function for full fit using gaussian and crystal ball for each peak
    """
    
    return -np.sum(np.log(pdf_crys_all(df["InvMass"], F1, F2, F3, s_1, mu_1, mu_2, mu_3, sigma1)))


# In[53]:


#Determine optimal paramaters for the full gaussian+crystal ball fit

m_comb_crys = Minuit(like_cb, F1 = 0.82, F2 = 0.12, F3 = 0.013, s_1 = 0.04, mu_1 = 9.45, mu_2=10.01, mu_3=10.35, sigma1 = 0.02)

m_comb_crys.migrad()  # run optimiser
m_comb_crys.hesse()   # run covariance estimator

print(m_comb_crys.values)  # print estimated values
print(m_comb_crys.errors)  


# In[54]:


#Plot the Mass dstribution, fitted function, and signal distribution

(n_cry, bins_cry, patches_cry) = plt.hist(df['InvMass'], bins=1000, color = 'thistle', density = True, label = 'Mass Histogram')
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Candidates')
plt.title('Triple Composite Fit')


y_tot2_cry = pdf_crys_all(bins_cry[:-1], *m_comb_crys.values)
plt.plot(bins_cry[:-1], y_tot2_cry, color = 'darkblue', label = 'Fitted Function')

signal = pdf_for_plot(bins_cry[:-1], *m_comb_crys.values)
plt.plot(bins_cry[:-1], signal , color = 'red', linestyle = '--', label = 'Signal Distribution')
plt.legend()
plt.savefig('fullcrystalfit.png', dpi = 500);


# In[55]:


#Plot the residuals

residuals_cry = y_tot2_cry - n_cry
plt.plot(bins[:-1], residuals_cry, color = 'slateblue', marker = '.', linestyle = 'None')
plt.axhline(y=0, color='Grey', linestyle='-')
plt.xlabel('Mass (GeV/c^2)')
plt.ylabel('Residuals')
plt.savefig('fullcrystalfitres.png', dpi = 500);


# # Systematic erros due to the signal model

# systematic error on the masses

# In[56]:


sysmu1 = np.abs( m_comb_crys.values[4] - m_comb_tot2.values[4])
sysmu2 = np.abs( m_comb_crys.values[5] - m_comb_tot2.values[5])
sysmu3 = np.abs( m_comb_crys.values[6] - m_comb_tot2.values[6])


# In[57]:


print(sysmu1)
print(sysmu2)
print(sysmu3)


# In[58]:


totmu1 = np.sqrt( sysmu1**2 + m_comb_crys.errors[4]**2)
totmu2 = np.sqrt( sysmu2**2 + m_comb_crys.errors[5]**2)
totmu3 = np.sqrt( sysmu3**2 + m_comb_crys.errors[6]**2)


# In[59]:


print(totmu1)
print(totmu2)
print(totmu3)


# "Final" values for signal masses

# Results presented as  M = Value +- Etot

# In[60]:


print("S1 =  " + str(m_comb_crys.values[4]) + "+/-" + str(totmu1) )
print("S2 =  " + str(m_comb_crys.values[5]) + "+/-" + str(totmu2) )
print("S3 =  " + str(m_comb_crys.values[6]) + "+/-" + str(totmu3) )


# In[ ]:





# # Signal yields

# In[61]:


df_pbin = df[ (df['PseudoRapid']>=2.5) & (df['PseudoRapid']<=5.5) ]


# In[62]:


def bin_anal ( dfpbin ): 
    
    """
    Performs the crystal ball composite fit for 13 bins of transverse momentum in order to compare parameters
    """
    
    param = []
    
    for i in range(13):
    
        df_cuta = dfpbin[ (dfpbin['TransMomPair']>=(0 + i)) & (dfpbin['TransMomPair']<=(1 + i)) ]
        
        def pdf_crys_bina(x, F1, F2, F3, s_1, sigma1):
    
            a = x.min()
            b = x.max()

            
            mu_1 = m_comb_crys.values[4]
            mu_2 = m_comb_crys.values[5]
            mu_3 = m_comb_crys.values[6]

            tau = m_comb_bgn.values[0]
            s_2 = s_1 * mu_2/mu_1
            s_3 = s_1 * mu_3/mu_1

            beta = m_comb_mc.values[0]
            m = 1.0001

            function1 = lambda x: stats.crystalball.pdf(x, beta, m, mu_1, s_1)
            function2 = lambda x: stats.crystalball.pdf(x, beta, m, mu_2, s_2)
            function3 = lambda x: stats.crystalball.pdf(x, beta, m, mu_3, s_3)

            area1 = integrate.quad(function1, a, b)[0]
            area2 = integrate.quad(function2, a, b)[0]
            area3 = integrate.quad(function3, a, b)[0]

            sigma2 = sigma1 * mu_2/mu_1
            sigma3 = sigma1 * mu_3/mu_1

            # one crystall ball PDF per mass peak 
            cb1 = stats.crystalball.pdf(x, beta, m, mu_1, s_1)/area1
            g1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*sigma1**2))  

            cb2 = stats.crystalball.pdf(x, beta, m, mu_2, s_2)/area2 
            g2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*sigma2**2))  

            cb3 = stats.crystalball.pdf(x, beta, m, mu_3, s_3)/area3
            g3 = (1/(sigma3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*sigma3**2))  


            # exponential pdf for background    
            exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )


            return F1*(exponential) + F2*(0.75*cb1 + 0.25*g1) + (1 - F1 - F2 - F3)*(0.75*cb2 + 0.25*g2) + F3*(0.75*cb3 + 0.25*g3)


        def like_bina(F1, F2, F3, s_1, sigma1):
    
            return -np.sum(np.log(pdf_crys_bina(df_cuta["InvMass"], F1, F2, F3, s_1, sigma1)))
    
        m_comb_bina = Minuit(like_bina, F1 = 0.82, F2 = 0.12, F3 = 0.013, s_1 = 0.04, sigma1 = 0.02)

        m_comb_bina.migrad()  # run optimiser
        m_comb_bina.hesse()   # run covariance estimator

        param.extend([m_comb_bina.values[0],m_comb_bina.errors[0] , m_comb_bina.values[1],m_comb_bina.errors[1], m_comb_bina.values[2],m_comb_bina.errors[2], m_comb_bina.values[3], m_comb_bina.values[4]])

    return param


# In[63]:


param = np.array (bin_anal ( df_pbin ))


# In[64]:


# create new data frame for transverse momentum bin analysis
nevent = int(len(param)/8)
xdata = np.split(param,nevent) 

cols = ["F1", "eF1", "F2", "eF2", "F3", "eF3", "s_1", "sigma1"]
frameBIN = pd.DataFrame(xdata, columns = cols)


# In[65]:


#Ratios of second and third peak yield with respect to first peak yield

ratios = []

for i in range(13):
    
    parameters = np.array(frameBIN.iloc[i].tolist())
    F1 = parameters[0]
    eF1 = parameters[1]
    F2 = parameters[2]
    eF2 = parameters[3]
    F3 = parameters[4]
    eF3 = parameters[5]
    
    F4 = 1 - F1 - F2 - F3
    eF4 = np.sqrt(eF1**2 + eF2**2 + eF3**2)
    
    
    #Errors calculated using propogation of errors
    cros2 = F4/F2
    eC2 =   cros2 * np.sqrt( (eF4/F4)**2 + (eF2/F2)**2 )
    cros3 = F3/F2
    eC3 =   cros3 * np.sqrt( (eF4/F4)**2 + (eF3/F3)**2 )

    ratios.extend([cros2, eC2, cros3, eC3])
    

ratios = np.array(ratios)


# In[66]:


#Data frame to contain ratio data and errors
ratio = int(len(ratios)/4)
ratio_data = np.split(ratios,ratio) 

cols = ["cros2", "eC2", "cros3", "eC3"]
dfR = pd.DataFrame(ratio_data, columns = cols)


# In[67]:


x = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])


# In[68]:


plt.errorbar(x , dfR["cros2"], yerr= dfR["eC2"], color = 'Blue', label = 'Y(2S)/Y(1S)', fmt = '+')
plt.errorbar(x , dfR["cros3"], yerr= dfR["eC3"], color = 'fuchsia', label = 'Y(3S)/Y(1S)', fmt = '+')
plt.xlabel('Transverse Momentum (GeV/c)', fontsize = 12)
plt.ylabel(r'$R^{iS/1S}$', fontsize = 12)
plt.text(6.5, 0.04,'2.5 < ' r'$\eta$' ' < 5.5', fontsize = 14)
plt.legend( fontsize = 12)
plt.savefig('BinsPT.png', dpi = 500);


# # Pseudorapidity bins 

# In[ ]:





# In[69]:


df_pbin2 = df[ (df['TransMomPair']>=0) & (df['TransMomPair']<=15) ]


# In[70]:


def bin_rap ( dfpbin ): 
    
    """
    Performs the crystal ball composite fit for 5 bins of pseudorapidity in order to compare parameters
    """
    
    param = []
    
    for i in range(5):
    
        df_cuta = dfpbin[ (dfpbin['PseudoRapid']>=(2.5 + 0.5 * i)) & (dfpbin['PseudoRapid']<=(3.0 + 0.5*i)) ]
        
        def pdf_crys_bina(x, F1, F2, F3, s_1, sigma1):
    
            a = x.min()
            b = x.max()

            #pr = 1 + 2.5 + 0.5 * i 
            
            mu_1 = m_comb_crys.values[4]
            mu_2 = m_comb_crys.values[5]
            mu_3 = m_comb_crys.values[6]

            tau = m_comb_bgn.values[0]
            #s_1 = pr * s_1
            s_2 = s_1 * mu_2/mu_1
            s_3 = s_1 * mu_3/mu_1

            beta = m_comb_mc.values[0]
            m = 1.0001

            function1 = lambda x: stats.crystalball.pdf(x, beta, m, mu_1, s_1)
            function2 = lambda x: stats.crystalball.pdf(x, beta, m, mu_2, s_2)
            function3 = lambda x: stats.crystalball.pdf(x, beta, m, mu_3, s_3)

            area1 = integrate.quad(function1, a, b)[0]
            area2 = integrate.quad(function2, a, b)[0]
            area3 = integrate.quad(function3, a, b)[0]

            sigma2 = sigma1 * mu_2/mu_1
            sigma3 = sigma1 * mu_3/mu_1

            # one crystall ball PDF per mass peak 
            cb1 = stats.crystalball.pdf(x, beta, m, mu_1, s_1)/area1
            g1 = (1/(sigma1*np.sqrt(2*np.pi))) * np.exp(-(x-mu_1)**2 / (2*sigma1**2))  

            cb2 = stats.crystalball.pdf(x, beta, m, mu_2, s_2)/area2 
            g2 = (1/(sigma2*np.sqrt(2*np.pi))) * np.exp(-(x-mu_2)**2 / (2*sigma2**2))  

            cb3 = stats.crystalball.pdf(x, beta, m, mu_3, s_3)/area3
            g3 = (1/(sigma3*np.sqrt(2*np.pi))) * np.exp(-(x-mu_3)**2 / (2*sigma3**2))  


            # exponential pdf for background    
            exponential = (1/tau)*np.exp(-x/tau) / (np.exp(-a/tau) - np.exp(-b/tau) )


            return F1*(exponential) + F2*(0.75*cb1 + 0.25*g1) + (1 - F1 - F2 - F3)*(0.75*cb2 + 0.25*g2) + F3*(0.75*cb3 + 0.25*g3)


        def like_bina(F1, F2, F3, s_1, sigma1):
    
            return -np.sum(np.log(pdf_crys_bina(df_cuta["InvMass"], F1, F2, F3, s_1, sigma1)))
    
        m_comb_bina = Minuit(like_bina, F1 = 0.82, F2 = 0.12, F3 = 0.013, s_1 = 0.04, sigma1 = 0.02)

        m_comb_bina.migrad()  # run optimiser
        m_comb_bina.hesse()   # run covariance estimator
  

        param.extend([m_comb_bina.values[0], m_comb_bina.errors[0], m_comb_bina.values[1], m_comb_bina.errors[1], m_comb_bina.values[2], m_comb_bina.errors[2], m_comb_bina.values[3], m_comb_bina.errors[3], m_comb_bina.values[4], m_comb_bina.errors[4]])

    return param


# In[71]:


param_rapid = np.array(bin_rap ( df_pbin2 ))


# In[72]:


# create new data frame for pseudorapidity bin analysis
nevent_rapid = int(len(param_rapid)/10)
xdata_rapid = np.split(param_rapid,nevent_rapid) 

cols = ["F1", "error F1",  "F2", "error F2", "F3", "error F3", "s_1", "error s_1", "sigma1", "error sigma1"]
frameBINrapid = pd.DataFrame(xdata_rapid, columns = cols)


# In[73]:


#Ratios of second and third peak yield with respect to first peak yield

ratios_rapid = []

for i in range(5):
    
    parameters = np.array(frameBINrapid.iloc[i].tolist())
    F1 = parameters[0]
    eF1 = parameters[1]
    F2 = parameters[2]
    eF2 = parameters[3]
    F3 = parameters[4]
    eF3 = parameters[5]
    
    #Errors calculated using propogation of errors
    F4 = 1 - F1 - F2 - F3
    eF4 = np.sqrt(eF1**2 + eF2**2 + eF3**2)

    cros2 = F4/F2
    eC2 =   cros2 * np.sqrt( (eF4/F4)**2 + (eF2/F2)**2 )
    
    cros3 = F3/F2
    eC3 = cros3*   np.sqrt( (eF4/F4)**2 + (eF3/F3)**2 )

    ratios_rapid.extend([cros2, eC2, cros3, eC3])
    
    #print(cros2)

ratios_rapid = np.array(ratios_rapid)


# In[74]:


#data frame to contain ratios and errors
ratio_rapid = int(len(ratios_rapid)/4)
ratio_datarapid = np.split(ratios_rapid,ratio_rapid) 

cols = ["cros2", "errorC2", "cros3", "errorC3"]
dfR_rapid = pd.DataFrame(ratio_datarapid, columns = cols)


# In[75]:


x1 = np.array([2.75, 3.25, 3.75, 4.25, 4.75])


# In[76]:


plt.errorbar(x1 , dfR_rapid["cros2"], yerr= dfR_rapid["errorC2"], color = 'Blue', label = 'Y(2S)/Y(1S)', fmt = '+')
plt.errorbar(x1 , dfR_rapid["cros3"], yerr= dfR_rapid["errorC3"], color = 'fuchsia', label = 'Y(3S)/Y(1S)', fmt = '+')
plt.xlabel('Pseudorapidity', fontsize = 12)
plt.ylabel(r'$R^{iS/1S}$',fontsize = 12 )
plt.legend(loc = [0.6, 0.4], fontsize = 12)
plt.text(4.3, 0.19, r'$p_T$' ' < 15', fontsize = 14)
plt.savefig('BinsPseudo.png', dpi = 500);


# In[ ]:




