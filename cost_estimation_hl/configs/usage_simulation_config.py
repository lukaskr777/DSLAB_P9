from dataclasses import dataclass
from abc import abstractmethod, ABC
import numpy as np
from pandas import Timestamp
from scipy.stats import lognorm, vonmises

@dataclass
class UsageSimulationCfg:
    days: int
    hour_peak: float
    num_users: int
    daily_prompts: int
    median_prompt_tokens: int
    start_date : Timestamp

    median_decode_prompts: int = None
    ## special params
    kappa : float = 0.65
    prompt_sigma: float = 1.08
    decode_sigma: float = 1.08


    def __post_init__(self):
        self.start_date = self.start_date.floor("D")





class CompletionTokenDist:


    def __init__(self,mu_c = 0.42 ,mu_b = 3.2,sig_c = -0.145,sig_b = 2.4, indep_sigma = None, indep_scale = None):
        self.mu_c = mu_c
        self.mu_b = mu_b
        self.sig_c = sig_c
        self.sig_b = sig_b

        self.indep_sigma = indep_sigma
        self.indep_scale = indep_scale

    def rvs(self, x):

        if self.indep_sigma == None:
            logx = np.log(x)
            sigma = np.maximum(self.sig_c *logx + self.sig_b,0.01)
            mu = np.exp(self.mu_c *logx + self.mu_b)
            dist = lognorm(s = sigma,scale = mu,loc = 0)

            return dist.rvs() ## check if it returns number
        
        else:
            return lognorm(s = np.array([self.indep_sigma]*len(x)), scale = np.array([self.indep_scale]*len(x)),loc = np.array([0]*len(x))).rvs()


class TimeDistribution:

    def __init__(self, hour_peak, kappa):
        self.kappa = kappa
        self.loc = 2*np.pi *hour_peak / 24

    
    def rvs(self, n ):

        angles = vonmises.rvs(kappa = self.kappa, loc=  self.loc, size = n)
        hours = (angles % (2*np.pi) * 24 / (2*np.pi))

        return hours
    
    def pdf(self, hours):
        """
        PDF evaluated at given hours (0-24)
        """
        angles = hours * 2*np.pi / 24
        return vonmises.pdf(angles, kappa=self.kappa, loc=self.loc)
    
