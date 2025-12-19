from dataclasses import dataclass
from abc import abstractmethod, ABC
import numpy as np
from scipy.stats import lognorm, vonmises
import json
from typing import Callable



@dataclass
class ComputeSimulationCfg:
    cluster_cfg_file: str
    batch_size : int
    dynamic_lim_sec: float
    util_f: Callable = None

    cluster_cfgs : list = None
    def __post_init__(self):
        if self.util_f is None:
            self.util_f = lambda b: min(1,b/float(self.batch_size))

        if self.cluster_cfgs == None:
            self.cluster_cfgs = [ComputeCluster(**cfg) for cfg in json.load(open(self.cluster_cfg_file,"r"))["configurations"]]



@dataclass
class ComputeCluster:
    name: str
    num_replicas: int 
    num_gpu_per_replica: int
    gpu_vram_B: float
    model_size_B: float
    peak_prefill_tks_per_sec_per_replica: int
    peak_decode_tks_per_sec_per_replica: int
    price_per_hour: float
    is_price_monthly: bool = True  # True because it usually monthly

    def __post_init__(self):
        # Check if model fits
        if self.num_gpu_per_replica * self.gpu_vram_B < self.model_size_B:
            raise Exception(f"Model of size {self.model_size_B/1e9} GB cannot fit into replica with {self.num_gpu_per_replica*self.gpu_vram_B/1e9} GB VRAM")
        
        print(f"Model {self.name} instantiated:")
        print(f" - {self.num_replicas} replicas",sep=" ")
        print(f" - Cluster price per hour: {self.price_per_hour:.2f}",sep=" ")

    def peak_tks_per(self, days = 0, hours = 0, minutes = 0, seconds = 0,prefill_R = 1):

        total_sec =(days * 24 * 60 * 60 + hours * 60 * 60 + minutes * 60 + seconds) * self.num_replicas
        prefill = total_sec * self.peak_prefill_tks_per_sec_per_replica
        decode = total_sec * self.peak_decode_tks_per_sec_per_replica
        
        time_ratio = prefill_R * decode / ((1 - prefill_R) * prefill + prefill_R * decode)
        return  prefill * time_ratio, decode * (1 - time_ratio)
    
    def price_per(self,days = 0, hours = 0, minutes = 0, seconds =  0):
        total_sec =(days * 24 * 60 * 60 + hours * 60 * 60 + minutes * 60 + seconds)
        return total_sec * self.price_per_hour/(60*60)


@dataclass
class ComputeAPI:
    name: str
    price_per_prefill_tkn: float
    price_per_decode_tkn: float


        







        


    




