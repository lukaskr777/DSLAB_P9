
from inference_config import *
import numpy as np
## TODO: add syncing costs for data size - and refine them
## TODO: add GPU  flops worse persofmance for small enough flops, and gpu parralelism - basically if the total  flops per second is too large it doesnt mean we can do everything almost simoutenously
## TODO: potential memory loading off loading to other memory types (HBM, DDR, disk) - if too large - we could assume that the hdd is  inifinte, but it would incurr additional costs
## TODO: add electircity consumption costs
## TODO: study the two types of parallelism - data and model parallelism - and their effect on the model training/inference time
## TODO: add tokens per second calculations



class Inference:

    def __init__(self,gpu_config, model_config, parallelize_config,datat_config):
        self.gpu_config = gpu_config
        self.llm_config = model_config
        self.par_config = parallelize_config
        self.dt_config = datat_config


        self.prefill_logs = {"ML": 0, "AL": 0, "FFNL": 0, "KVL": 0,"TPS": 0}
        self.decode_logs = {"ML": 0, "AL": 0, "FFNL": 0, "KVL": 0, "TPS":0}


    def get_tp_sync_latency(self, batch, completion_tokens):
        if self.par_config.tp_size <= 1:
            return 0.0  # single GPU, no inter-node comm
         
        sync_activation_size = (sum(map(lambda x: x.prompt_tokens*self.llm_config.attn_hidden_dim, batch)) if completion_tokens == 0 else len(batch)*self.llm_config.attn_hidden_dim 
                           * self.dt_config.activation_B) // self.par_config.tp_size
                
        return 2*self.gpu_config.server_node_fix_latency_sec + sync_activation_size / self.gpu_config.server_node_bandwidth_B_sec## 2 times coz we need to sen dand recieve - all reduce



    def get_gpu_flops(self,batch_size):
        base_efficiency = 0.3
        return self.gpu_config.peak_flops_fp16*(1-np.exp(-batch_size/32)) * base_efficiency 


    def get_model_latency(self):
        model_size = (self.llm_config.model_size // self.par_config.tp_size)*self.dt_config.weight_B
        model_latency = model_size / self.gpu_config.hbm_bandwidth_B_sec
        return model_latency
    
    def get_attn_latency(self,batch,completion_tks, kv_cache = True):

        if kv_cache:
            attn_flops_layer = sum(map(lambda x: 
                                        2*self.llm_config.attn_hidden_dim**2 +
                                        2*(x.prompt_tokens+completion_tks)*self.llm_config.attn_hidden_dim
                                        , batch)) // self.par_config.tp_size 
            
            attn_latency = attn_flops_layer*self.llm_config.num_layers/self.get_gpu_flops(len(batch))
   
        else:
            attn_flops_layer = sum(map(lambda x: 
                                        2*(x.prompt_tokens+completion_tks)*self.llm_config.attn_hidden_dim**2 +
                                        2*( (x.prompt_tokens+completion_tks)**2)*self.llm_config.attn_hidden_dim
                                        , batch)) // self.par_config.tp_size 
            
            attn_latency = attn_flops_layer*self.llm_config.num_layers/self.get_gpu_flops(len(batch))
            

        return attn_latency
    
    def get_ffn_latency(self,batch,completion_tks):

        ffn_flops_layer = sum(map(lambda x: 
                                    2*(x.prompt_tokens if completion_tks == 0 else 1)*self.llm_config.ffn_hidden_dim*self.llm_config.attn_hidden_dim
                                    , batch)) // self.par_config.tp_size
        
        ffn_latency = ffn_flops_layer*self.llm_config.num_layers/self.get_gpu_flops(len(batch))

        return ffn_latency
    
    def get_load_store_kv_cache(self,batch,completion_tks):
        kv_size = self.llm_config.num_layers*self.dt_config.activation_B*sum(map(lambda x: 
                                    2*(x.prompt_tokens + completion_tks)*self.llm_config.attn_hidden_dim
                                    , batch)) // self.par_config.tp_size
        kv_latency = kv_size / self.gpu_config.hbm_bandwidth_B_sec

        return kv_latency 
    

    def token_decode_latency(self,batch :list, completion_tks,kv_caching):

        
        model_latency = self.get_model_latency()
        
        attn_latency = self.get_attn_latency(batch,completion_tks,False if completion_tks == 0 else kv_caching)
    
        ffn_latency = self.get_ffn_latency(batch,completion_tks)
         
        kv_latency = self.get_load_store_kv_cache(batch,completion_tks) if kv_caching else 0   

        tp_sync_latency = self.get_tp_sync_latency(batch,completion_tks) 

      
        if completion_tks == 0:
            self.prefill_logs["ML"] += model_latency
            self.prefill_logs["AL"] += attn_latency
            self.prefill_logs["FFNL"] += ffn_latency
            self.prefill_logs["KVL"] += kv_latency
            self.prefill_logs["TPS"] += tp_sync_latency
        
        else:
            self.decode_logs["ML"] += model_latency
            self.decode_logs["AL"] += attn_latency
            self.decode_logs["FFNL"] += ffn_latency
            self.decode_logs["KVL"] += kv_latency
            self.decode_logs["TPS"] += tp_sync_latency
        
            
        
        total_latency = model_latency +attn_latency + ffn_latency + kv_latency

        return total_latency 



    def forward(self,batch : list, kv_caching = True):
        
        
        prefill_latency = self.token_decode_latency(batch,0,kv_caching)
            

        decode_latency = 0

        q_batch = sorted(batch,key= lambda x: x.completion_tokens)

        
        max_comp_tks= q_batch[-1].completion_tokens
    
        for ctks in range(1,max_comp_tks+1):

            tmp_batch=  []
            for i, _ in enumerate(q_batch):
                if _.completion_tokens <= ctks: 
                    _.completion_time = prefill_latency + decode_latency
                
                    
                else: 
                    tmp_batch = q_batch[i:]   
                    break
           
            
            q_batch = tmp_batch
    

            if len(q_batch) > 0:
                decode_latency += self.token_decode_latency(q_batch,ctks,kv_caching)
            else: break
        
           
        return prefill_latency + decode_latency
        
    


# WE ASUME FP16, which affeects flops and model size computations
if __name__ == "__main__":
    

    gpu = GPUConfig("gpu1",40*10e9,1.555*10e12,0.3*10e12,8e-06,20*10e12)
    model = ModelConfig("llm1",70*10e9,80,8192,28672)
    par = ParallelizeConfig(4)
    dtype = DTypeConfig(2,2)
    
    
    batch = [Prompt(1000,500) for i in range(65)]


    inf = Inference(gpu,model,par,dtype)

    print(inf.forward(batch,kv_caching=True))
    print(inf.prefill_logs)
    print(inf.decode_logs)













    