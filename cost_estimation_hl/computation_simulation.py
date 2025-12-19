from configs.computation_config import *
import pandas as pd
import numpy as np
import glob
import os





class ComputeSimulation:


    def __init__(self,cfg : ComputeSimulationCfg, labels = ["batchEndTime", "prefillTime", "decodeTime"]):

        self.cfg = cfg
        self.labels = labels

        
        

    def simulate(self,data : pd.DataFrame,start_time_ix, prompt_tks_ix, completion_tks_ix):
        ## our data must be sorted by the request time
        ## will always return completien times w.r.t the ordered dataset
        data= data.sort_values(by = data.columns[start_time_ix], ascending=True,inplace=False)
        a = data.to_numpy()
        dfs = []
        for cluster_cfg in self.cfg.cluster_cfgs:
            times = self.get_batch_times(a,start_time_ix,prompt_tks_ix,completion_tks_ix,cluster_cfg=cluster_cfg)
            df_ = pd.DataFrame(times, columns=self.labels)
            for col in self.labels:
                df_[col] = pd.to_datetime(df_[col]).dt.round("ms")

            dfs.append(df_)

        return dfs
    
    def get_batch_times(self,a,start_time_ix,prompt_tks_ix, completion_tks_ix, cluster_cfg : ComputeCluster):
        

        ## set the indices of the relevant stuff like start time
        start_time_ix = start_time_ix
        prompt_tks_ix = prompt_tks_ix
        comp_tks_ix = completion_tks_ix

        ## get the dimensions and params
        num_repl = cluster_cfg.num_replicas
        num_batch = self.cfg.batch_size
        peak_prefill_tks_per_sec_per_replica = cluster_cfg.peak_prefill_tks_per_sec_per_replica ## batching is done per replica
        peak_decode_tks_per_sec_per_replica = cluster_cfg.peak_decode_tks_per_sec_per_replica
        
        dyn_lim_sec = self.cfg.dynamic_lim_sec
        
        
        ## keep track of the last completion times of replica
        last_times_per_replica = np.full(num_repl, a[0,start_time_ix], dtype='object')
 
   
        batch_prefill_decode_times = []
        
        ## keep track of the currect batch
        ix = 0
        current_batch = [a[ix]]
        while(len(current_batch)) > 0:
            
            #print(current_batch)
            start_time = current_batch[0][start_time_ix]
            
     
            ## if its still withing constraints, we continue, also if we are at the end, we also must terminate and store stuff
            ix += 1
            if ix < len(a) and (a[ix][start_time_ix] - start_time).total_seconds() < dyn_lim_sec and len(current_batch) < num_batch:
                
                current_batch.append(a[ix])
                continue 
            else:
                possible_start_time = None
                if (len(current_batch) == num_batch or ix == len(a)):
                    possible_start_time = current_batch[-1][start_time_ix]

                else:
                    possible_start_time = start_time + pd.Timedelta(seconds=dyn_lim_sec)


                replica_ix = np.argmin([ts.value for ts in last_times_per_replica])## the replica that ends the earliest is the one who will process the batch
            
                
                actual_start_time = max(
                    possible_start_time,
                    pd.Timestamp(
                        ((last_times_per_replica[replica_ix].value))
                        )
                    )
                
  
                ## 1. End of Batch Fill Phase - when we can start processing
                batch_end_time = actual_start_time
                
                ## 2. Prefill Phase
                prefill_toks = float(sum(map(lambda x:x[prompt_tks_ix],current_batch))) 
                prefill_time_sec = prefill_toks/(peak_prefill_tks_per_sec_per_replica)

                batch_prefill_time = batch_end_time + pd.Timedelta(seconds=prefill_time_sec)

                ## 3. Decode Phase - thi should be handled so that it doesnt block if one is too big
                current_batch = sorted(current_batch,key=lambda x: x[comp_tks_ix])
                timedelta = 0  
                last_tks = 0
                decode_deltas = []

                for i, _ in enumerate(current_batch):
                    timedelta += (_[comp_tks_ix] - last_tks)*len(current_batch[i:]) / (peak_decode_tks_per_sec_per_replica*self.cfg.util_f(len(current_batch[i:]))) ## scale by the batch size
                    decode_deltas.append(timedelta)
                
            
                decode_times = [batch_end_time + pd.Timedelta(seconds=prefill_time_sec + decode_time_sec) for decode_time_sec in decode_deltas]


                ## make sure to remember until what time is this replica occupied
                
                ## we dont want to take the last delay because replcia can simoutaneously do other stuff - if it is no already saturated - lets do threshold //4 size of batch
                delay_time = actual_start_time + pd.Timedelta(seconds=prefill_time_sec + (decode_deltas[-2] if len(decode_times) > 1 else 0))
                last_times_per_replica[replica_ix] = delay_time
                
          
                for i in range(len(current_batch)):
                    batch_prefill_decode_times.append([batch_end_time,batch_prefill_time,decode_times[i]])

                current_batch = [] if ix == len(a) else [a[ix]]



        return batch_prefill_decode_times
    



    def concat(self,df1,df2):

        return pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
    
 


if __name__ == "__main__":

    models_of_interest = ["apertus-8b-instruct","apertus-70b-instruct"]
    moi = models_of_interest[1]
    # 1. Collect files
    files = glob.glob("../data/litellm/public.LiteLLM_SpendLogs/1/*.parquet")

    df_list = []
    for file in files:
        temp_df = pd.read_parquet(file)
        temp_df = temp_df[temp_df["model"] == moi]
        temp_df = temp_df[temp_df["status"] == "success"]

        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)

    # 2. Define the columns you want to KEEP
    keep_cols = [
        "end_user",
        "startTime",
        "endTime",
        "completionStartTime",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]

    # Keep ONLY those (ignore missing if any file doesn't have it)
    df = df[[c for c in keep_cols if c in df.columns]].copy()


    # 3. Parse timestamps and sort
    df["startTime"] = pd.to_datetime(df["startTime"],format = "mixed")
    df["completionStartTime"] = pd.to_datetime(df["completionStartTime"],format = "mixed")
    df = df.sort_values("startTime")
    df_day = df[df["startTime"].dt.date == pd.Timestamp("2025-09-04").date()]
    
    cfg = ComputeSimulationCfg("configs/cluster_configs.json",16,0.1)

    cfg.cluster_cfgs = cfg.cluster_cfgs[:2]
    print(cfg.cluster_cfgs)
    print(df_day.head(3))

    sim = ComputeSimulation(cfg)
    
    dfs= sim.simulate(df_day,1,4,5)
    print("----------------------------")
    for df in dfs:
        print(df.head(3))
        print("-----------------------------------------------")

    df_day_pred = dfs[1]

    df_f = sim.concat(df_day,df_day_pred)
    import matplotlib.pyplot as plt

    # --- Compute TTFT for simulated data ---
    df_f["TTFT_sim_sec"] = (df_f["prefillTime"] - df_f["startTime"]).dt.total_seconds()

    # --- Compute real TTFT ---
    df_f["TTFT_real_sec"] = (df_f["completionStartTime"] - df_f["startTime"]).dt.total_seconds()

    # --- Add hour column for grouping ---
    df_f["hour"] = df_f["startTime"].dt.hour


    # --- Choose quantile to plot ---
    quant = 0.5  # median, can adjust per hour if needed

    # --- Group by hour and compute quantiles ---
    real_hourly = df_f.groupby("hour")["TTFT_real_sec"].quantile(quant)
    sim_hourly = df_f.groupby("hour")["TTFT_sim_sec"].quantile(quant)

    # --- Plot ---
    plt.figure(figsize=(10,5))
    plt.plot(real_hourly.index, real_hourly.values, marker='o', label='Real TTFT')
    plt.plot(sim_hourly.index, sim_hourly.values, marker='s', label='Simulated TTFT')
    plt.xlabel("Hour of Day")
    plt.ylabel(f"TTFT (seconds, {quant*100:.0f}th percentile)")
    plt.title("Time to First Token (TTFT) per Hour")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()







