from  configs.usage_simulation_config import * 
from scipy.stats import lognorm
import pandas as pd
import matplotlib.pyplot as plt


class UsageSimulation:


    def __init__(self,config : UsageSimulationCfg, labels = ["startTime", "prompt_tokens","completion_tokens"]):
        
        self.cfg = config
        self.labels = labels

        self.time_d = TimeDistribution(hour_peak=self.cfg.hour_peak, kappa=self.cfg.kappa)
        self.prompt_d = lognorm(s = self.cfg.prompt_sigma, scale = self.cfg.median_prompt_tokens)
        self.completion_d = CompletionTokenDist() if self.cfg.median_decode_prompts is None else CompletionTokenDist(indep_sigma= self.cfg.decode_sigma, indep_scale = self.cfg.media_decode_prompts)

    
    def simulate(self):
        
        all_prompts = []

        for day in range(self.cfg.days):
            total_prompts = self.cfg.num_users * self.cfg.daily_prompts

            times = np.sort(self.time_d.rvs(total_prompts))

            prompt_tks = self.prompt_d.rvs(size= total_prompts)
            completion_tks = self.completion_d.rvs(prompt_tks)
            for i in range(total_prompts):
                all_prompts.append([self.cfg.start_date + pd.Timedelta(days = day,hours=times[i]),prompt_tks[i],completion_tks[i]])


        return pd.DataFrame(all_prompts,columns=self.labels)
    



if __name__ == "__main__":

    cfg = UsageSimulationCfg(
        days = 1,
        hour_peak= 12,
        users= 4800,
        daily_prompts= 3,
        median_prompt_tokens= 1200,
        start_date=pd.Timestamp(year=2025,month=9,day=4)
    )

    sim = UsageSimulation(cfg)

    df = sim.simulate()
    print(df.head(5))

        # Add hour column
    df["hour"] = df["startTime"].dt.hour

    # Compute total tokens
    df["total_tokens"] = df["prompt_tokens"] + df["completion_tokens"]

    # Group by hour
    hourly = df.groupby("hour")["total_tokens"].sum()

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(
        hourly.index,
        hourly.values,
        marker="s",      # square markers
        markersize=6
    )

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("Hour")
    plt.ylabel("Total Tokens")
    plt.title("Total Tokens per Hour")
    plt.tight_layout()
    plt.show()
   

