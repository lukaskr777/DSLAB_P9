from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from fastapi import Request
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import glob
from cost_estimation_hl.configs.usage_simulation_config import UsageSimulationCfg
from cost_estimation_hl.usage_simulation import UsageSimulation
from cost_estimation_hl.computation_simulation import *
from cost_estimation_hl.configs.computation_config import *

BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

REAL_DF = None  # Global variable for real data

# --- Lifespan context to load real data once ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global REAL_DF
    print("Loading real data...")
    models_of_interest = ["apertus-8b-instruct", "apertus-70b-instruct"]
    moi = models_of_interest[1]  # pick model
    files = glob.glob("data/litellm/public.LiteLLM_SpendLogs/1/*.parquet")

    df_list = []
    for file in files:
        temp_df = pd.read_parquet(file)
        temp_df = temp_df[temp_df["model"] == moi]
        temp_df = temp_df[temp_df["status"] == "success"]
        df_list.append(temp_df)

    df_real = pd.concat(df_list, ignore_index=True)

    keep_cols = [
        "end_user",
        "startTime",
        "endTime",
        "completionStartTime",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    df_real = df_real[[c for c in keep_cols if c in df_real.columns]].copy()
    df_real["startTime"] = pd.to_datetime(df_real["startTime"], errors="coerce")
    df_real["completionStartTime"] = pd.to_datetime(df_real["completionStartTime"], errors="coerce")

    REAL_DF = df_real.sort_values("startTime")
    print("Real data loaded:", len(REAL_DF), "rows")

    yield  # app running

    print("Application shutdown")

# --- FastAPI app ---
app = FastAPI(lifespan=lifespan)

# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Pydantic Models ---
class SimRequest(BaseModel):
    hour_peak: int
    users: int
    daily_prompts: int
    median_prompt_tokens: int
    repetitions: int
    real_date: str

class LatencyRequest(BaseModel):
    batch_size: int
    dynamic_lim_sec: float
    pricing_ratio: float
    data_source: str   # "real" or "sim"
    real_date: str
    usage_params: SimRequest = None  # nested object for sim 1 values


# --- Simulation endpoint ---
@app.post("/simulate")
def simulate(req: SimRequest):
    global REAL_DF
    all_hourly = []

    # Run repetitions
    for _ in range(req.repetitions):
        cfg = UsageSimulationCfg(
            days=1,
            hour_peak=req.hour_peak,
            users=req.users,
            daily_prompts=req.daily_prompts,
            median_prompt_tokens=req.median_prompt_tokens,
            start_date=pd.Timestamp("2025-09-04")
        )
        sim = UsageSimulation(cfg)
        df_sim = sim.simulate()
        df_sim["hour"] = df_sim["startTime"].dt.hour
        df_sim["total_tokens"] = df_sim["prompt_tokens"] + df_sim["completion_tokens"]
        hourly = df_sim.groupby("hour")["total_tokens"].sum()
        # Ensure length 24
        hourly_full = hourly.reindex(range(24), fill_value=0)
        all_hourly.append(hourly_full.values)

    arr = np.array(all_hourly)
    median = np.median(arr, axis=0).tolist()
    p5 = np.percentile(arr, 5, axis=0).tolist()
    p95 = np.percentile(arr, 95, axis=0).tolist()
    hours = list(range(24))

    # --- Filter real data for selected date ---
    selected_date = pd.to_datetime(req.real_date).date()
    df_real_date = REAL_DF[REAL_DF["startTime"].dt.date == selected_date].copy()
    df_real_date["hour"] = df_real_date["startTime"].dt.hour
    df_real_date["total_tokens"] = df_real_date.get(
        "total_tokens",
        df_real_date["prompt_tokens"] + df_real_date["completion_tokens"]
    )
    real_hourly = df_real_date.groupby("hour")["total_tokens"].sum().reindex(hours, fill_value=0).tolist()

    return {
        "hours": hours,
        "median": median,
        "p5": p5,
        "p95": p95,
        "real": real_hourly
    }




# --- Latency Simulation ---
@app.post("/latency_simulate")
def latency_simulate(req: LatencyRequest):
    selected_date = pd.to_datetime(req.real_date).date()
    cfg_compute = ComputeSimulationCfg("cost_estimation_hl/configs/cluster_configs.json",
                                       batch_size=req.batch_size,
                                       dynamic_lim_sec=req.dynamic_lim_sec,
                                       util_f=lambda b: min(1,b/25.0)
                                       )
    sim_compute = ComputeSimulation(cfg_compute)

    # Choose input data
    if req.data_source == "real":
        df_input = REAL_DF[REAL_DF["startTime"].dt.date == selected_date].copy()
        dfs_compute = sim_compute.simulate(df_input, start_time_ix=1, prompt_tks_ix=4, completion_tks_ix=5)
    else:
        # Use usage_params from sim request
        u = req.usage_params
        cfg_sim = UsageSimulationCfg(
            days=1,
            hour_peak=u.hour_peak,
            users=u.users,
            daily_prompts=u.daily_prompts,
            median_prompt_tokens=u.median_prompt_tokens,
            start_date=pd.Timestamp(selected_date)
        )
        sim = UsageSimulation(cfg_sim)
        df_input = sim.simulate()
        dfs_compute = sim_compute.simulate(df_input, start_time_ix=0, prompt_tks_ix=1, completion_tks_ix=2)

    # Run ComputeSimulation for all clusters


    

    prompt_tks = df_input["prompt_tokens"].sum()
    comp_tks = df_input["completion_tokens"].sum()
    R = req.pricing_ratio

    results = []
    for cluster_cfg, df_est in zip(cfg_compute.cluster_cfgs, dfs_compute):
        df_full = sim_compute.concat(df_input,df_est)

        # Latency per 1k tokens
        median_latency_1k = 1e3 * ((df_full["prefillTime"] - df_full["startTime"]).dt.total_seconds() / df_full["prompt_tokens"]).quantile(0.5)

        # Price per 1k tokens
        prefill_price_1K = cluster_cfg.price_per(days=1) / (prompt_tks + R * comp_tks)
        decode_price_1K = prefill_price_1K * R

        results.append({
            "cluster_name": cluster_cfg.name,
            "median_latency_1k": median_latency_1k,
            "prefill_price_1K": prefill_price_1K,
            "decode_price_1K": decode_price_1K
        })

    # Compute true latency if using real data (hline)
    true_latency_1k = None
    if req.data_source == "real":
        true_latency_1k = 1e3 * ((df_input["completionStartTime"] - df_input["startTime"]).dt.total_seconds() / df_input["prompt_tokens"]).quantile(0.5)

    return {"results": results, "true_latency_1k": true_latency_1k}