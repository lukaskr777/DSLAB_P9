

from dataclasses import dataclass


@dataclass
class Prompt:
    prompt_tokens: int
    completion_tokens: int
    completion_time: float = 0

    def __str__(self):
        return f"pt: {self.prompt_tokens}, ct:{self.completion_tokens}, time:{self.completion_time}"

@dataclass
class GPUConfig:
    name: str
    vram_B: float
    hbm_bandwidth_B_sec: float
    server_node_bandwidth_B_sec: float
    server_node_fix_latency_sec: float
    peak_flops_fp16: float


@dataclass
class ModelConfig:
    name: str
    model_size: int
    num_layers: int
    attn_hidden_dim: int
    ffn_hidden_dim: int


    def __post_init__(self):
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = self.attn_hidden_dim*4


@dataclass
class ParallelizeConfig:
    tp_size: int # tensor parallelism


@dataclass
class DTypeConfig:
    weight_B: int
    activation_B: int




if __name__ == "__main__":
    pass


