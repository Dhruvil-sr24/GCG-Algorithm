 
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig: 
    name: str = os.getenv("MODEL_NAME", "gpt2")
    device: str = os.getenv("DEVICE", "cpu")     # "cpu" | "cuda" | "mps"
    dtype: str = os.getenv("DTYPE", "float32")   # "float32" | "float16" | "bfloat16"


@dataclass
class GCGConfig: 
    suffix_len: int = int(os.getenv("GCG_SUFFIX_LEN", "20"))
    init_token: str = os.getenv("GCG_INIT_TOKEN", "!")    
 
    num_steps: int = int(os.getenv("GCG_NUM_STEPS", "500"))
    top_k: int = int(os.getenv("GCG_TOP_K", "256"))        
    batch_size: int = int(os.getenv("GCG_BATCH_SIZE", "512"))   

    # Early stopping
    success_threshold: float = float(os.getenv("GCG_SUCCESS_THRESHOLD", "0.0"))   
    check_success_every: int = int(os.getenv("GCG_CHECK_EVERY", "25"))
 
    log_every: int = int(os.getenv("GCG_LOG_EVERY", "10"))


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    gcg: GCGConfig = field(default_factory=GCGConfig)
    api: APIConfig = field(default_factory=APIConfig)
    results_dir: Path = Path(os.getenv("RESULTS_DIR", "results"))


cfg = Config()
