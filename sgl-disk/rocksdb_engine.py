import torch
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass
from hash_generator import HashGenerator
import rocksdb_storage_backend
from sglang.srt.configs.model_config import ModelConfig
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RocksDBConfig:
    engine_name: str
    chunk_size: int
    db_path: str

class RocksDBStorageEngine:
    def __init__(self, config: RocksDBConfig, model_config: ModelConfig):
        self.engine_name = config.engine_name
        self.config = config
        self.model_config = model_config
        self.hash_generator = HashGenerator(config.chunk_size)
        try:
            self.db = rocksdb_storage_backend.RocksDBStorageBackend()
            self.db.init(config.db_path)
            logger.info(f"RocksDB storage engine initialized with config: {config}")
        except Exception as e:
            logger.error(f"Error initializing RocksDB storage engine: {e}")
            raise e


    def _blob_size(self) -> int:
        head_dim = self.model_config.head_dim
        num_heads = self.model_config.num_attention_heads
        num_layers = self.model_config.num_hidden_layers
        
        base_size = self.config.chunk_size * (num_heads * head_dim * 2) * num_layers
        
        alignment = 1024 * 1024
        aligned_size = math.ceil(base_size / alignment) * alignment
        
        logger.info(f"Calculated blob size: {base_size} bytes, aligned to {aligned_size} bytes")
        return aligned_size

    def storekv(
        self, 
        input_ids: torch.Tensor,
        kv: torch.Tensor,
        hash_keys: Optional[List[str]] = None,
        prefix_hash: Optional[str] = None
    ) -> List[str]:
        if hash_keys is None:
            hash_keys = self.hash_generator.process_tokens(input_ids, prefix_hash)
        
        if kv.shape[1] != len(hash_keys) * self.config.chunk_size:
            raise ValueError(f"KV length ({kv.shape[1]}) does not match input_ids length ({len(hash_keys)})")        
        
        for i, key in enumerate(hash_keys):
            start_idx = i * self.config.chunk_size
            end_idx = start_idx + self.config.chunk_size
            self.db.storekv(key + "_K", kv[0, start_idx:end_idx])
            self.db.storekv(key + "_V", kv[1, start_idx:end_idx])
        
        return input_ids
    
    def batch_storekv(
        self,
        input_ids: torch.Tensor,
        kv: torch.Tensor,
        hash_keys: Optional[List[str]] = None,
        prefix_hash: Optional[str] = None
    ) -> List[str]:
        if hash_keys is None:
            hash_keys = self.hash_generator.process_tokens(input_ids, prefix_hash)

        expected_len = len(hash_keys) * self.config.chunk_size
        if kv.shape[1] != expected_len:
            raise ValueError(
                f"KV width ({kv.shape[1]}) != #hashes ({len(hash_keys)}) * chunk_size ({self.config.chunk_size})"
            )

        batch: List[Tuple[str, torch.Tensor]] = []
        for i, key in enumerate(hash_keys):
            start = i * self.config.chunk_size
            end   = start + self.config.chunk_size

            k_slice = kv[0, start:end]
            v_slice = kv[1, start:end]

            batch.append((f"{key}_K", k_slice))
            batch.append((f"{key}_V", v_slice))

        self.db.storekv_batch(batch)

        return hash_keys

    def retrievekv(
        self,
        input_ids: Union[List[str], torch.Tensor],
        kv: torch.Tensor,
        hash_keys: Optional[List[str]] = None,
        prefix_hash: Optional[str] = None
    ) -> None:
        if hash_keys is None:
            hash_keys = self.hash_generator.process_tokens(input_ids, prefix_hash)
            
        if kv.shape[1] != len(hash_keys) * self.config.chunk_size:
            raise ValueError(f"KV length ({kv.shape[1]}) does not match input_ids length ({len(hash_keys)})")
        
        for i, key in enumerate(hash_keys):
            start_idx = i * self.config.chunk_size
            end_idx = start_idx + self.config.chunk_size
            self.db.retrievekv(key + "_K", kv[0, start_idx:end_idx])
            self.db.retrievekv(key + "_V", kv[1, start_idx:end_idx])
        return kv
    
    def batch_retrievekv(
        self,
        input_ids: Union[List[str], torch.Tensor],
        kv: torch.Tensor,
        hash_keys: Optional[List[str]] = None,
        prefix_hash: Optional[str] = None
    ) -> torch.Tensor:
        if hash_keys is None:
            hash_keys = self.hash_generator.process_tokens(input_ids, prefix_hash)

        expected_len = len(hash_keys) * self.config.chunk_size
        if kv.shape[1] != expected_len:
            raise ValueError(
                f"KV width ({kv.shape[1]}) != #hashes ({len(hash_keys)}) * chunk_size ({self.config.chunk_size})"
            )

        batch: List[tuple[str, torch.Tensor]] = []
        for i, key in enumerate(hash_keys):
            start = i * self.config.chunk_size
            end   = start + self.config.chunk_size

            k_view = kv[0, start:end]
            v_view = kv[1, start:end]
            batch.append((f"{key}_K", k_view))
            batch.append((f"{key}_V", v_view))

        self.db.retrievekv_batch(batch)

        return kv
    
    def remove(
        self,
        keys: List[str]
    ) -> None:
        for key in keys:
            self.db.remove(key)

    def close(self) -> None:
        self.db.close()