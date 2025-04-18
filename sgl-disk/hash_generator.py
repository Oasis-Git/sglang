import hashlib
import torch
from typing import Optional, Iterable

class HashGenerator:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size

    def _get_init_hash(self) -> str:
        return ""

    def _hash(self, tokens: torch.Tensor, prefix_hash: str) -> str:
        return hashlib.sha256(
            prefix_hash.encode("ascii") +
            tokens.cpu().numpy().tobytes()
        ).hexdigest()

    def _chunk_tokens(self, tokens: torch.Tensor) -> Iterable[torch.Tensor]:
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i:i + self.chunk_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[torch.Tensor],
        prefix_hash: Optional[str] = None,
    ) -> Iterable[str]:
        if prefix_hash is None:
            prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            yield prefix_hash

    def process_tokens(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prefix_hash: Optional[str] = None,
    ) -> Iterable[tuple[int, int, str]]:
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum()
        else:
            num_falses = 0

        if num_falses % self.chunk_size != 0:
            raise ValueError("The number of Falses in the mask is not a multiple of the chunk size.")

        total_len = len(tokens)
        token_chunks = self._chunk_tokens(tokens)
        prefix_hashes = self._prefix_hash(token_chunks, prefix_hash)

        hash_keys = []
        start_idx = 0
        for chunk_id, hash_val in enumerate(prefix_hashes):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_len)
            if start_idx < num_falses:
                continue
            else:
                hash_keys.append(hash_val)
        return hash_keys