from abc import ABC, abstractmethod
from typing import List, Optional, Union, Tuple, Any
import torch
from sglang.srt.configs.model_config import ModelConfig
from rocksdb_engine import RocksDBStorageEngine, RocksDBConfig
import logging

class BaseSSDConnector(ABC):
    """
    A base virtual connector class that provides a common interface for different connector implementations.
    This class defines the core API that all connectors should implement.
    """

    def __init__(self):
        """
        Initialize the base virtual connector   
        """
        pass

    @abstractmethod
    def get_hash(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prefix_hash: Optional[Any] = None
    ) -> List[Any]:
        """
        Get the hash for the given token IDs.
        
        Args:
            token_ids (torch.Tensor): The token IDs to hash
            mask (Optional[torch.Tensor]): Optional mask for the token IDs
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            List[Any]: List of hash values for the token IDs
        """
        raise NotImplementedError

    @abstractmethod
    def store_kv(
        self,
        token_ids: torch.Tensor,
        kv_caches: Union[torch.Tensor, List[torch.Tensor]],
        prefix_hash: Optional[Any] = None
    ) -> Tuple[bool, List[Any]]:
        """
        Store the KV caches.
        
        Args:
            token_ids (torch.Tensor): The token IDs
            kv_caches (Union[torch.Tensor, List[torch.Tensor]]): The KV caches to store
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            Tuple[bool, List[Any]]: Success status and list of hash values for stored KV caches
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_kv(
        self,
        token_ids: torch.Tensor,
        kv_caches: Union[torch.Tensor, List[torch.Tensor]],
        prefix_hash: Optional[Any] = None
    ) -> bool:
        """
        Retrieve the KV caches.
        
        Args:
            token_ids (torch.Tensor): The token IDs
            kv_caches (Union[torch.Tensor, List[torch.Tensor]]): The KV caches to retrieve into
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            bool: Success status of the retrieval
        """
        raise NotImplementedError

    @abstractmethod
    def store_kv_hash(
        self,
        hash_: List[Any],
        kv_caches: List[torch.Tensor]
    ) -> Tuple[bool, List[Any]]:
        """
        Store the KV caches using existing hash values.
        
        Args:
            hash_ (List[Any]): List of hash values to use for storage
            kv_caches (List[torch.Tensor]): The KV caches to store
            
        Returns:
            Tuple[bool, List[Any]]: Success status and list of hash values for stored KV caches
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_kv_hash(
        self,
        hash_: List[Any],
        kv_caches: List[torch.Tensor]
    ) -> bool:
        """
        Retrieve the KV caches using existing hash values.
        
        Args:
            hash_ (List[Any]): List of hash values to use for retrieval
            kv_caches (List[torch.Tensor]): The KV caches to retrieve into
            
        Returns:
            bool: Success status of the retrieval
        """
        raise NotImplementedError 
    
    @abstractmethod
    def remove(self, hash_keys: List[Any]) -> None:
        """
        Remove the KV caches using the hash keys.
        """
        raise NotImplementedError
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the SSD connection.
        """
        raise NotImplementedError
    

logger = logging.getLogger(__name__)

class RocksDBConnector(BaseSSDConnector):
    """
    A connector implementation that uses RocksDB as the storage backend.
    This connector implements the BaseSSDCon interface and uses
    RocksDBStorageEngine for actual storage operations.
    """

    def __init__(
        self,
        chunk_size: int,
        engine_name: str,
        db_path: str,
        model_config: ModelConfig
    ):
        """
        Initialize the RocksDB connector.
        
        Args:
            hidden_dim_size (int): The size of the hidden dimension
            num_layers (int): The number of layers in the model
            chunk_size (int): The size of each chunk for processing
            engine_name (str): Name of the RocksDB engine
            db_path (str): Path to the RocksDB database
            model_config (ModelConfig): Model configuration
        """
        config = RocksDBConfig(
            engine_name=engine_name,
            chunk_size=chunk_size,
            db_path=db_path
        )

        self.engine = RocksDBStorageEngine(config, model_config)
        logger.info(f"RocksDB connector initialized with config: {config}")

    def get_hash(
        self,
        token_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prefix_hash: Optional[Any] = None
    ) -> List[str]:
        """
        Get the hash for the given token IDs using the RocksDB engine's hash generator.
        
        Args:
            token_ids (torch.Tensor): The token IDs to hash
            mask (Optional[torch.Tensor]): Optional mask for the token IDs (not used)
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            List[str]: List of hash values for the token IDs
        """
        return self.engine.hash_generator.process_tokens(token_ids, prefix_hash)

    def store_kv(
        self,
        token_ids: torch.Tensor,
        kv_caches: Union[torch.Tensor, List[torch.Tensor]],
        prefix_hash: Optional[Any] = None,
        batch=True
    ) -> Tuple[bool, List[str]]:
        """
        Store the KV caches using the RocksDB engine.
        
        Args:
            token_ids (torch.Tensor): The token IDs
            kv_caches (Union[torch.Tensor, List[torch.Tensor]]): The KV caches to store
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            Tuple[bool, List[str]]: Success status and list of hash values for stored KV caches
        """
        try:
            hash_keys = self.get_hash(token_ids, prefix_hash=prefix_hash)
            
            if not batch:
                self.engine.storekv(token_ids, kv_caches, hash_keys, prefix_hash)
            else:
                self.engine.batch_storekv(token_ids, kv_caches, hash_keys, prefix_hash)
                
            return True, hash_keys
        except Exception as e:
            logger.error(f"Error storing KV caches: {e}")
            return False, []

    def retrieve_kv(
        self,
        token_ids: torch.Tensor,
        kv_caches: Union[torch.Tensor, List[torch.Tensor]],
        prefix_hash: Optional[Any] = None,
        batch=True
    ) -> bool:
        """
        Retrieve the KV caches using the RocksDB engine.
        
        Args:
            token_ids (torch.Tensor): The token IDs
            kv_caches (Union[torch.Tensor, List[torch.Tensor]]): The KV caches to retrieve into
            prefix_hash (Optional[Any]): Optional prefix hash to use
            
        Returns:
            bool: Success status of the retrieval
        """
        try:
            hash_keys = self.get_hash(token_ids, prefix_hash=prefix_hash)
            
            if not batch:
                self.engine.retrievekv(token_ids, kv_caches, hash_keys, prefix_hash)
            else:
                self.engine.batch_retrievekv(token_ids, kv_caches, hash_keys, prefix_hash)
                
            return True
        except Exception as e:
            logger.error(f"Error retrieving KV caches: {e}")
            return False

    def store_kv_hash(
        self,
        hash_: List[str],
        kv_caches: List[torch.Tensor]
    ) -> Tuple[bool, List[str]]:
        """
        Store the KV caches using existing hash values.
        
        Args:
            hash_ (List[str]): List of hash values to use for storage
            kv_caches (List[torch.Tensor]): The KV caches to store
            
        Returns:
            Tuple[bool, List[str]]: Success status and list of hash values for stored KV caches
        """
        try:
            if len(kv_caches) > 1:
                kv_tensor = torch.stack(kv_caches)
            else:
                kv_tensor = kv_caches[0]
                
            self.engine.storekv(None, kv_tensor, hash_keys=hash_)
            return True, hash_
        except Exception as e:
            logger.error(f"Error storing KV caches with hash: {e}")
            return False, []

    def retrieve_kv_hash(
        self,
        hash_: List[str],
        kv_caches: List[torch.Tensor]
    ) -> bool:
        """
        Retrieve the KV caches using existing hash values.
        
        Args:
            hash_ (List[str]): List of hash values to use for retrieval
            kv_caches (List[torch.Tensor]): The KV caches to retrieve into
            
        Returns:
            bool: Success status of the retrieval
        """
        try:
            if len(kv_caches) > 1:
                kv_tensor = torch.stack(kv_caches)
            else:
                kv_tensor = kv_caches[0]
                
            self.engine.retrievekv(None, kv_tensor, hash_keys=hash_)
            return True
        except Exception as e:
            logger.error(f"Error retrieving KV caches with hash: {e}")
            return False
        
    def remove(self, hash_keys: List[Any]) -> None:
        """
        Remove the KV caches using the hash keys.
        """
        self.engine.remove(hash_keys)

    def close(self) -> None:
        """
        Close the RocksDB connection.
        """
        self.engine.close() 