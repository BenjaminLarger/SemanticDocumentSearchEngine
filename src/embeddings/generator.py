#!/usr/bin/env python3
"""
Embedding Generation Module
Handles generating vector embeddings from document chunks using SentenceTransformers.
Supports multiple models with batch processing and caching.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from ..document_processor.splitter import DocumentChunk
except ImportError:
    # For direct execution
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.document_processor.splitter import DocumentChunk


@dataclass
class Embedding:
    """Container for embedding vector with metadata"""
    vector: np.ndarray
    chunk_id: str
    text_hash: str
    model_name: str
    chunk_metadata: Dict[str, Any]


class EmbeddingModel:
    """Available embedding models with their configurations"""
    
    MODELS = {
        "lightweight": {
            "name": "all-MiniLM-L6-v2",
            "dimension": 384,
            "description": "Lightweight, good performance"
        },
        "high_quality": {
            "name": "all-mpnet-base-v2", 
            "dimension": 768,
            "description": "Higher quality, larger size"
        },
        "qa_optimized": {
            "name": "multi-qa-MiniLM-L6-cos-v1",
            "dimension": 384,
            "description": "Optimized for Q&A tasks"
        }
    }


class EmbeddingGeneratorError(Exception):
    """Custom exception for embedding generation errors"""
    pass


class EmbeddingGenerator:
    """
    Generates vector embeddings from document chunks using SentenceTransformers
    """
    
    def __init__(
        self,
        model_name: str = "lightweight",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        enable_caching: bool = True
    ):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Model to use (lightweight, high_quality, qa_optimized) or direct model name
            batch_size: Batch size for processing multiple texts
            cache_dir: Directory for caching embeddings
            enable_caching: Whether to enable embedding caching
        """
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        
        # Resolve model name
        if model_name in EmbeddingModel.MODELS:
            self.model_config = EmbeddingModel.MODELS[model_name]
            self.model_name = self.model_config["name"]
        else:
            self.model_name = model_name
            self.model_config = {"name": model_name, "dimension": None}
        
        # Initialize cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / ".embedding_cache"
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
        self._embedding_cache = {}
        
        # Load model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            self.logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Update dimension if not known
            if self.model_config["dimension"] is None:
                test_embedding = self.model.encode("test", show_progress_bar=False)
                self.model_config["dimension"] = len(test_embedding)
            
            self.logger.info(f"Model loaded successfully. Dimension: {self.model_config['dimension']}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise EmbeddingGeneratorError(f"Model loading failed: {str(e)}")
    
    def generate_embeddings(
        self, 
        chunks: List[DocumentChunk],
        show_progress: bool = True
    ) -> List[Embedding]:
        """
        Generate embeddings for a list of document chunks
        
        Args:
            chunks: List of document chunks to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of Embedding objects
            
        Raises:
            EmbeddingGeneratorError: If embedding generation fails
        """
        if not chunks:
            return []
        
        try:
            self.logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Prepare texts and metadata
            texts = [chunk.content for chunk in chunks]
            embeddings_list = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_chunks = chunks[i:i + self.batch_size]
                
                # Check cache first
                batch_embeddings = []
                uncached_texts = []
                uncached_indices = []
                
                for j, (text, chunk) in enumerate(zip(batch_texts, batch_chunks)):
                    text_hash = self._get_text_hash(text)
                    cached_embedding = self._get_cached_embedding(text_hash)
                    
                    if cached_embedding is not None:
                        # Use cached embedding
                        embedding = Embedding(
                            vector=cached_embedding,
                            chunk_id=f"{chunk.metadata.filename}_{chunk.chunk_index}",
                            text_hash=text_hash,
                            model_name=self.model_name,
                            chunk_metadata=self._extract_chunk_metadata(chunk)
                        )
                        batch_embeddings.append((j, embedding))
                    else:
                        # Need to generate embedding
                        uncached_texts.append(text)
                        uncached_indices.append(j)
                
                # Generate embeddings for uncached texts
                if uncached_texts:
                    try:
                        vectors = self.model.encode(
                            uncached_texts,
                            batch_size=len(uncached_texts),
                            show_progress_bar=show_progress and i == 0
                        )
                        
                        # Create embeddings and cache them
                        for idx, vector in enumerate(vectors):
                            original_idx = uncached_indices[idx]
                            chunk = batch_chunks[original_idx]
                            text = uncached_texts[idx]
                            text_hash = self._get_text_hash(text)
                            
                            # Cache the embedding
                            self._cache_embedding(text_hash, vector)
                            
                            # Create embedding object
                            embedding = Embedding(
                                vector=vector,
                                chunk_id=f"{chunk.metadata.filename}_{chunk.chunk_index}",
                                text_hash=text_hash,
                                model_name=self.model_name,
                                chunk_metadata=self._extract_chunk_metadata(chunk)
                            )
                            batch_embeddings.append((original_idx, embedding))
                    
                    except Exception as e:
                        self.logger.error(f"Batch embedding generation failed: {str(e)}")
                        raise EmbeddingGeneratorError(f"Embedding generation failed: {str(e)}")
                
                # Sort embeddings back to original order
                batch_embeddings.sort(key=lambda x: x[0])
                embeddings_list.extend([emb for _, emb in batch_embeddings])
            
            self.logger.info(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            raise EmbeddingGeneratorError(f"Embedding generation failed: {str(e)}")
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (useful for queries)
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        text_hash = self._get_text_hash(text)
        cached_embedding = self._get_cached_embedding(text_hash)
        
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            vector = self.model.encode(text, show_progress_bar=False)
            self._cache_embedding(text_hash, vector)
            return vector
            
        except Exception as e:
            self.logger.error(f"Failed to generate single embedding: {str(e)}")
            raise EmbeddingGeneratorError(f"Single embedding generation failed: {str(e)}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()[:16]
    
    def _get_cached_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding if available"""
        if not self.enable_caching:
            return None
        
        # Check in-memory cache first
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{text_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    vector = pickle.load(f)
                self._embedding_cache[text_hash] = vector
                return vector
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {str(e)}")
        
        return None
    
    def _cache_embedding(self, text_hash: str, vector: np.ndarray):
        """Cache embedding in memory and disk"""
        if not self.enable_caching:
            return
        
        # Store in memory
        self._embedding_cache[text_hash] = vector
        
        # Store on disk
        try:
            cache_file = self.cache_dir / f"{text_hash}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(vector, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding to disk: {str(e)}")
    
    def _extract_chunk_metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Extract relevant metadata from document chunk"""
        return {
            "filename": chunk.metadata.filename,
            "filepath": chunk.metadata.filepath,
            "file_type": chunk.metadata.file_type,
            "chunk_index": chunk.chunk_index,
            "chunk_size": chunk.chunk_size,
            "start_position": chunk.start_position,
            "end_position": chunk.end_position,
            "token_count": chunk.token_count
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "dimension": self.model_config.get("dimension"),
            "description": self.model_config.get("description", "Custom model"),
            "batch_size": self.batch_size,
            "cache_enabled": self.enable_caching,
            "cache_dir": str(self.cache_dir) if self.enable_caching else None
        }
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        self._embedding_cache.clear()
        if self.enable_caching and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete cache file {cache_file}: {str(e)}")
        self.logger.info("Embedding cache cleared")


def main():
    """Example usage of the embedding generator"""
    import sys
    
    try:
        from ..document_processor.ingestion import DocumentIngestor
        from ..document_processor.splitter import DocumentSplitter, SplittingStrategy
    except ImportError:
        from src.document_processor.ingestion import DocumentIngestor
        from src.document_processor.splitter import DocumentSplitter, SplittingStrategy
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) != 2:
        print("Usage: python generator.py <file_path>")
        return
    
    file_path = sys.argv[1]
    
    try:
        # Ingest and split document
        ingestor = DocumentIngestor()
        document = ingestor.ingest_document(file_path)
        
        splitter = DocumentSplitter(
            chunk_size=300,
            overlap_size=30,
            strategy=SplittingStrategy.SENTENCE
        )
        chunks = splitter.split_document(document)
        
        print(f"Document: {document.metadata.filename}")
        print(f"Chunks: {len(chunks)}")
        
        # Test different embedding models
        models_to_test = ["lightweight", "high_quality", "qa_optimized"]
        
        for model_type in models_to_test:
            print(f"\n--- {model_type.upper()} MODEL ---")
            
            try:
                generator = EmbeddingGenerator(
                    model_name=model_type,
                    batch_size=8,
                    enable_caching=True
                )
                
                model_info = generator.get_model_info()
                print(f"Model: {model_info['model_name']}")
                print(f"Dimension: {model_info['dimension']}")
                print(f"Description: {model_info['description']}")
                
                # Generate embeddings
                embeddings = generator.generate_embeddings(chunks[:3])  # Test first 3 chunks
                
                print(f"Generated embeddings: {len(embeddings)}")
                if embeddings:
                    print(f"Vector shape: {embeddings[0].vector.shape}")
                    print(f"Sample vector (first 5 values): {embeddings[0].vector[:5]}")
                
                # Test single embedding (query simulation)
                query_text = "sample query text"
                query_embedding = generator.generate_single_embedding(query_text)
                print(f"Query embedding shape: {query_embedding.shape}")
                
            except Exception as e:
                print(f"Error with {model_type} model: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()