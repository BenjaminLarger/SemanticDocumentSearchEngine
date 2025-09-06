#!/usr/bin/env python3
"""
Embedding Pipeline Module
Handles the complete pipeline: text chunks → embeddings → normalization → storage
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import normalize

try:
    from .generator import EmbeddingGenerator, Embedding
    from ..document_processor.splitter import DocumentChunk
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.embeddings.generator import EmbeddingGenerator, Embedding
    from src.document_processor.splitter import DocumentChunk


@dataclass
class ProcessedEmbedding:
    """Processed embedding with normalized vector and metadata"""
    embedding_id: str
    normalized_vector: np.ndarray
    original_vector: np.ndarray
    chunk_metadata: Dict[str, Any]
    created_at: str


class EmbeddingPipeline:
    """Pipeline for converting chunks to normalized, stored embeddings"""
    
    def __init__(
        self,
        model_name: str = "lightweight",
        storage_dir: str = "embeddings_storage",
        batch_size: int = 32
    ):
        self.logger = logging.getLogger(__name__)
        self.generator = EmbeddingGenerator(model_name=model_name, batch_size=batch_size)
        
        # Setup storage
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        (self.storage_dir / "vectors").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
    
    def process_chunks(self, chunks: List[DocumentChunk]) -> List[ProcessedEmbedding]:
        """Convert chunks to normalized embeddings with metadata"""
        if not chunks:
            return []
        
        self.logger.info(f"Processing {len(chunks)} chunks through embedding pipeline")
        
        # Step 1: Generate embeddings
        embeddings = self.generator.generate_embeddings(chunks)
        
        # Step 2: Normalize and process
        processed_embeddings = []
        for embedding in embeddings:
            # Normalize vector for cosine similarity
            normalized_vector = normalize(
                embedding.vector.reshape(1, -1), 
                norm='l2'
            )[0]
            
            processed = ProcessedEmbedding(
                embedding_id=f"{embedding.chunk_id}_{embedding.text_hash}",
                normalized_vector=normalized_vector,
                original_vector=embedding.vector,
                chunk_metadata=embedding.chunk_metadata,
                created_at=datetime.now().isoformat()
            )
            processed_embeddings.append(processed)
        
        # Step 3: Store embeddings
        self._store_embeddings(processed_embeddings)
        
        self.logger.info(f"Pipeline processed {len(processed_embeddings)} embeddings")
        return processed_embeddings
    
    def _store_embeddings(self, embeddings: List[ProcessedEmbedding]):
        """Store embeddings with associated metadata"""
        for embedding in embeddings:
            try:
                # Store vector data
                vector_file = self.storage_dir / "vectors" / f"{embedding.embedding_id}.npz"
                np.savez_compressed(
                    vector_file,
                    normalized=embedding.normalized_vector,
                    original=embedding.original_vector
                )
                
                # Store metadata
                metadata = {
                    "embedding_id": embedding.embedding_id,
                    "chunk_metadata": embedding.chunk_metadata,
                    "created_at": embedding.created_at,
                    "vector_shape": embedding.normalized_vector.shape
                }
                
                metadata_file = self.storage_dir / "metadata" / f"{embedding.embedding_id}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            except Exception as e:
                self.logger.error(f"Failed to store embedding {embedding.embedding_id}: {str(e)}")
    
    def load_embeddings(self, embedding_ids: List[str]) -> List[ProcessedEmbedding]:
        """Load stored embeddings by IDs"""
        embeddings = []
        
        for embedding_id in embedding_ids:
            try:
                # Load metadata
                metadata_file = self.storage_dir / "metadata" / f"{embedding_id}.json"
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load vectors
                vector_file = self.storage_dir / "vectors" / f"{embedding_id}.npz"
                vectors = np.load(vector_file)
                
                embedding = ProcessedEmbedding(
                    embedding_id=embedding_id,
                    normalized_vector=vectors['normalized'],
                    original_vector=vectors['original'],
                    chunk_metadata=metadata['chunk_metadata'],
                    created_at=metadata['created_at']
                )
                embeddings.append(embedding)
                
            except Exception as e:
                self.logger.error(f"Failed to load embedding {embedding_id}: {str(e)}")
        
        return embeddings


def main():
    """Example usage"""
    import sys
    from src.document_processor.ingestion import DocumentIngestor
    from src.document_processor.splitter import DocumentSplitter, SplittingStrategy
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <file_path>")
        return
    
    file_path = sys.argv[1]
    
    try:
        # Create pipeline
        pipeline = EmbeddingPipeline(model_name="lightweight")
        
        # Process document
        ingestor = DocumentIngestor()
        document = ingestor.ingest_document(file_path)
        
        splitter = DocumentSplitter(chunk_size=300, strategy=SplittingStrategy.SENTENCE)
        chunks = splitter.split_document(document)
        
        # Run through pipeline
        processed_embeddings = pipeline.process_chunks(chunks)
        
        print(f"Processed {len(processed_embeddings)} chunks")
        print(f"Vector dimension: {processed_embeddings[0].normalized_vector.shape}")
        print(f"Sample normalized vector norm: {np.linalg.norm(processed_embeddings[0].normalized_vector):.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()