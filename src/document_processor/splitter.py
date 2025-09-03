#!/usr/bin/env python3
"""
Document Chunking Module
Handles splitting documents into chunks for embedding generation.
Supports multiple splitting strategies: character-based, sentence-aware, and paragraph-aware.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

try:
    from .ingestion import Document, DocumentMetadata
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.document_processor.ingestion import Document, DocumentMetadata


class SplittingStrategy(Enum):
    """Available text splitting strategies"""
    CHARACTER = "character"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"


@dataclass
class DocumentChunk:
    """Document chunk container with content, metadata, and chunk-specific info"""
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    chunk_size: int
    overlap_size: int
    start_position: int
    end_position: int
    token_count: Optional[int] = None


class DocumentSplitterError(Exception):
    """Custom exception for document splitting errors"""
    pass


class DocumentSplitter:
    """
    Handles splitting documents into chunks using various strategies
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 50,
        strategy: SplittingStrategy = SplittingStrategy.CHARACTER
    ):
        """
        Initialize the document splitter
        
        Args:
            chunk_size: Target size of each chunk (in tokens/characters)
            overlap_size: Number of characters/tokens to overlap between chunks
            strategy: Text splitting strategy to use
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        if overlap_size >= chunk_size:
            raise DocumentSplitterError("Overlap size must be smaller than chunk size")
    
    def split_document(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks based on the configured strategy
        
        Args:
            document: Document to split
            
        Returns:
            List of DocumentChunk objects
            
        Raises:
            DocumentSplitterError: If splitting fails
        """
        try:
            self.logger.info(f"Splitting document: {document.metadata.filename} using {self.strategy.value} strategy")
            
            if not document.content.strip():
                self.logger.warning(f"Document {document.metadata.filename} has no content to split")
                return []
            
            if self.strategy == SplittingStrategy.CHARACTER:
                return self._split_by_characters(document)
            elif self.strategy == SplittingStrategy.SENTENCE:
                return self._split_by_sentences(document)
            elif self.strategy == SplittingStrategy.PARAGRAPH:
                return self._split_by_paragraphs(document)
            else:
                raise DocumentSplitterError(f"Unsupported splitting strategy: {self.strategy}")
                
        except Exception as e:
            self.logger.error(f"Failed to split document {document.metadata.filename}: {str(e)}")
            raise DocumentSplitterError(f"Document splitting failed: {str(e)}")
    
    def _split_by_characters(self, document: Document) -> List[DocumentChunk]:
        """Split document by character count with overlap"""
        content = document.content
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_content = content[start:end]
            
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=document.metadata,
                chunk_index=chunk_index,
                chunk_size=len(chunk_content),
                overlap_size=self.overlap_size if chunk_index > 0 else 0,
                start_position=start,
                end_position=end,
                token_count=self._estimate_token_count(chunk_content)
            )
            chunks.append(chunk)
            
            start = end - self.overlap_size
            chunk_index += 1
            
            if start >= len(content):
                break
            print()
        
        self.logger.info(f"Split document into {len(chunks)} character-based chunks")
        return chunks
    
    def _split_by_sentences(self, document: Document) -> List[DocumentChunk]:
        """Split document by sentences with approximate chunk size"""
        content = document.content
        
        # Split into sentences using regex
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, content)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        char_position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            if current_chunk and len(current_chunk + " " + sentence) > self.chunk_size:
                # Create chunk from current content
                chunk_start = char_position - len(current_chunk)
                chunk = DocumentChunk(
                    content=current_chunk,
                    metadata=document.metadata,
                    chunk_index=chunk_index,
                    chunk_size=len(current_chunk),
                    overlap_size=self._calculate_overlap(chunks, current_chunk),
                    start_position=chunk_start,
                    end_position=char_position,
                    token_count=self._estimate_token_count(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.overlap_size)
                current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            char_position += len(sentence) + 1  # +1 for space
        
        # Add final chunk if there's remaining content
        if current_chunk:
            chunk_start = char_position - len(current_chunk)
            chunk = DocumentChunk(
                content=current_chunk,
                metadata=document.metadata,
                chunk_index=chunk_index,
                chunk_size=len(current_chunk),
                overlap_size=self._calculate_overlap(chunks, current_chunk),
                start_position=chunk_start,
                end_position=char_position,
                token_count=self._estimate_token_count(current_chunk)
            )
            chunks.append(chunk)
        
        self.logger.info(f"Split document into {len(chunks)} sentence-aware chunks")
        return chunks
    
    def _split_by_paragraphs(self, document: Document) -> List[DocumentChunk]:
        """Split document by paragraphs with approximate chunk size"""
        content = document.content
        
        # Split into paragraphs (double newline or more)
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        char_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed chunk size
            if current_chunk and len(current_chunk + "\n\n" + paragraph) > self.chunk_size:
                # Create chunk from current content
                chunk_start = char_position - len(current_chunk)
                chunk = DocumentChunk(
                    content=current_chunk,
                    metadata=document.metadata,
                    chunk_index=chunk_index,
                    chunk_size=len(current_chunk),
                    overlap_size=self._calculate_overlap(chunks, current_chunk),
                    start_position=chunk_start,
                    end_position=char_position,
                    token_count=self._estimate_token_count(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.overlap_size)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                chunk_index += 1
            else:
                # Add paragraph to current chunk
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            char_position += len(paragraph) + 2  # +2 for double newline
        
        # Add final chunk if there's remaining content
        if current_chunk:
            chunk_start = char_position - len(current_chunk)
            chunk = DocumentChunk(
                content=current_chunk,
                metadata=document.metadata,
                chunk_index=chunk_index,
                chunk_size=len(current_chunk),
                overlap_size=self._calculate_overlap(chunks, current_chunk),
                start_position=chunk_start,
                end_position=char_position,
                token_count=self._estimate_token_count(current_chunk)
            )
            chunks.append(chunk)
        
        self.logger.info(f"Split document into {len(chunks)} paragraph-aware chunks")
        return chunks
    
    def _calculate_overlap(self, existing_chunks: List[DocumentChunk], current_chunk: str) -> int:
        """Calculate the actual overlap size for the current chunk"""
        if not existing_chunks:
            return 0
        
        previous_chunk = existing_chunks[-1]
        overlap_text = self._get_overlap_text(previous_chunk.content, self.overlap_size)
        
        if overlap_text and current_chunk.startswith(overlap_text):
            return len(overlap_text)
        return 0
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last overlap_size characters from text for overlap"""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count using simple heuristic
        Roughly 4 characters per token for English text
        """
        return max(1, len(text) // 4)
    
    def split_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """
        Split multiple documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        for document in documents:
            try:
                chunks = self.split_document(document)
                all_chunks.extend(chunks)
            except DocumentSplitterError as e:
                self.logger.error(f"Failed to split document {document.metadata.filename}: {str(e)}")
                continue
        
        self.logger.info(f"Split {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        sizes = [chunk.chunk_size for chunk in chunks]
        token_counts = [chunk.token_count for chunk in chunks if chunk.token_count]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "avg_token_count": sum(token_counts) / len(token_counts) if token_counts else 0,
            "total_characters": sum(sizes),
            "estimated_total_tokens": sum(token_counts) if token_counts else 0
        }


def main():
    """Example usage of the document splitter"""
    import sys
    try:
        from .ingestion import DocumentIngestor
    except ImportError:
        from src.document_processor.ingestion import DocumentIngestor
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) != 2:
        print("Usage: python splitter.py <file_path>")
        return
    
    file_path = sys.argv[1]
    
    try:
        # Ingest document
        ingestor = DocumentIngestor()
        document = ingestor.ingest_document(file_path)
        
        print(f"Ingested document: {document.metadata.filename}")
        print(f"Content length: {len(document.content)} characters")
        
        # Test different splitting strategies
        strategies = [
            # (SplittingStrategy.CHARACTER, 300, 30),
            (SplittingStrategy.SENTENCE, 400, 40),
            (SplittingStrategy.PARAGRAPH, 500, 50)
        ]
        
        for strategy, chunk_size, overlap in strategies:
            print(f"\n--- {strategy.value.upper()} SPLITTING ---")
            splitter = DocumentSplitter(
                chunk_size=chunk_size,
                overlap_size=overlap,
                strategy=strategy
            )
            
            chunks = splitter.split_document(document)
            stats = splitter.get_chunk_stats(chunks)
            
            print(f"Strategy: {strategy.value}")
            print(f"Chunks created: {stats['total_chunks']}")
            print(f"Average chunk size: {stats['avg_chunk_size']:.1f} characters")
            print(f"Average tokens: {stats['avg_token_count']:.1f}")
            
            # Show first chunk as example
            if chunks:
                print(f"First chunk preview: {chunks[0].content[:100]}...")
    
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()