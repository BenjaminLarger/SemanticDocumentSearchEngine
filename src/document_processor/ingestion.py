#!/usr/bin/env python3
"""
Document Ingestion Module
Handles ingestion of various document formats including PDF and text files.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import PyPDF2
from PyPDF2 import PdfReader


@dataclass
class DocumentMetadata:
    """Document metadata container"""
    filename: str
    filepath: str
    file_size: int
    creation_date: datetime
    modification_date: datetime
    file_type: str
    page_count: Optional[int] = None


@dataclass
class Document:
    """Document container with content and metadata"""
    content: str
    metadata: DocumentMetadata


class DocumentIngestionError(Exception):
    """Custom exception for document ingestion errors"""
    pass


class DocumentIngestor:
    """
    Handles ingestion of various document formats
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def ingest_document(self, file_path: str) -> Document:
        """
        Ingest a single document from file path
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document object with content and metadata
            
        Raises:
            DocumentIngestionError: If file cannot be processed
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise DocumentIngestionError(f"File not found: {file_path}")
        
        # Validate file extension
        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise DocumentIngestionError(
                f"Unsupported file type: {file_path.suffix}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        
        try:
            # Extract metadata
            metadata = self._extract_metadata(file_path)
            
            # Extract content based on file type
            content = self._extract_content(file_path)
            
            return Document(content=content, metadata=metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to ingest document {file_path}: {str(e)}")
            raise DocumentIngestionError(f"Failed to process document: {str(e)}")
    
    def ingest_directory(self, directory_path: str, recursive: bool = True) -> List[Document]:
        """
        Ingest all supported documents from a directory
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to search subdirectories
            
        Returns:
            List of Document objects
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            raise DocumentIngestionError(f"Directory not found: {directory_path}")
        
        documents = []
        
        # Get file pattern based on recursive flag
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    document = self.ingest_document(str(file_path))
                    documents.append(document)
                    self.logger.info(f"Successfully ingested: {file_path}")
                except DocumentIngestionError as e:
                    self.logger.warning(f"Skipped {file_path}: {str(e)}")
                    continue
        
        self.logger.info(f"Ingested {len(documents)} documents from {directory_path}")
        return documents
    
    def _extract_metadata(self, file_path: Path) -> DocumentMetadata:
        """Extract metadata from file"""
        stat = file_path.stat()
        
        metadata = DocumentMetadata(
            filename=file_path.name,
            filepath=str(file_path.absolute()),
            file_size=stat.st_size,
            creation_date=datetime.fromtimestamp(stat.st_ctime),
            modification_date=datetime.fromtimestamp(stat.st_mtime),
            file_type=file_path.suffix.lower()
        )
        
        # Add page count for PDFs
        if file_path.suffix.lower() == '.pdf':
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    metadata.page_count = len(pdf_reader.pages)
            except Exception as e:
                self.logger.warning(f"Could not extract page count from {file_path}: {str(e)}")
                metadata.page_count = None
        
        return metadata
    
    def _extract_content(self, file_path: Path) -> str:
        """Extract text content from file based on type"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._extract_pdf_content(file_path)
        elif file_extension in {'.txt', '.md'}:
            return self._extract_text_content(file_path)
        else:
            raise DocumentIngestionError(f"Unsupported file type: {file_extension}")
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract text content from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                
                # Extract text from all pages
                text_content = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1} of {file_path}: {str(e)}")
                        continue
                
                if not text_content:
                    raise DocumentIngestionError("No text content could be extracted from PDF")
                
                return "\n\n".join(text_content)
                
        except Exception as e:
            raise DocumentIngestionError(f"Failed to process PDF {file_path}: {str(e)}")
    
    def _extract_text_content(self, file_path: Path) -> str:
        """Extract content from text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            if not content.strip():
                raise DocumentIngestionError("File is empty or contains only whitespace")
                
            return content
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                return content
            except Exception as e:
                raise DocumentIngestionError(f"Failed to decode text file {file_path}: {str(e)}")
        except Exception as e:
            raise DocumentIngestionError(f"Failed to read text file {file_path}: {str(e)}")


def main():
    """Example usage of DocumentIngestor"""
    logging.basicConfig(level=logging.INFO)
    
    ingestor = DocumentIngestor()
    
    # Example: Ingest a single document
    try:
        document = ingestor.ingest_document("data/raw/MarketResearch.pdf")
        print(f"Ingested: {document.metadata.filename}")
        print(f"Content length: {len(document.content)} characters")
        print(f"File size: {document.metadata.file_size} bytes")
    except DocumentIngestionError as e:
        print(f"Error: {e}")
    
    # Example: Ingest all documents from directory
    try:
        documents = ingestor.ingest_directory("data/raw")
        print(f"Ingested {len(documents)} documents")
        for doc in documents:
            print(f"- {doc.metadata.filename} ({doc.metadata.file_type})")
    except DocumentIngestionError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()