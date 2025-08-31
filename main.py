#!/usr/bin/env python3
"""
Semantic Document Search Engine
Entry point for the application
"""

import uvicorn
from src.api.endpoints import app

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )