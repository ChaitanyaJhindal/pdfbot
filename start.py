#!/usr/bin/env python
"""
Startup script for the PDF Chatbot API
Optimized for Render deployment with HackRx 6.0 compliance
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        access_log=True,
        log_level="info"
    )
