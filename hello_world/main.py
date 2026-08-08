#!/usr/bin/env python3
"""Minimal CPU hello service for vkong."""

from fastapi import FastAPI
import os

PORT = int(os.environ.get("PORT", "8000"))
app = FastAPI(title="vkong-hello")


@app.get("/")
def root():
    return {"ok": True, "service": "hello", "port": PORT}
