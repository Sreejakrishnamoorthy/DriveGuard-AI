"""
DriveGuard AI - FastAPI Backend Entry Point.

Run with:
    uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import routes_analysis, routes_health, routes_incidents, routes_risk, routes_sessions
from backend.services.analysis_service import AnalysisService
from src.utils.config import Config
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models and services ONCE at startup (not per-request), and
    clean up on shutdown.
    """
    logger.info("DriveGuard AI backend starting up...")
    Config.print_config()
    app.state.analysis_service = AnalysisService()
    logger.info("DriveGuard AI backend ready.")

    yield

    logger.info("DriveGuard AI backend shutting down...")
    app.state.analysis_service.database_service.close()


app = FastAPI(
    title="DriveGuard AI",
    description="Multimodal Real-Time Driver Risk & Road Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relax for local/academic use; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_health.router)
app.include_router(routes_analysis.router)
app.include_router(routes_risk.router)
app.include_router(routes_sessions.router)
app.include_router(routes_incidents.router)


@app.get("/")
async def root() -> dict:
    """Root endpoint — basic API info and links."""
    return {
        "name": "DriveGuard AI",
        "description": "Multimodal Real-Time Driver Risk & Road Intelligence System",
        "docs": "/docs",
        "health": "/health",
    }
