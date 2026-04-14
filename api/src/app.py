"""
FastAPI application — Forging Line Delay Diagnostics API.
Diagnosis logic lives in diagnose.py (pure function).
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from diagnose import diagnose


# ---------------------------------------------------------------------------
# Startup: load reference times ONCE
# ---------------------------------------------------------------------------

REFERENCE_TIMES: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global REFERENCE_TIMES
    ref_path = Path(__file__).parent.parent / "reference_times.json"
    with open(ref_path) as f:
        REFERENCE_TIMES = json.load(f)
    yield


app = FastAPI(
    title="Forging Line Delay Diagnostics API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"error": "invalid request body"})


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PieceRequest(BaseModel):
    piece_id: str
    die_matrix: int
    lifetime_2nd_strike_s: Optional[float] = None
    lifetime_3rd_strike_s: Optional[float] = None
    lifetime_4th_strike_s: Optional[float] = None
    lifetime_auxiliary_press_s: Optional[float] = None
    lifetime_bath_s: Optional[float] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/diagnose")
def diagnose_piece(body: PieceRequest):
    try:
        result = diagnose(body.model_dump(), REFERENCE_TIMES)
        return JSONResponse(content=result)
    except KeyError as e:
        message = e.args[0] if e.args else "unknown die_matrix"
        return JSONResponse(status_code=400, content={"error": message})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
