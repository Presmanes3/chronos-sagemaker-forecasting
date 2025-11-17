from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from inference import model_fn, input_fn, predict_fn, output_fn, log

MODEL_PATH = "/opt/ml/model/fine_tuned"  # Mounted model directory in SageMaker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to load and release resources (model, connections, etc.)"""
    log("Starting up — loading model...")
    try:
        app.state.model = model_fn(MODEL_PATH)
        log("Model loaded successfully ✅")
    except Exception as e:
        log(f"Startup failed: {e}", "ERROR")
        raise

    # Yield control to the app while running
    yield

    # Cleanup section (runs on shutdown)
    log("Shutting down — releasing resources.")


app = FastAPI(title="Chronos Bolt Inference API", lifespan=lifespan)


@app.get("/ping")
def ping():
    """Health check endpoint (required by SageMaker)."""
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(request: Request):
    """Main inference endpoint."""
    try:
        body = await request.body()
        ts_df = input_fn(body, "application/json")
        preds = predict_fn(ts_df, app.state.model)
        result = output_fn(preds, "application/json")
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        log(f"Inference request failed: {e}", "ERROR")
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    log("Starting FastAPI server on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
