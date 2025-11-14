from fastapi import FastAPI, Request
from inference import model_fn, input_fn, predict_fn, output_fn
import uvicorn

app = FastAPI()
model = model_fn("/opt/ml/model")

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
async def invocations(request: Request):
    body = await request.body()
    parsed = input_fn(body, "application/json")
    prediction = predict_fn(parsed, model)
    return output_fn(prediction, "application/json")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
