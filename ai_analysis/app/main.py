from fastapi import FastAPI
from .tasks import process_vibration_data

app = FastAPI()

@app.post("/analyze")
async def trigger_analysis(data: dict):
    task = process_vibration_data.delay(data)
    return {"task_id": task.id}
