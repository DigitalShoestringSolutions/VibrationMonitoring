from fastapi import FastAPI
from .tasks import process_vibration_data
from .influx import influx_service

app = FastAPI()

@app.post("/analyze")
async def trigger_analysis(data: dict):
    task = process_vibration_data.delay(data)
    return {"task_id": task.id}

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down"""
    influx_service.close()
