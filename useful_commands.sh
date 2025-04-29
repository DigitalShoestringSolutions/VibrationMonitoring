#!/bin/bash
# Define tasks
docker exec vibrationmonitoring-kapacitor-1 kapacitor define etl_task -type stream -tick /home/kapacitor/etl.tick
# Enable tasks
docker exec vibrationmonitoring-kapacitor-1 kapacitor enable etl_task
# for testing api on data
curl -X POST http://localhost:8000/analyze   -H "Content-Type: application/json"   -d '{"start_time": "2025-04-24T15:45:42Z", "end_time": "2025-04-24T15:45:43Z"}'

curl -X POST http://localhost:8000/train   -H "Content-Type: application/json"   -d '{"start_time": "2025-04-24T15:45:42Z", "end_time": "2025-04-24T15:45:43Z"}'