#!/bin/bash
# Define tasks
docker exec vibrationmonitoring-kapacitor-1 kapacitor define etl_task \
    -type stream \
    -tick /home/kapacitor/etl.tick
# Enable tasks
docker exec vibrationmonitoring-kapacitor-1 kapacitor enable etl_task