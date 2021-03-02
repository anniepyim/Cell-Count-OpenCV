#!/bin/bash
pid=`ps ax | grep gunicorn | grep cell-count-opencv | awk '{split($0,a," "); print a[1]}' | head -n 1`
while [ "$pid" ]
do
  kill $pid
  echo "killed gunicorn deamon on port with pid $pid"
  pid=`ps ax | grep gunicorn | grep cell-count-opencv | awk '{split($0,a," "); print a[1]}' | head -n 1`
done