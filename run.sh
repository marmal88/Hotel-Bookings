#!/bin/sh

gunicorn src.main:server -b 0.0.0.0:6006
