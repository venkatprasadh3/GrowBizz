#!/bin/bash
apt-get update
apt-get install -y libpango-1.0-0 libpangocairo-1.0-0 libcairo2
pip install -r requirements.txt
