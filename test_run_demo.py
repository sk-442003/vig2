"""Smoke test for demo fusion behavior"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from src.app import predict_all
print('Running smoke test predict_all with text-only input...')
res = predict_all(None, None, 'I am very sad and upset')
print(json.dumps(res, indent=2))
