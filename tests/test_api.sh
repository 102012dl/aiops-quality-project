#!/bin/bash
API_URL="http://localhost:8000"

echo "Testing AIOps API..."
curl -s $API_URL/health | jq .
echo ""
curl -s -X POST $API_URL/predict -H "Content-Type: application/json" -d '{"data": [1,2,3]}' | jq .
