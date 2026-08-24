# 3. services/prediction_service.py

# This should be the main ML orchestration class.

# It coordinates:

# PredictionService
#        │
#        ├── Preprocessor
#        │
#        ├── SATD Model
#        │
#        └── Bug Model

# For example:

# Input
#   ↓
# Preprocessing
#   ↓
# Feature Vector
#   ↓
# SATD Model
#   ↓
# Bug Model
#   ↓
# Prediction Response

# The service should not contain all preprocessing code or model-loading code directly.