# api/routes/prediction.py

# This is the one part of your original structure that I would not consider essential.

# Your actual production flow is:

# Application Backend
#        ↓
# RabbitMQ
#        ↓
# FastAPI ML Service

# not:

# Application Backend
#        ↓
# POST /predict
#        ↓
# FastAPI

# Therefore, you don't need an HTTP /predict endpoint for the main system.

# However, I would still keep an API layer for:
# Health checks
# Testing
# Debugging
# Docker/Kubernetes health probes

# For example:

# GET /health

# You could therefore change it to:

# api/
# └── routes/
#     └── health.py

# If you want a manual prediction endpoint for development/testing, then keeping:

# prediction.py

# is perfectly fine.