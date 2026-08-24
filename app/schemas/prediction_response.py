# prediction_response.py

# Represents the results:

# PredictionResponse
# ├── analysisJobId
# └── predictions
#     ├── className
#     ├── SATD prediction
#     ├── SATD probability
#     ├── bug prediction
#     └── bug probability

# Since you're using RabbitMQ, these schemas are message schemas, not necessarily HTTP request/response models.