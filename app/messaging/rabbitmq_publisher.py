# rabbitmq_publisher.py

# Publishes the prediction results back.

# prediction_service
#        ↓
# rabbitmq_publisher.py
#        ↓
# ML Result Queue
#        ↓
# Application Backend

# For example:

# className
# satdPrediction
# satdProbability
# bugPrediction
# bugProbability