# rabbitmq_consumer.py

# Receives the data from the Application Backend.

# Your actual flow:

# Application Backend
#         │
#         ▼
#    ML Job Queue
#         │
#         ▼
# rabbitmq_consumer.py

# The message could contain:

# analysisJobId
# repositoryId
# class information
# metrics
# comments

# The consumer should then pass that data to:

# prediction_service.py

# It should not contain the ML logic itself.