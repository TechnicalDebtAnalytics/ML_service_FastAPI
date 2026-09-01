# feature_preprocessor.py

# For your SATD model, this could handle things like:

# Text cleaning
# Tokenization/preparation
# Word TF-IDF
# Character TF-IDF
# Keyword features
# Combining feature vectors

# This is especially important because your trained model expects the same feature representation used during training.

# Your pipeline was essentially:

# Comments
#     ↓
# Word TF-IDF
#     +
# Character TF-IDF
#     +
# SATD keyword features
#     ↓
# Combined feature vector
#     ↓
# LinearSVC

# So the production preprocessing must reproduce that process exactly.