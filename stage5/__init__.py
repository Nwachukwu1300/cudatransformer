"""Stage 5: The recommendation pivot.

Reuses the Stage 3 transformer unchanged, pointed at movie item IDs instead of
word tokens, to predict the next item in a user's interaction history.
"""
