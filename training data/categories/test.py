import torch

# Load the .pt file
# For security, consider using weights_only=True if you are only loading model weights
# and don't need to execute arbitrary code within the file.
loaded_object = torch.load("emogator_cats.pt", weights_only=True) 

# Now you can inspect the loaded_object
print(type(loaded_object)) 
print(loaded_object) 

# If it's a state_dict (model weights), you can iterate through its items
if isinstance(loaded_object, dict):
    for key, value in loaded_object.items():
        print(f"Key: {key}, Type: {type(value)}, Shape: {value.shape}")

# <class 'list'>
# ['Adoration', 'Amusement', 'Anger', 'Awe', 'Confusion', 'Contempt', 'Contentment', 'Desire', 'Disappointment', 
#  'Disgust', 'Distress', 'Ecstasy', 'Elation', 'Embarrassment', 'Fear', 'Guilt', 'Interest', 'Neutral', 'Pain', 
#  'Pride', 'Realization', 'Relief', 'Romantic Love', 'Sadness', 'Serenity', 'Shame', 'Surprise (Negative)', 
#  'Surprise (Positive)', 'Sympathy', 'Triumph']