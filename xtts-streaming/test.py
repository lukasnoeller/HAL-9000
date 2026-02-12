import sys
import os

# Ensure the model directory is in the path
sys.path.append(os.getcwd())

from model.model import Model

print("--- Initializing Model Class ---")
# Truss usually passes a data_dir kwarg pointing to /model/data
model = Model(data_dir="/app/data") 

print("--- Calling load() method ---")
try:
    model.load()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Load failed with error: {e}")
    import traceback
    traceback.print_exc()