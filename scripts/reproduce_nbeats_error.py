from analysis.deep_learning_model import DLMacroModel
import traceback
import sys

with open("debug_log.txt", "w") as log:
    sys.stdout = log
    sys.stderr = log
    
    print("--- Testing N-BEATS Prediction ---")
    try:
        dl = DLMacroModel()
        print("Model initialized.")
        
        print("Calling predict_latest('nbeats')...")
        results = dl.predict_latest(model_type='nbeats')
        
        print("Prediction Result:", results)

    except Exception as e:
        print("\nCRITICAL ERROR:")
        traceback.print_exc()
