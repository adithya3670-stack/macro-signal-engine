import os
import sys

# Ensure we can import from analysis
sys.path.append(os.getcwd())

from analysis.portfolio_manager import PortfolioManager

print("Initializing Manager...")
try:
    pm = PortfolioManager()
    print(f"Manager initialized. Data Dir: {pm.data_dir}")
    print(f"File Path: {pm.file_path}")
    
    print("Attempting to save test profile...")
    config = {"test": "data", "value": 123}
    pm.save_profile("Test Profile", config)
    print("Save successful.")
    
    print("Attempting to load...")
    prof = pm.get_profile("Test Profile")
    print(f"Loaded: {prof}")
    
    if prof and prof['test'] == 'data':
        print("VERIFICATION PASSED: Read/Write works.")
    else:
        print("VERIFICATION FAILED: Data mismatch.")
        
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
