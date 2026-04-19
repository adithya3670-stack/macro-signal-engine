
import inspect
from routes.ensembling import api_generate_winner_ensemble
import sys

print("Inspecting api_generate_winner_ensemble source:")
src = inspect.getsource(api_generate_winner_ensemble)
print(src)

if "winner_ensemble_{metric}.json" in src:
    print("SUCCESS: Code on disk is updated.")
else:
    print("FAILURE: Code on disk is OLD.")
