import os
import json

def list_models():
    print("--- Starting Debug ---")
    base_dirs = {
        'ml': 'models/holdout',
        'dl': 'models/holdout_dl'
    }
    
    models = []
    
    print(f"Base Dirs: {base_dirs}")
    print(f"CWD: {os.getcwd()}")
    
    for category, path in base_dirs.items():
        print(f"Checking Category: {category} at Path: {path}")
        if not os.path.exists(path): 
            print(f"  Path does not exist: {path}")
            continue
        
        print(f"  Path exists. Listing contents...")
        for year in os.listdir(path):
            year_path = f"{path}/{year}"
            print(f"  Checking Item: {year} -> {year_path}")
            
            manifest_path = f"{year_path}/manifest.json"
            print(f"    Manifest Path: {manifest_path}")
            
            if os.path.exists(manifest_path):
                print("    Manifest FOUND.")
                meta = None
                try:
                    with open(manifest_path, 'r') as f:
                        meta = json.load(f)
                        # Enrich
                        meta['id'] = year 
                        meta['category'] = category
                        print(f"    Loaded Meta: {meta}")
                except Exception as e:
                    print(f"    Error loading manifest: {e}")
                    continue
                    
                # Add stats if available
                try:
                    metrics_path = f"{year_path}/model_metrics.json"
                    print(f"    Checking Metrics: {metrics_path}")
                    if os.path.exists(metrics_path):
                         with open(metrics_path, 'r') as mf:
                             mdata = json.load(mf)
                             meta['train_end'] = mdata.get('train_end_date')
                             print("    Metrics Loaded.")
                    else:
                        print("    Metrics NOT found.")
                except Exception as e:
                    print(f"    Error loading metrics: {e}")
                    
                models.append(meta)
            else:
                print("    Manifest NOT found.")

    print(f"Total Models Found: {len(models)}")
    return models

if __name__ == "__main__":
    list_models()
