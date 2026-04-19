
import os

file_path = r"c:\Users\adith\GIT_Repo\macroEconomic\routes\training.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False

# The block we want to replace starts at roughly line 124
# and ends at line 165
# We will identify start by "# 2. DISCOVERY"
# We will identify end by "# 3. PRODUCTION" block end, roughly around "manual_epochs_dict=manual_epochs,"

# New content to insert
new_block = """                 # 2. ENSEMBLE TRAINING (70%)
                 # ---------------------------------------------------------
                 def ens_cb(pct, msg):
                     # Map 0-100 -> 30-100
                     overall = 30 + (pct * 0.7)
                     q_callback(overall, f"[ENS] {msg}")
                 
                 q_callback(30, f"Training 3-Fold Bagging Ensemble...")
                 # Train 3 models per asset using Gap Bagging
                 dl.train_all_models(
                     model_type=model_type, 
                     epochs=epochs, 
                     train_cutoff_date=cutoff, 
                     use_bagging_ensemble=True,
                     progress_callback=ens_cb
                 )
"""
# Note: The indentation of new_block must match the file. 
# Looking at view_file, indentation seems to be roughly 17 spaces?
# 8 (worker) + 5 (try) + 4 = 17?
# Let's inspect the first line of the targeted block in `lines` to get exact indentation.

start_marker = "2. DISCOVERY (20%)"
end_marker = "manual_epochs_dict=manual_epochs,"

inserted = False
i = 0
while i < len(lines):
    line = lines[i]
    
    if start_marker in line and not inserted:
        # Found the start
        # Calculate indentation from this line
        indent = line[:line.find('#')]
        
        # Adjust new_block indentation
        adjusted_block = ""
        for bline in new_block.split('\n'):
            if bline.strip():
                # remove simplistic indentation from string and add correct indent
                # But my string has generic indentation.
                # Let's just use the indentation of the found line.
                # The inner lines need more indent?
                # My new_block is already indented relatively.
                # Let's just blindly use the hardcoded indentation I estimated (17 spaces) 
                # or just use the string as is if I matched it well.
                # Let's simpler: Replace lines i to i+40 (approx).
                
                # Scan ahead to find end of Production block
                # Production block ends at `progress_callback=prod_cb` usually + 1 paren line.
                
                # Actually, I'll search for the *Start* of "3. PRODUCTION" as well.
                pass
        
        # We need to skip until we find the end of the Production block.
        # Production block ends at `progress_callback=prod_cb`
        # Then `)`
        # Then empty line?
        
        # Let's loop ahead
        j = i
        while j < len(lines):
            if "progress_callback=prod_cb" in lines[j]:
                # Found end of production call
                # Skip 2 more lines for `)` and blank?
                j += 2
                break
            j += 1
        
        # Insert New Block
        # Adjust indentation of new_block to match `indent`
        # My `new_block` above has ~17 spaces.
        # Let's assume `indent` is the source of truth.
        
        # Construct new block using `indent`
        # Base indent = `indent`
        # Inner indent = `indent` + 4 spaces
        
        base_indent = indent
        inner_indent = indent + "    "
        
        final_block_lines = [
            f"{base_indent}# 2. ENSEMBLE TRAINING (70%)\n",
            f"{base_indent}# ---------------------------------------------------------\n",
            f"{base_indent}def ens_cb(pct, msg):\n",
            f"{inner_indent}# Map 0-100 -> 30-100\n",
            f"{inner_indent}overall = 30 + (pct * 0.7)\n",
            f"{inner_indent}q_callback(overall, f\"[ENS] {{msg}}\")\n",
            f"\n",
            f"{base_indent}q_callback(30, f\"Training 3-Fold Bagging Ensemble...\")\n",
            f"{base_indent}# Train 3 models per asset using Gap Bagging\n",
            f"{base_indent}dl.train_all_models(\n",
            f"{inner_indent}model_type=model_type, \n",
            f"{inner_indent}epochs=epochs, \n",
            f"{inner_indent}train_cutoff_date=cutoff, \n",
            f"{inner_indent}use_bagging_ensemble=True,\n",
            f"{inner_indent}progress_callback=ens_cb\n",
            f"{base_indent})\n"
        ]
        
        new_lines.extend(final_block_lines)
        
        i = j # Skip to end of old block
        inserted = True
    else:
        new_lines.append(line)
        i += 1

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Successfully updated training.py")
