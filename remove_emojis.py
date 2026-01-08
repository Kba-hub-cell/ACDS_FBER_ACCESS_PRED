#!/usr/bin/env python3
"""
Script pour supprimer tous les emojis du notebook Jupyter
"""
import json
import re

# Lire le notebook
notebook_path = 'notebooks/Fiber_Acces_pred.ipynb'
print(f"Lecture du notebook: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Pattern pour matcher les emojis
emoji_pattern = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE
)

total_cells = 0
total_emojis = 0

# Parcourir toutes les cellules
for cell in nb['cells']:
    if 'source' in cell:
        cell_modified = False
        for i, line in enumerate(cell['source']):
            original_line = line
            cleaned_line = emoji_pattern.sub('', line)
            
            if original_line != cleaned_line:
                cell['source'][i] = cleaned_line
                emojis_removed = len(original_line) - len(cleaned_line)
                total_emojis += emojis_removed
                cell_modified = True
        
        if cell_modified:
            total_cells += 1

# Sauvegarder le notebook modifié
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n[SUCCESS] Optimisation terminée!")
print(f"  - Cellules modifiées: {total_cells}")
print(f"  - Caractères supprimés (emojis): {total_emojis}")
print(f"  - Notebook sauvegardé: {notebook_path}")
