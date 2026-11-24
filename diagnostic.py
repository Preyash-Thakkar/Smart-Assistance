import glob
import json
import os
import re

def extract_dicts(data):
    if isinstance(data, dict): 
        yield data
    elif isinstance(data, list):
        for item in data: 
            yield from extract_dicts(item)

def main():
    print("Scanning source mappings...\n")
    batch_files = glob.glob("Config Assist Multiples/**/batch*_source_mapping.json", recursive=True)
    
    mapping = {}
    for filepath in batch_files:
        # Extract just "batchX" from the path
        match = re.search(r'(batch\d+)_', filepath)
        if not match: continue
        batch_name = match.group(1)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            dicts = []
            try:
                dicts.extend(list(extract_dicts(json.loads(content))))
            except:
                pass
            
            found_q = "MISSING/UNKNOWN"
            for d in dicts:
                if "source_complaint_record" in d:
                    found_q = d["source_complaint_record"].get("question", "MISSING/UNKNOWN")
                    break
            
            mapping[batch_name] = found_q.strip()
        except Exception as e:
            mapping[batch_name] = f"ERROR: {e}"

    # Sort numerically (batch1, batch2... batch10)
    sorted_batches = sorted(mapping.items(), key=lambda x: int(x[0].replace('batch', '')))
    
    for b, q in sorted_batches:
        print(f"'{b}': '{q}',")

if __name__ == "__main__":
    main()