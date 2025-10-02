#!/usr/bin/env python3
import json
import csv
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Union

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file."""
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            else:
                data = [content]
    
    return data

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, json.dumps(v, ensure_ascii=False)))
        else:
            items.append((new_key, v))
    
    return dict(items)

def convert_to_csv(json_file: str, csv_file: str = None) -> str:
    """Convert JSON/JSONL file to CSV."""
    if csv_file is None:
        csv_file = os.path.splitext(json_file)[0] + '.csv'
    
    data = load_json_data(json_file)
    
    if not data:
        print(f"No data found in {json_file}")
        return csv_file
    
    # Handle case where JSON is a dictionary with IDs as keys
    if len(data) == 1 and isinstance(data[0], dict):
        json_dict = data[0]
        # Check if this looks like an ID-based structure
        first_key = next(iter(json_dict.keys()))
        if isinstance(json_dict[first_key], dict):
            # Transform ID-based structure to row-based
            rows = []
            for item_id, item_data in json_dict.items():
                row = {'id': item_id}
                row.update(flatten_dict(item_data))
                rows.append(row)
            data = rows
    
    if not isinstance(data, list):
        data = [data]
    
    flattened_data = [flatten_dict(item) if isinstance(item, dict) else item for item in data]
    
    # Preserve order of keys from the first item, then add any additional keys
    fieldnames = []
    all_keys = set()
    
    if flattened_data and isinstance(flattened_data[0], dict):
        # Use the order from the first item
        fieldnames = list(flattened_data[0].keys())
        all_keys.update(fieldnames)
        
        # Add any additional keys from other items
        for item in flattened_data[1:]:
            if isinstance(item, dict):
                for key in item.keys():
                    if key not in all_keys:
                        fieldnames.append(key)
                        all_keys.add(key)
    
    # Increase CSV field size limit to handle long response texts
    csv.field_size_limit(sys.maxsize)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in flattened_data:
            if isinstance(item, dict):
                writer.writerow(item)
    
    return csv_file

def main():
    parser = argparse.ArgumentParser(description='Convert JSON/JSONL files to CSV')
    parser.add_argument('input', help='Input JSON or JSONL file path')
    parser.add_argument('-o', '--output', help='Output CSV file path (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} does not exist")
        return
    
    try:
        output_file = convert_to_csv(args.input, args.output)
        print(f"Successfully converted {args.input} to {output_file}")
    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    main()