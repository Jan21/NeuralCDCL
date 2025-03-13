#!/usr/bin/env python3
import json
import os
import glob
import argparse
import re
import random


def parse_filename(filename):
    """Extract parameters from filename using regex."""
    pattern = r"cdcl_(train|test)_vars(\d+)_coef_([0-9\.]+)\.json"
    match = re.search(pattern, filename)
    if match:
        data_type = match.group(1)  # 'train' or 'test'
        num_vars = int(match.group(2))
        coefficient = float(match.group(3))
        return data_type, num_vars, coefficient
    return None, None


def main():
    parser = argparse.ArgumentParser(description='Join all JSON files into one file, with exclusion options')
    parser.add_argument('--input_dir', type=str, default='temp/train',
                        help='Directory containing the data files (default: temp/train)')
    parser.add_argument('--test_dir', type=str, default='temp/test',
                        help='Directory containing the test files (default: temp/test)')
    parser.add_argument('--output_file', type=str, default='combined_train_data.json',
                        help='Output JSON file name for training data (default: combined_train_data.json)')
    parser.add_argument('--test_output_file', type=str, default='combined_test_data.json',
                        help='Output JSON file name for test data (default: combined_test_data.json)')
    parser.add_argument('--combine_test', action='store_true',
                        help='Also combine test data files')
    parser.add_argument('--test_samples', type=int, default=3,
                        help='Number of samples to take from each test file (default: 20)')
    parser.add_argument('--exclude_vars', type=int, nargs='+',
                        help='Exclude files with these num_vars values')
    parser.add_argument('--exclude_coefs', type=float, nargs='+',
                        help='Exclude files with these coefficient values')
    parser.add_argument('--exclude_combinations', type=str, nargs='+',
                        help='Exclude specific combinations in format "num_vars:coef" (e.g., "7:4.2")')
    
    args = parser.parse_args()
    
    # Create excluded combinations from args
    excluded_combinations = set()
    
    # Add specific combinations
    if args.exclude_combinations:
        for combo in args.exclude_combinations:
            try:
                vars_val, coef_val = combo.split(':')
                excluded_combinations.add((int(vars_val), float(coef_val)))
            except (ValueError, TypeError):
                print(f"Warning: Couldn't parse combination '{combo}'. Expected format 'num_vars:coef'")
    
    # Process function to handle both train and test files
    def process_files(input_dir, output_file, is_test=False, samples_per_file=20):
        # Get all JSON files in the specified directory
        json_files = glob.glob(os.path.join(input_dir, '*.json'))
        
        combined_data = []
        excluded_files = []
        file_sample_counts = {}
        
        for json_file in json_files:
            # Extract parameters from filename
            data_type, num_vars, coefficient = parse_filename(json_file)
            
            # Skip if we couldn't parse the filename
            if num_vars is None or coefficient is None:
                print(f"Warning: Couldn't parse parameters from filename: {json_file}")
                continue
            
            # Check if this file should be excluded
            exclude_file = False
            
            # Check by num_vars
            if args.exclude_vars and num_vars in args.exclude_vars:
                exclude_file = True
                
            # Check by coefficient
            if args.exclude_coefs and any(abs(coefficient - c) < 1e-6 for c in args.exclude_coefs):
                exclude_file = True
                
            # Check by specific combination
            if (num_vars, coefficient) in excluded_combinations:
                exclude_file = True
            
            if exclude_file:
                excluded_files.append(os.path.basename(json_file))
                continue
            
            # Read and combine JSON data
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                    # For test files, sample only a specific number of examples
                    if is_test and len(data) > samples_per_file:
                        sampled_data = random.sample(data, samples_per_file)
                        combined_data.extend(sampled_data)
                        file_sample_counts[os.path.basename(json_file)] = samples_per_file
                    else:
                        combined_data.extend(data)
                        file_sample_counts[os.path.basename(json_file)] = len(data)
                        
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error reading {json_file}: {e}")
    
        # Write combined data to output file
        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        # Output additional info for test data
        included_files = len(json_files) - len(excluded_files)
        if is_test:
            sample_info = ", ".join([f"{file}: {count}" for file, count in sorted(file_sample_counts.items())])
            print(f"Combined {len(combined_data)} records from {included_files} files (sampling {samples_per_file} per file)")
            print(f"Samples per file: {sample_info}")
        else:
            print(f"Combined {len(combined_data)} records from {included_files} files")
            
        print(f"Output written to {output_file}")
        
        if excluded_files:
            print(f"Excluded {len(excluded_files)} files: {', '.join(excluded_files)}")
        
        return len(combined_data), included_files, file_sample_counts
    
    # Process training data
    train_records, train_files, train_counts = process_files(args.input_dir, args.output_file, is_test=False)
    
    # Process test data if requested
    if args.combine_test:
        test_records, test_files, test_counts = process_files(
            args.test_dir, 
            args.test_output_file, 
            is_test=True, 
            samples_per_file=args.test_samples
        )
        
        # Calculate total test files with sampling info
        expected_samples = test_files * args.test_samples
        print(f"\nSummary:")
        print(f"- Train: {train_records} records from {train_files} files")
        print(f"- Test: {test_records} records from {test_files} files (sampled {args.test_samples} from each)")
        print(f"  Expected test samples if all files had {args.test_samples} samples: {expected_samples}")
        
        # Report if any test files had fewer than the requested samples
        files_with_fewer = [f for f, count in test_counts.items() if count < args.test_samples]
        if files_with_fewer:
            print(f"  Note: {len(files_with_fewer)} files had fewer than {args.test_samples} samples available")


if __name__ == "__main__":
    main()