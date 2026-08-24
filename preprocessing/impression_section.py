import os
import json
import pandas as pd
import random
import argparse
from transformers import pipeline
import torch
from tqdm import tqdm


def load_complete_pairs(train_csv_path):
    """Load all complete findings-impression pairs from train_reports.csv."""
    # Read the training data
    df = pd.read_csv(train_csv_path)
    
    # Filter for complete findings-impression pairs (non-empty and not "Not given")
    complete_pairs = df[
        (df['Impressions_EN'].notna()) & 
        (df['Impressions_EN'] != '') & 
        (df['Impressions_EN'].str.lower() != 'not given') &
        (df['Impressions_EN'].str.lower() != 'not given.')
    ]
    
    return complete_pairs[['Findings_EN', 'Impressions_EN']]

def sample_few_shot_examples(complete_pairs_df, num_examples=8):
    """Randomly sample few-shot examples from the complete pairs."""
    if len(complete_pairs_df) < num_examples:
        print(f"Warning: Only {len(complete_pairs_df)} complete pairs available, using all of them.")
        num_examples = len(complete_pairs_df)
    
    # Sample without a fixed random state to get different examples each time
    few_shot_examples = complete_pairs_df.sample(n=num_examples)
    
    return few_shot_examples.to_dict('records')

def create_few_shot_prompt(findings_text, few_shot_examples):
    """Create a prompt with few-shot examples for impression generation."""
    prompt = "You are an experienced radiologist. Based on the following examples, generate a concise impression from the given findings.\n\n"
    
    # Add few-shot examples
    for i, example in enumerate(few_shot_examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Findings: {example['Findings_EN']}\n"
        prompt += f"Impression: {example['Impressions_EN']}\n\n"
    
    # Add the current case
    prompt += f"Now, act as a radiologist, based on the pattern from the examples above, generate an impression for the following findings:\n\n"
    prompt += f"Findings: {findings_text}\n"
    prompt += f"Impression:"
    
    return prompt

def process_target_file(target_csv_path, complete_pairs_df, num_examples, output_path, max_new_tokens=8192):
    """Process the entire CSV file - keep existing impressions, generate new ones for empty/missing cases."""
    # Read the target data
    df = pd.read_csv(target_csv_path)
    
    # Identify rows that need impression generation
    needs_generation = (
        (df['Impressions_EN'].isna()) | 
        (df['Impressions_EN'] == '') | 
        (df['Impressions_EN'].str.lower() == 'not given') |
        (df['Impressions_EN'].str.lower() == 'not given.')
    )
    
    incomplete_rows = df[needs_generation]
    complete_rows = df[~needs_generation]
    
    print(f"Total rows: {len(df)}")
    print(f"Rows with existing impressions: {len(complete_rows)}")
    print(f"Rows needing impression generation: {len(incomplete_rows)}")
    
    if len(incomplete_rows) == 0:
        print("No rows need impression generation. Saving original file.")
        # Just copy the original data to output
        df.to_csv(output_path.replace('.json', '.csv'), index=False)
        return
    
    # Initialize the transformers pipeline only if needed
    pipe = pipeline(
        "text-generation",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_kwargs={
            "low_cpu_mem_usage": True,
        }
    )
    
    # Create a copy of the dataframe to modify
    result_df = df.copy()
    generation_log = []
    
    for idx, row in tqdm(incomplete_rows.iterrows(), total=len(incomplete_rows), desc="Generating impressions"):
        findings = row['Findings_EN']
        
        if pd.isna(findings) or findings.strip() == '':
            print(f"Skipping row {idx}: No findings available")
            continue
        
        # Sample new few-shot examples for each case
        few_shot_examples = sample_few_shot_examples(complete_pairs_df, num_examples)
        
        # Create the prompt with few-shot examples
        prompt = create_few_shot_prompt(findings, few_shot_examples)
        
        try:
            # Create messages in the format expected by medgemma
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful medical assistant and expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
            
            # Generate using transformers pipeline
            output = pipe(messages, max_new_tokens=max_new_tokens)
            generated_impression = output[0]["generated_text"][-1]["content"].strip()
            
            # Update the dataframe with the generated impression
            result_df.at[idx, 'Impressions_EN'] = generated_impression
            
            # Log the generation for reference
            log_entry = {
                "row_index": idx,
                "volume_name": row['VolumeName'],
                "original_findings": findings,
                "generated_impression": generated_impression,
                "original_impression": row['Impressions_EN'] if pd.notna(row['Impressions_EN']) else "Not given",
                # "few_shot_examples_used": [ex['Impressions_EN'][:50] + "..." for ex in few_shot_examples]
            }
            generation_log.append(log_entry)
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue
    
    # Save the complete updated CSV
    csv_output_path = output_path.replace('.json', '.csv')
    result_df.to_csv(csv_output_path, index=False)
    
    # Save the generation log as JSON
    with open(output_path, 'w') as f:
        json.dump(generation_log, f, indent=2)
    
    print(f"Complete dataset with generated impressions saved to {csv_output_path}")
    print(f"Generation log saved to {output_path}")
    print(f"Successfully generated impressions for {len(generation_log)} cases.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Complete CSV with generated impressions using few-shot learning')
    parser.add_argument('--target_file', type=str, required=True, 
                        help='Path to the target CSV file (e.g., validation_reports.csv)')
    parser.add_argument('--output_file', type=str, default='completed_reports.json',
                        help='Output file for generation log (CSV will be saved with .csv extension)')
    parser.add_argument('--num_examples', type=int, default=8,
                        help='Number of few-shot examples to use')
    parser.add_argument('--max_new_tokens', type=int, default=8192,
                        help='Maximum number of new tokens to generate')
    
    args = parser.parse_args()
    
    # Don't set a global random seed to allow resampling for each case
    # Each call to sample_few_shot_examples will use different examples
    
    # Load all complete pairs from train_reports.csv
    train_csv_path = '/path/to/ACT/model/data/ct_rate/train_reports.csv'
    complete_pairs_df = load_complete_pairs(train_csv_path)
    
    print(f"Loaded {len(complete_pairs_df)} complete pairs from training data.")
    print(f"Will resample {args.num_examples} examples for each incomplete case.")
    
    # Process the target file
    process_target_file(args.target_file, complete_pairs_df, args.num_examples, args.output_file, args.max_new_tokens)
        