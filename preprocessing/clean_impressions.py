#!/usr/bin/env python3
"""
Script to clean up impression text in merlin_impressions.csv to make it concise and uniform.
"""

import pandas as pd
import re


def clean_impression_text(text):
    """Clean impression text to be concise and uniform."""
    if pd.isna(text) or not text.strip():
        return text
    
    # Extract text after "FINDINGS:" or "IMPRESSION:" (case-insensitive)
    findings_match = re.search(r'findings:\s*(.*)', text, flags=re.IGNORECASE | re.DOTALL)
    impression_match = re.search(r'impression:\s*(.*)', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Use the last occurrence (impression takes priority if both exist)
    if impression_match:
        text = impression_match.group(1).strip()
    elif findings_match:
        text = findings_match.group(1).strip()
    
    # Remove common prefixes that might still be there
    text = re.sub(r'^(Okay, based on the provided findings, here\'s a concise impression:\s*)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(Here\'s a concise impression based on the provided findings:\s*)', '', text, flags=re.IGNORECASE)
    
    # Remove numbered lists (1., 2., etc.) and **bold** formatting
    text = re.sub(r'^\s*\d+\.\s*\*\*[^*]+\*\*:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Remove **bold** markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    
    # Remove all asterisks (*)
    text = re.sub(r'\*', '', text)
    
    # Remove category labels followed by colons (e.g., "Pyelonephritis:", "Lymphadenopathy:")
    text = re.sub(r'\b[A-Z][a-zA-Z\s]*:\s*', '', text)
    
    # Remove bullet points and list markers
    text = re.sub(r'^\s*[-•]\s*', '', text, flags=re.MULTILINE)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove trailing periods followed by more content that looks like metadata
    text = re.sub(r'\.\s*(There are no substantial differences|Findings discussed|Preliminary results).*$', '.', text, flags=re.IGNORECASE)
    
    # Split into sentences and keep only the most relevant clinical findings
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Filter out metadata sentences
    filtered_sentences = []
    for sentence in sentences:
        # Skip sentences that are clearly metadata or administrative
        if any(phrase in sentence.lower() for phrase in [
            'there are no substantial differences',
            'preliminary results',
            'findings discussed',
            'this change in report',
            'final report',
            'treating clinician',
            'phone by',
            'deleted',
            'at time on date'
        ]):
            continue
        filtered_sentences.append(sentence)
    
    # Remove duplicate sentences while preserving order
    unique_sentences = []
    seen = set()
    for sentence in filtered_sentences:
        # Normalize sentence for comparison (lowercase, strip whitespace)
        normalized = sentence.lower().strip()
        if normalized not in seen and normalized:  # Skip empty sentences
            seen.add(normalized)
            unique_sentences.append(sentence)
    
    # Rejoin the unique sentences
    if unique_sentences:
        cleaned_text = '. '.join(unique_sentences)
        if not cleaned_text.endswith('.'):
            cleaned_text += '.'
        return cleaned_text
    
    return text


def main():
    # Read the CSV file
    print("Loading merlin_impressions.csv...")
    df = pd.read_csv('/path/to/data/merlin_impressions.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean all impressions
    print("Cleaning all impressions...")
    df['Impressions_EN_cleaned'] = df['Impressions_EN'].apply(clean_impression_text)
    
    # Calculate statistics
    original_lengths = df['Impressions_EN'].str.len()
    cleaned_lengths = df['Impressions_EN_cleaned'].str.len()
    reductions = original_lengths - cleaned_lengths
    reduction_pcts = (reductions / original_lengths * 100)
    
    print(f"\n=== CLEANING STATISTICS ===")
    print(f"Total impressions processed: {len(df)}")
    print(f"Average original length: {original_lengths.mean():.1f} characters")
    print(f"Average cleaned length: {cleaned_lengths.mean():.1f} characters")
    print(f"Average reduction: {reductions.mean():.1f} characters ({reduction_pcts.mean():.1f}%)")
    print(f"Total characters saved: {reductions.sum()}")
    
    # Save the cleaned version
    output_path = '/path/to/data/merlin_impressions_cleaned.csv'
    print(f"\nSaving cleaned CSV to: {output_path}")
    df.to_csv(output_path, index=False)
    
    print("✅ Cleaning completed successfully!")
    
    # Show a few examples
    print(f"\n=== SAMPLE CLEANED IMPRESSIONS ===")
    for i in range(min(3, len(df))):
        print(f"\n{i+1}. {df.iloc[i]['VolumeName']}:")
        print(f"CLEANED: {df.iloc[i]['Impressions_EN_cleaned']}")
        print("-" * 60)


if __name__ == "__main__":
    main()