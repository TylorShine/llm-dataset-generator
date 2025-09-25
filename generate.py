# generate.py

import os
import uuid
import json
import time
import argparse
import random
from pathlib import Path

import yaml
from jinja2 import Template
from openai import OpenAI, RateLimitError, APIError
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import nltk

# --- Text Chunking Helper ---

# def chunk_text(text: str, by: str, size: int) -> list[str]:
def chunk_text(text: str, by: str, size_min: int, size_max: int) -> list[str]:
    """
    Splits text into units (sentences, words, paragraphs) and groups them into chunks.
    """
    if not text or size_max <= 0:
        return []

    # 1. Split text into basic units
    if by == 'sentences':
        units = nltk.sent_tokenize(text)
    elif by == 'words':
        units = text.split()
    elif by == 'paragraphs':
        units = [p.strip() for p in text.split('\n\n') if p.strip()]
    elif by == 'lines':
        units = [p.strip() for p in text.split('\n') if p.strip()]
    else:
        return [text] # Should not happen with argparse choices

    # 2. Group units into chunks of the specified size
    chunks = []
    # for i in range(0, len(units), size):
    #     chunk_group = units[i:i + size]
    chunk_count = 0
    while chunk_count < len(units):
        rand_size = random.randint(size_min, size_max)
        chunk_group = units[chunk_count:chunk_count + rand_size]
        chunk_count += rand_size

        if by == 'paragraphs':
            chunks.append('\n\n'.join(chunk_group))
        if by == 'lines':
            chunks.append('\n'.join(chunk_group))
        else:
            chunks.append(' '.join(chunk_group))
    
    return chunks

# --- Configuration & Argument Parsing ---

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a dataset using an OpenAI-compatible API.")
    
    # File and Path Arguments
    parser.add_argument("--prompt-file", type=Path, required=True, help="Path to the YAML prompt template file.")
    parser.add_argument("--output-file", type=Path, required=True, help="Path to save the generated JSON dataset.")
    
    # Dataset Arguments
    parser.add_argument("--dataset-name", type=str, default="squad", help="Name of the Hugging Face dataset to use for prompts.")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'validation').")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate.")
    parser.add_argument("--start-index", type=int, default=0, help="Index of the dataset to start from.")
    
    # Splitting / Chunking Arguments
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--split-column", type=str, default=None, help="Name of a column to split by a separator.")
    split_group.add_argument("--chunk-column", type=str, default=None, help="Name of a column to chunk by a fixed number of units.")
    
    parser.add_argument("--split-separator", type=str, default="\n\n", help="The separator for --split-column.")
    # parser.add_argument("--chunk-size", type=int, default=3, help="The number of units per chunk for --chunk-column.")
    parser.add_argument("--chunk-size-min", type=int, default=3, help="The minimum number of units per chunk for --chunk-column.")
    parser.add_argument("--chunk-size-max", type=int, default=10, help="The maximum number of units per chunk for --chunk-column.")
    parser.add_argument("--chunk-by", type=str, default="lines", choices=['sentences', 'words', 'paragraphs', 'lines'], help="The unit to chunk by.")

    # API and Model Arguments
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Name of the model to use for generation.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation.")
    
    # Control and Logging
    parser.add_argument("--save-interval", type=int, default=10, help="How often to save the intermediate dataset.")
    return parser.parse_args()

# --- API Interaction with Retry Logic ---
def call_api_with_retry(client, model, system_prompt, user_prompt, max_tokens, temperature):
    max_retries = 5
    backoff_factor = 2
    wait_time = 1
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], max_tokens=max_tokens, temperature=temperature)
            return response.choices[0].message.content.strip()
        except (RateLimitError, APIError) as e:
            print(f"API/Rate limit error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
        time.sleep(wait_time)
        wait_time *= backoff_factor
    raise Exception("Failed to get a response from the API after multiple retries.")

# --- Data Loading and Processing ---
def load_prompt_template(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('system_prompt', ''), config['user_prompt_template']

def save_dataset(data: list, file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSuccessfully saved {len(data)} items to {file_path}")

# --- Main Generation Logic ---
def main():
    """Main function to orchestrate the dataset generation process."""
    args = parse_arguments()
    
    # One-time download for NLTK's sentence tokenizer
    print("Ensuring NLTK 'punkt' tokenizer is available...")
    try:
        nltk.data.find('tokenizers/punkt')
    # except nltk.downloader.DownloadError:
    except:
        nltk.download('punkt')
    print("NLTK setup complete.")
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL must be set in the .env file.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Initialized OpenAI client for model: {args.model}")
    
    system_prompt, user_prompt_str = load_prompt_template(args.prompt_file)
    user_template = Template(user_prompt_str)
    print(f"Loaded prompt template from: {args.prompt_file}")
    
    print(f"Loading dataset '{args.dataset_name}' (split: {args.dataset_split}) in streaming mode...")
    hf_dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True)
    dataset_iterator = iter(hf_dataset)
    
    if args.start_index > 0:
        print(f"Skipping to start index: {args.start_index}...")
        for _ in tqdm(range(args.start_index), desc="Skipping records"):
            next(dataset_iterator)

    generated_data = []
    print(f"Starting generation of {args.num_samples} samples...")
    pbar = tqdm(total=args.num_samples, desc="Generating Samples")
    
    while len(generated_data) < args.num_samples:
        try:
            hf_record = next(dataset_iterator)
            
            records_to_process = []
            
            # Prioritize chunking logic
            if args.chunk_column and args.chunk_column in hf_record:
                text_to_chunk = hf_record[args.chunk_column]
                chunks = chunk_text(text_to_chunk, by=args.chunk_by, size_min=args.chunk_size_min, size_max=args.chunk_size_max)
                for chunk in chunks:
                    modified_record = hf_record.copy()
                    modified_record[args.chunk_column] = chunk
                    records_to_process.append(modified_record)
            
            # Fallback to splitting logic
            elif args.split_column and args.split_column in hf_record:
                text_to_split = hf_record[args.split_column]
                chunks = [chunk.strip() for chunk in text_to_split.split(args.split_separator) if chunk.strip()]
                for chunk in chunks:
                    modified_record = hf_record.copy()
                    modified_record[args.split_column] = chunk
                    records_to_process.append(modified_record)
            
            # Default case: no splitting or chunking
            else:
                records_to_process.append(hf_record)

            # Process each record (or chunk)
            for record in records_to_process:
                if len(generated_data) >= args.num_samples: break
                
                final_user_prompt = user_template.render(record)
                model_response = call_api_with_retry(client, args.model, system_prompt, final_user_prompt, args.max_tokens, args.temperature)

                conversation_item = {"id": str(uuid.uuid4()), "conversations": [{"from": "human", "value": final_user_prompt}, {"from": "gpt", "value": model_response}]}
                generated_data.append(conversation_item)
                pbar.update(1)

                if len(generated_data) % args.save_interval == 0:
                    save_dataset(generated_data, args.output_file)
                    pbar.set_postfix_str(f"Saved {len(generated_data)} records")

        except StopIteration:
            print("\nReached the end of the Hugging Face dataset.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Saving progress and stopping generation.")
            break
    
    pbar.close()
    save_dataset(generated_data, args.output_file)

if __name__ == "__main__":
    main()