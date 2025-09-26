# generate.py

import os
import uuid
import json
import time
import argparse
import random
import asyncio
from pathlib import Path

import yaml
from jinja2 import Template
from openai import AsyncOpenAI, RateLimitError, APIError
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
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
    parser.add_argument("--concurrency", type=int, default=5, help="Number of asynchronous requests to make.")
    return parser.parse_args()

# --- Asynchronous API Interaction with Retry Logic ---
async def generate_conversation_async(client, model, system_prompt, user_prompt, max_tokens, temperature, semaphore):
    """
    An async worker that calls the API for a single prompt, with retries and concurrency control.
    """
    async with semaphore:
        max_retries = 5
        wait_time = 1
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                model_response = response.choices[0].message.content.strip()
                return {
                    "id": str(uuid.uuid4()),
                    "conversations": [
                        {"from": "human", "value": user_prompt},
                        {"from": "gpt", "value": model_response}
                    ]
                }
            except (RateLimitError, APIError) as e:
                if attempt == max_retries - 1:
                    print(f"Final attempt failed for a prompt. Error: {e}")
                    return None # Failed after all retries
                await asyncio.sleep(wait_time * (2 ** attempt)) # Exponential backoff
        return None # Should not be reached, but for safety

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
async def main():
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

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    print(f"Initialized AsyncOpenAI client for model: {args.model}")
    
    system_prompt, user_prompt_str = load_prompt_template(args.prompt_file)
    user_template = Template(user_prompt_str)
    print(f"Loaded prompt template from: {args.prompt_file}")
    
    # 1. Prepare all inputs first to know the total number of API calls
    print("Preparing generation inputs from the dataset...")
    generation_inputs = []
    hf_dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=True).skip(args.start_index)
    
    pbar_prepare = tqdm(total=args.num_samples, desc="Scanning dataset")
    for hf_record in hf_dataset:
        if len(generation_inputs) >= args.num_samples: break
        
        records_to_process = []
        if args.chunk_column and args.chunk_column in hf_record:
            chunks = chunk_text(hf_record[args.chunk_column], by=args.chunk_by, size_min=args.chunk_size_min, size_max=args.chunk_size_max)
            for chunk in chunks:
                records_to_process.append({**hf_record, args.chunk_column: chunk})
        elif args.split_column and args.split_column in hf_record:
            chunks = [c.strip() for c in hf_record[args.split_column].split(args.split_separator) if c.strip()]
            for chunk in chunks:
                records_to_process.append({**hf_record, args.split_column: chunk})
        else:
            records_to_process.append(hf_record)
            
        for record in records_to_process:
            if len(generation_inputs) < args.num_samples:
                final_user_prompt = user_template.render(record)
                generation_inputs.append(final_user_prompt)
                pbar_prepare.update(1)
    pbar_prepare.close()
    
    if not generation_inputs:
        print("No inputs were prepared. Exiting.")
        return

    print(f"Prepared {len(generation_inputs)} prompts. Starting concurrent generation...")
    
    # 2. Run generation tasks concurrently
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        generate_conversation_async(client, args.model, system_prompt, user_prompt, args.max_tokens, args.temperature, semaphore)
        for user_prompt in generation_inputs
    ]
    
    generated_data = []
    for future in tqdm.as_completed(tasks, desc="Generating Samples"):
        result = await future
        if result:
            generated_data.append(result)
            if len(generated_data) % args.save_interval == 0:
                save_dataset(generated_data, args.output_file)
    
    # Final save
    save_dataset(generated_data, args.output_file)

if __name__ == "__main__":
    asyncio.run(main())