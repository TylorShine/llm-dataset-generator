# LLM Dataset Generator

This is a flexible, command-line tool for generating datasets using an OpenAI-compatible Large Language Model (LLM). It leverages Hugging Face datasets as a source of input data, processes it through customizable Jinja2 prompt templates, and generates responses from an LLM to create new datasets for a variety of tasks.

## Features

*   **Connect to any OpenAI-compatible API**: Configure the base URL and API key to use various LLM providers.
*   **Leverage Hugging Face Datasets**: Use any dataset from the Hugging Face Hub as a source for prompt generation.
*   **Customizable Prompts**: Use Jinja2 templates to easily format data from the source dataset into prompts for the LLM.
*   **Text Chunking and Splitting**: Automatically split text from a dataset column into smaller chunks by paragraph, sentence, or line, allowing for more granular data generation.
*   **Resilient and Efficient**:
    *   Includes automatic retries with exponential backoff for API calls to handle rate limits and transient errors.
    *   Streams datasets to minimize memory usage.
    *   Saves progress periodically to prevent data loss during long generation runs.
*   **Easy to Use**: A straightforward command-line interface.

## Prerequisites

*   Python 3.8+
*   pip

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TylorShine/llm-dataset-generator.git
    cd llm-dataset-generator
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file** by copying the example file:
    ```bash
    cp .env.example .env
    ```

2.  **Edit the `.env` file** to add your API key and the base URL for your LLM provider:
    ```
    OPENAI_API_KEY="sk-your_api_key_here"
    OPENAI_BASE_URL="https://api.openai.com/v1"
    ```
    Change `OPENAI_BASE_URL` to the appropriate endpoint if you are using a different service.

## Usage

The main script is `generate.py`. You can run it from the command line with various arguments to control the dataset generation process.

### Basic Example: Question Answering

This example uses the `squad` dataset and the `qa_template.yaml` to generate answers to questions.

```bash
python generate.py \
    --prompt-file prompts/qa_template.yaml \
    --output-file output/squad_qa_dataset.json \
    --dataset-name squad \
    --num-samples 50 \
    --model gpt-4o
```

### Command-Line Arguments

*   `--prompt-file`: (Required) Path to the YAML prompt template file.
*   `--output-file`: (Required) Path to save the generated JSON dataset.
*   `--dataset-name`: Name of the Hugging Face dataset to use (default: `squad`).
*   `--dataset-split`: Dataset split to use (default: `train`).
*   `--num-samples`: Number of samples to generate (default: 10).
*   `--start-index`: Index of the dataset to start from (default: 0).
*   `--chunk-column`: Name of a column to chunk into smaller pieces.
*   `--chunk-by`: The unit to chunk by: `sentences`, `words`, `paragraphs`, `lines` (default: `lines`).
*   `--chunk-size-min`/`--chunk-size-max`: The random range of units per chunk.
*   `--model`: Name of the model to use (default: `gpt-3.5-turbo`).
*   `--max-tokens`: Maximum number of tokens to generate (default: 512).
*   `--temperature`: Sampling temperature for generation (default: 0.7).
*   `--save-interval`: How often to save the intermediate dataset (default: 10).

## How to Create Prompt Templates

Prompt templates are YAML files that define the system and user prompts for the LLM.

1.  Create a new `.yaml` file in the `prompts/` directory.
2.  Define a `system_prompt` (optional) and a `user_prompt_template` (required).
3.  Use Jinja2 syntax (`{{ column_name }}`) to insert data from the source dataset into your `user_prompt_template`.

See [`prompts/qa_template.yaml`](./prompts/qa_template.yaml) for examples.