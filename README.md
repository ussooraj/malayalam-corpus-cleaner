# malayalam-corpus-cleaner

A toolkit for generating high-quality text corpora from uncleaned base datasets using regex-based cleaning and LLM semantic scoring.

## Key Features

*   **Hybrid Cleaning Strategy**
    Combines fast, rule-based methods (regex cleaning, markup removal) with powerful, deep semantic analysis using a Language Model.

*   **Configuration-Driven**
    The entire pipeline is controlled by a single [config.yaml](./config.yaml) file. Adjust models, paths, and filtering strategies without touching the Python code.

*   **Dual LLM Provider Support**
    -   **Local Mode**: Run GGUF-quantized models (e.g., Gemma, Llama) on your own machine for privacy, offline use, and cost-effectiveness.
    -   **API Mode**: Utilize powerful cloud models like Google Gemini for state-of-the-art speed and quality.

*   **Semantic Chunking**
    Long documents are split along natural paragraph and sentence boundaries, preserving context for more accurate LLM scoring.

*   **Multi-Format Ingestion**
    Processes text from `.txt`, `.md`, and `.docx` files, automatically parsing both structured and unstructured content.

*   **Detailed Logging and Outputs**
    Generates clean and discarded `.jsonl` files for easy inspection and provides detailed logs in `processing.log` for monitoring the entire process.

## Quick Start

1. Install dependencies and activate environment
2. Place `.txt` or `.md` or `.docx` files in [data/raw/](./data/raw/)
3. Run `python main.py`

See [Installation](#installation) for detailed setup.

## Pipeline Workflow

The toolkit processes documents through a sequential, multi-stage pipeline designed for efficiency and quality.

1.  **Ingestion**
    Documents are loaded from the `raw_data_dir` in the [config.yaml](./config.yaml) file.

2.  **Initial Cleaning**
    Basic cleaning functions are applied to all documents, such as removing wiki/HTML markup and normalizing whitespace.

3.  **Pre-filtering**
    Documents are filtered based on fast, rule-based checks defined in the configuration (e.g., minimum word count, Malayalam character ratio).

4.  **LLM Semantic Scoring**
    If enabled, the documents that passed pre-filtering are scored by a Language Model for semantic coherence, fluency, and logical consistency.

5.  **Final Filtering and Deduplication**
    -   Documents are filtered based on the `score_threshold` from the LLM.
    -   Duplicate documents are identified using a SHA256 hash and discarded.

6.  **Output Generation**
    The final, high-quality documents are saved to a `.jsonl` file, while all discarded documents are logged in a separate file with the reason for exclusion.

## Project Structure

```
malayalam-corpus-cleaner/
├── main.py                      # Entry point
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
├── .env                         # API keys (create this)
├── text_cleaner/                # Core modules
├── data/
│   ├── raw/                     # Input files (create this)
│   └── processed/               # Output files (auto-created)
└── processing.log               # Execution logs
```

## Prerequisites

- Python 3.11
- Conda package manager
- CUDA-compatible GPU (8GB VRAM recommended for local LLM inference)
- Google Gemini API key (if using API-based scoring)

## Installation

### Basic Installation (No local LLM support)

This lightweight setup uses Google Gemini API for scoring and doesn't require GPU or heavy dependencies.

1. #### Clone Repository

    ```bash
    git clone https://github.com/ussooraj/malayalam-corpus-cleaner.git
    cd malayalam-corpus-cleaner
    ```

2. #### Create Conda Environment

    ```bash
    conda create -n corpus-toolkit python=3.11 -y
    conda activate corpus-toolkit
    ```

3. #### Install Basic Dependencies

    ```bash
    pip install beautifulsoup4 python-docx Markdown PyYAML python-dotenv tqdm google-generativeai indic-nlp-library transformers huggingface-hub
    ```

4. #### Setup Environment

    ```bash
    mkdir -p data/raw data/processed
    echo "GOOGLE_API_KEY=your_api_key_here" > .env
    ```

    Get your API key from [Google AI Studio](https://aistudio.google.com/apikey)

5. #### Configure for API Mode

    Edit `config.yaml`:
    ```yaml
    llm_scorer:
    enabled: true
    provider: "api"
    ```


### Full Installation with Local Model Support

This setup enables local GGUF model inference on GPU for complete offline processing.

1. #### Clone Repository

    ```bash
    git clone https://github.com/ussooraj/malayalam-corpus-cleaner.git
    cd malayalam-corpus-cleaner
    ```

2. #### Create Conda Environment

    ```bash
    conda create -n corpus-toolkit python=3.11 -y
    conda activate corpus-toolkit
    ```

3. #### Install CUDA and PyTorch

    ```bash
    conda install nvidia/label/cuda-12.4.1::cuda -y
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    ```

4. #### Install Build Tools

    ```bash
    conda install -c conda-forge gxx_linux-64 cmake libidn2 ninja pybind11 -y
    ```

5. #### Install llama-cpp-python

    ```bash
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    ```

6. #### Install All Dependencies

    ```bash
    pip install -r requirements.txt
    ```

7. #### Setup Environment

    ```bash
    mkdir -p data/raw data/processed
    ```

8. #### Configure for Local Mode

    Edit `config.yaml`:
    ```yaml
    llm_scorer:
    enabled: true
    provider: "local"
    ```

## Configuration

Edit [config.yaml](./config.yaml) to customize the pipeline:

```yaml
paths:
  raw_data_dir: data/raw
  processed_data_dir: data/processed
  output_filename: cleaned_malayalam_corpus.jsonl

filters:
  malayalam_ratio_threshold: 0.8
  min_word_count: 5

llm_scorer:
  enabled: true
  provider: "api"  # Options: 'api' or 'local'
  score_threshold: 5
```

See the full configuration file for API and local model settings.

## Usage

### Basic Usage

Place your text files in `data/raw/` and run:

```bash
python main.py
```

### Custom Configuration

```bash
python main.py --config custom_config.yaml
```

## Input Formats

### Supported File Types

- Plain text (`.txt`)
- Markdown (`.md`)
- Word documents (`.docx`)

## Output Format

The toolkit generates JSONL files with cleaned text and quality scores:

```json
{
  "id": "text_01.txt",
  "url": "path/to/text_01.txt",
  "title": "Document Title",
  "text": "ഇവിടെ വൃത്തിയാക്കിയ മലയാളം വാചകം ദൃശ്യമാകും...",
  "llm_score": 9,
  "llm_reason": "The text is fluent, coherent, and well-structured."
}
```

## LLM Scoring

### Score Range

- **8-10**: High quality, fluent, coherent
- **6-7**: Acceptable quality, minor issues
- **4-5**: Low quality, grammatical errors
- **1-3**: Garbage quality, broken or nonsensical

### Provider Selection

- **API (`provider: "api"`)**: Use Google Gemini API (requires API key)
- **Local (`provider: "local"`)**: Use local GGUF models (requires GPU)

## Troubleshooting


- ### API Key Not Set

    If you see an error like `GOOGLE_API_KEY not found in environment variables`:

    - Check for `.env` file and verify the API key is properly set

        ```bash
        cat .env
        ```

        Should display:
        ```
        GOOGLE_API_KEY=your_actual_api_key_here
        ```

    - If missing, create the file

        ```bash
        echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
        ```

    - Get your API key and paste it in the `.env` file

        Visit [Google AI Studio](https://aistudio.google.com/apikey) to generate a new API key.


- ### VRAM Issues

    Reduce `n_ctx` in [config.yaml](./config.yaml):

    ```yaml
    local_config:
    n_ctx: 8192  # Lower value for limited VRAM
    ```

- ### Empty Output

    Check `processing.log` for errors and verify:
    - Input files exist in `data/raw/`
    - Configuration thresholds are not too strict
    - API key is set correctly (if using API)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.