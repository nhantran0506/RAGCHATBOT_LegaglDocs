
# Project Setup and Execution Guide

This guide will walk you through the process of setting up and running the project.

## Prerequisites

- Python 3.10.14
- Git
- Ollama with llama3.1 model

## Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/nhantran0506/RAGCHATBOT_LegaglDocs.git
   cd .\RAG\
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Ollama and the llama3.1 model:
   - Follow the instructions at [Ollama's official website](https://ollama.ai/) to install Ollama for your operating system.
   - After installation, run the following command to download the llama3.1 model:
     ```
     ollama run llama3.1
     ```

## Running the Project

1. Ensure you're in the project directory and your virtual environment is activated (if you created one).

2. Run the main Python script:
   ```
   python llama.py/gpt.py
   ```

## Troubleshooting

- Ensure that Ollama is running and the llama3.1 model is properly installed before executing the main script.
- If you face any Python-related errors, double-check that you're using Python 3.10.14.

## Additional Notes

- This project was developed and tested with Python 3.10.14. Using other versions may lead to compatibility issues.
- Make sure all environment variables are properly set if your project requires them.
