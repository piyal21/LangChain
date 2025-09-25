# LangChain 

This repository contains code and utilities for working with the LangChain framework, including document loaders, text splitters, output parsers, chaining logic, and integration with Hugging Face models.

## Features
- **Document Loading**: Load and process text and PDF documents.
- **Text Splitting**: Multiple strategies for splitting text, including semantic and length-based methods.
- **Chaining**: Build simple, parallel, and conditional chains for LLM workflows.
- **Output Parsing**: Parse and structure LLM outputs using string, JSON, and Pydantic parsers.
- **Hugging Face Integration**: Use Hugging Face models securely with environment variables.
- **Tools**: Utility functions for LangChain workflows.

## Project Structure
```
Chaining/                # Chain logic (simple, parallel, conditional)
Document_Loader/         # Document loading utilities
LLM_Chatmodel/           # Chat model integration (Hugging Face)
Output Parsers/          # Output parsing utilities
Text Splitters/          # Text splitting strategies
Tools/                   # Additional tools
TypeDict/                # TypedDict and structured output examples
py_pdf_loader.py         # PDF loader example
```

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/piyal21/LangChain.git
   cd LangChain
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up your Hugging Face API token:**
   - Copy `.env.example` to `.env` and add your token:
     ```
     HUGGINGFACE_API_TOKEN=your_token_here
     ```

## Security
- **Never commit your real API tokens.**
- The `.env` file is in `.gitignore` and will not be uploaded.
- All code uses environment variables for sensitive credentials.

## Usage
- Explore the folders for examples and utilities.
- Modify and extend the chains, loaders, and parsers as needed for your LLM workflows.

## License
This project is for educational and research purposes. See `LICENSE` for more details.
