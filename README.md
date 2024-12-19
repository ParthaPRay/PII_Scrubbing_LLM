# PII Scrubbing Service with spaCy NER and Flexible LLM Integration (Local and Remote)

This FastAPI-based service accepts text input and a model name, scrubs Personally Identifiable Information (PII) using spaCy NER and regex patterns, then sends the sanitized input to a specified Large Language Model (LLM) endpoint. The results (including performance metrics, CPU usage, and PII detection details) are logged to both a local SQLite database and returned as a JSON response. 

NER using Spacy is the Python-based Natural Language Processing task that focuses on detecting and categorizing named entities.

Currently, the code supports legacy Text Completaions (https://docs.anthropic.com/en/api/complete) by Anthropic and chat completion (https://platform.openai.com/docs/api-reference/chat) by OpenAI. It supports text generation (https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) of Ollama.

## Key Features

1. **PII Scrubbing:**
   - Uses spaCy's `en_core_web_sm` model to detect named entities that could be considered sensitive (e.g., `PERSON`, `GPE`, `ORG`, `DATE`, `TIME`, `MONEY`, `NORP`, `LOC`, `FAC`).
   - Additional PII patterns like `IP_ADDRESS`, `SSN`, `CREDIT_CARD`, and `PASSWORD` are detected using custom regex patterns.
   - All detected PII are replaced with placeholder tags (e.g., `<PERSON>`, `<IP_ADDRESS>`).

2. **Flexible LLM Backend:**
   - **Local LLM (Ollama):** For model names that do **not** start with `"gpt-"` or `"claude-"`, the service assumes a locally running Ollama LLM endpoint at `http://0.0.0.0:11434/api/generate`.
   - **OpenAI ChatGPT:** For model names starting with `"gpt-"`, it calls the OpenAI ChatCompletion API. Requires `OPENAI_API_KEY` to be set as an environment variable.
   - **Anthropic Claude:** For model names starting with `"claude-"`, it calls the Anthropic API endpoint. Requires `ANTHROPIC_API_KEY` to be set as an environment variable.

3. **Performance and Metadata Logging:**
   - Logs details about execution (timestamps, CPU usage, network latency, token usage) and the chosen model into a SQLite database.
   - Information is returned as a JSON response including scrubbed input, detected entities, and LLM output.

4. **Local SQLite Database Logging:**
   - The code uses SQLAlchemy to store metadata of each request.
   - Columns include prompt, scrubbed PII, timestamps, durations, CPU usage, model name, and performance statistics.

## How It Works

1. **Receive Request:**  
   The `/process` endpoint accepts a JSON payload with:
   - `text`: The user-provided text containing potential PII.
   - `model`: The LLM model identifier (e.g., `"qwen2.5:0.5b-instruct-q4_K_M"`, `"gpt-4o"`, `gpt-4o-mini`, `"claude-2.1"`).

2. **Scrub PII:**
   - The code uses spaCy NER to find entities labeled as PII.  
   - Additional regex patterns detect things like IP addresses, SSNs, credit cards, and passwords.
   - Detected PII entities are replaced with placeholders (e.g., `<PERSON>`, `<IP_ADDRESS>`).
   - The scrubbed text and a list of detected entities are retained.

3. **LLM Call:**
   - Based on the `model` string, the service decides which API to call:
     - **Local/Ollama:** If no prefix (`gpt-` or `claude-`), calls the Ollama endpoint.
     - **OpenAI:** If the model name starts with `"gpt-"`, calls OpenAI's API. Requires `OPENAI_API_KEY`.
     - **Anthropic:** If the model name starts with `"claude-"`, calls Anthropic's API. Requires `ANTHROPIC_API_KEY`.
   - Measures CPU usage and network latency before and after the request.
   - Extracts response text and performance metrics (e.g., tokens processed).

4. **Logging and Response:**
   - All relevant data (prompt, scrubbed text, PII details, performance metrics, CPU usage, model name) are inserted into the local SQLite database.
   - The final JSON response returns:
     - `scrubbed_input`: The sanitized input.
     - `detected_entities`: A list of detected PII and their types.
     - `response`: The LLM's generated output.
     - `log`: A dictionary of metadata (durations, CPU usage, tokens/second, etc.).

## System Requirements

- Python 3.9+
- `pip install fastapi uvicorn spacy requests sqlalchemy` or `pip install -r requirements.txt`
- SpaCy English model: `python -m spacy download en_core_web_sm`
- Access to desired LLM endpoints (Ollama, OpenAI, Anthropic).  
- Appropriate API keys set as environment variables (if using OpenAI or Anthropic).

## Running the Application

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Set Environment Variables for Remote LLMs:**
   - For OpenAI:
     ```bash
     export OPENAI_API_KEY="your_openai_api_key"
     ```
   - For Anthropic:
     ```bash
     export ANTHROPIC_API_KEY="your_anthropic_api_key"
     ```

   If you're only using a local LLM (Ollama), you do not need these keys.

   Run below (serial no. 4) for running FastAPI server on the same terminal.

4. **Run the FastAPI Uvicron Server:**
   ```bash
   uvicorn __main__:app --host 0.0.0.0 --port 5000 --reload
   ```
   or

   ```bash
   python3 pii_scrub_ner_eng_hybrid.py
   ```

   This will start the FastAPI server on port 5000.

5. **Database Setup:**
   - On the first run, `Base.metadata.create_all(bind=engine)` creates the `requests` table in `ollama_log.db`.
   - If you change the schema, remove or rename `ollama_log.db` before restarting to recreate the schema.

## Example `curl` Calls

**Local LLM (Ollama):**
```bash
curl -X POST "http://127.0.0.1:5000/process" \
-H "Content-Type: application/json" \
-d '{"text": "Here is my password: super_secret123 and IP address 172.16.254.1.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'
```

**OpenAI (ChatGPT):**
```bash
export OPENAI_API_KEY="your_openai_api_key"
curl -X POST "http://127.0.0.1:5000/process" \
-H "Content-Type: application/json" \
-d '{"text": "John Doe has an email johndoe@example.com", "model": "gpt-4"}'
```

**Anthropic (Claude):**
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key"
curl -X POST "http://127.0.0.1:5000/process" \
-H "Content-Type: application/json" \
-d '{"text": "Alice from California, email alice123@gmail.com", "model": "claude-2"}'
```

## Entity Types and PII

Common entities recognized by spaCy and custom logic:

- **PERSON**: Individual human names (fictional or real).
- **GPE**: Geo-political entities like countries, states, and cities.
- **ORG**: Organizations, companies, agencies.
- **DATE**: Specific dates or date ranges.
- **TIME**: Times of day or durations within a day.
- **MONEY**: Monetary values and currencies.
- **NORP**: Nationalities, religious groups, political organizations.
- **LOC**: Non-GPE locations like mountains, lakes, etc.
- **FAC**: Facilities, buildings, airports.
- **EMAIL**, **PHONE**: Can be recognized via custom rules or regex.
- **IP_ADDRESS**, **SSN**, **CREDIT_CARD**, **PASSWORD**: Handled via regex patterns.

The service replaces these PII instances with placeholder tags (e.g., `<PERSON>`, `<IP_ADDRESS>`).

## Notes

- Make sure to run `uvicorn` and set API keys in the same environment and session.
- If you encounter `no such table: requests`, remove `ollama_log.db` and restart the server.
- Adjust the model name and LLM endpoint logic as needed for different providers.

---

This README should help you understand how to use and extend this PII scrubbing and LLM integration service.
