# PII_Scrubbing_LLM
This repo contains codes about PII scrubbing heuristics search before calling to LLM (local and external)

## NER for PII Scrubbing using Ollama API

## curl call example:

#  curl -X POST "http://localhost:5000/process" -H "Content-Type: application/json" -d '{"text": "John Doe lives in New York and his email is johndoe@example.com.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "John Doe lives in New York and his email is johndoe@example.com.",  "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "My IP address is 192.168.1.1, and my SSN is 123-45-6789.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "My credit card number is 4111-1111-1111-1111 and the password: mysecurepassword.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{ "text": "Alice from California has the email alice123@gmail.com, IP 10.0.0.1, and SSN 987-65-4321.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "The user handle @cooluser is from San Francisco, CA.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{ "text": "Here is my password: super_secret123 and IP address 172.16.254.1.",   "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

---
# Detect PII using spaCy NER
 Following PII are also possible
    - "NORP", "LOC", "FAC" 
    - Below is a brief description of each entity type commonly detected by spaCy’s en_core_web_sm model and related custom logic:
    - PERSON: Individual people, fictional or real.
    - GPE (Geo-Political Entity): Countries, cities, states, and similar geopolitical regions.
    - ORG (Organization): Companies, institutions, agencies, and similar groups.
    - DATE: Specific dates or periods, such as "January 10th, 2021" or "the 1990s".
    - TIME: Times of day, such as "2 p.m." or "morning".
    - MONEY: Monetary values, including units of currency, e.g. "$10", "€200".
    - EMAIL: Email addresses. (Not directly recognized by the base model, often handled via regex or custom rules.)
    - PHONE: Phone numbers. (Also often identified via custom regex rather than the base model.)
    - NORP: Nationalities, religious, and political groups (e.g., "American", "Christian", "Democrats").
    - LOC: Non-GPE locations, such as mountain ranges, bodies of water, or other geographical constructs.
    - FAC: Facilities, buildings, airports, highways, and other recognized structures.
---


### For openAI make sure the API is set before calling

# export OPENAI_API_KEY="your_api_key_here"

# export OPENAI_API_KEY="your_openai_key" curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "John Doe has an email johndoe@example.com", "model": "gpt-4"}'




### For Anthropic make sure the API is set before calling

# export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# export ANTHROPIC_API_KEY="your_anthropic_key" curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "Alice from California, email alice123@gmail.com", "model": "claude-2"}'

