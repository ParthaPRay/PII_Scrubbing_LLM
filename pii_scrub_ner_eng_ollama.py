## NER for PII Scrubing using Ollama API

## curl call example:

#  curl -X POST "http://localhost:5000/process" -H "Content-Type: application/json" -d '{"text": "John Doe lives in New York and his email is johndoe@example.com.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "John Doe lives in New York and his email is johndoe@example.com.",  "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "My IP address is 192.168.1.1, and my SSN is 123-45-6789.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "My credit card number is 4111-1111-1111-1111 and the password: mysecurepassword.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{ "text": "Alice from California has the email alice123@gmail.com, IP 10.0.0.1, and SSN 987-65-4321.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "The user handle @cooluser is from San Francisco, CA.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{ "text": "Here is my password: super_secret123 and IP address 172.16.254.1.",   "model": "qwen2.5:0.5b-instruct-q4_K_M"}'


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import psutil
import csv
from datetime import datetime
import spacy
import requests
import uvicorn
import re

app = FastAPI()

# Load spaCy's pre-trained NER model
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("The spaCy model 'en_core_web_sm' is not installed. Install it using 'python -m spacy download en_core_web_sm'.")

nlp = load_spacy_model()

# Regular expressions for additional PII patterns
PII_PATTERNS = {
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "PASSWORD": re.compile(r"(?i)(password:\s*\S+)")
}

class RequestData(BaseModel):
    text: str
    model: str

def scrub_pii(text):
    start_time = time.perf_counter_ns()
    initial_cpu = psutil.cpu_percent(interval=None)

    doc = nlp(text)
    scrubbed_text = text
    entities = []

    # Detect PII using spaCy NER
    # Following PII are also possible
    # "NORP", "LOC", "FAC" 
    # Below is a brief description of each entity type commonly detected by spaCy’s en_core_web_sm model and related custom logic:
    # PERSON: Individual people, fictional or real.
    # GPE (Geo-Political Entity): Countries, cities, states, and similar geopolitical regions.
    # ORG (Organization): Companies, institutions, agencies, and similar groups.
    # DATE: Specific dates or periods, such as "January 10th, 2021" or "the 1990s".
    # TIME: Times of day, such as "2 p.m." or "morning".
    # MONEY: Monetary values, including units of currency, e.g. "$10", "€200".
    # EMAIL: Email addresses. (Not directly recognized by the base model, often handled via regex or custom rules.)
    # PHONE: Phone numbers. (Also often identified via custom regex rather than the base model.)
    # NORP: Nationalities, religious, and political groups (e.g., "American", "Christian", "Democrats").
    # LOC: Non-GPE locations, such as mountain ranges, bodies of water, or other geographical constructs.
    # FAC: Facilities, buildings, airports, highways, and other recognized structures.

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG", "DATE", "TIME", "MONEY", "EMAIL", "PHONE"]:
            scrubbed_text = scrubbed_text.replace(ent.text, f"<{ent.label_}>")
            entities.append({"entity": ent.text, "type": ent.label_})

    # Detect additional PII using regex
    for pii_type, pattern in PII_PATTERNS.items():
        matches = pattern.findall(text)
        for match in matches:
            scrubbed_text = scrubbed_text.replace(match, f"<{pii_type}>")
            entities.append({"entity": match, "type": pii_type})

    final_cpu = psutil.cpu_percent(interval=None)
    end_time = time.perf_counter_ns()

    pii_scrub_duration = end_time - start_time
    cpu_usage_scrub = final_cpu - initial_cpu

    return scrubbed_text, entities, pii_scrub_duration, cpu_usage_scrub

def send_to_ollama(prompt, model):
    url = "http://0.0.0.0:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"model": model, "prompt": prompt, "stream": False}

    # Record CPU times before request
    cpu_times_before = psutil.cpu_times()
    start_time = time.perf_counter_ns()

    response = requests.post(url, headers=headers, json=payload)

    end_time = time.perf_counter_ns()
    cpu_times_after = psutil.cpu_times()

    # Calculate interval in seconds
    interval = (end_time - start_time) / 1e9

    # Calculate total CPU time differences
    # cpu_times() returns fields like user, system, idle, etc.
    total_before = sum(cpu_times_before)
    total_after = sum(cpu_times_after)

    total_diff = total_after - total_before
    idle_diff = cpu_times_after.idle - cpu_times_before.idle

    # Compute the percentage of busy time
    # If there's no total_diff (very unlikely), default to 0
    if total_diff > 0:
        cpu_usage_llm = (1.0 - (idle_diff / total_diff)) * 100.0
    else:
        cpu_usage_llm = 0.0
        
    cpu_usage_llm = round(cpu_usage_llm, 1) # Rounding to 1 decimal point
    
    network_latency = end_time - start_time

    if response.status_code == 200:
        return response.json(), network_latency, cpu_usage_llm
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)



def log_to_csv(data, filename="ollama_log.csv"):
    file_exists = False
    try:
        with open(filename, "r", encoding="utf-8") as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "timestamp", "prompt", "pii_scrub", "total_duration", "load_duration", "prompt_eval_count",
            "prompt_eval_duration", "eval_count", "eval_duration", "response", "tokens_per_second",
            "pii_scrub_duration", "cpu_usage_scrub", "cpu_usage_llm", "network_latency"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

@app.post("/process")
async def process_request(request_data: RequestData):
    try:
        # Scrub PII from the user input
        scrubbed_input, detected_entities, pii_scrub_duration, cpu_usage_scrub = scrub_pii(request_data.text)

        # Send the scrubbed prompt to Ollama API
        response_data, network_latency, cpu_usage_llm = send_to_ollama(scrubbed_input, request_data.model)

        # Extract required details from the API response
        total_duration = response_data.get("total_duration", 0)
        load_duration = response_data.get("load_duration", 0)
        prompt_eval_count = response_data.get("prompt_eval_count", 0)
        prompt_eval_duration = response_data.get("prompt_eval_duration", 0)
        eval_count = response_data.get("eval_count", 0)
        eval_duration = response_data.get("eval_duration", 1)  # Avoid division by zero
        response_text = response_data.get("response", "")

        tokens_per_second = (eval_count / eval_duration) * 1e9

        # Prepare data for logging
        pii_scrub_info = [f"{ent['type']}: {ent['entity']}" for ent in detected_entities]

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": request_data.text,
            "pii_scrub": "; ".join(pii_scrub_info),
            "total_duration": total_duration,
            "load_duration": load_duration,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration": prompt_eval_duration,
            "eval_count": eval_count,
            "eval_duration": eval_duration,
            "response": response_text,
            "tokens_per_second": tokens_per_second,
            "pii_scrub_duration": pii_scrub_duration,
            "cpu_usage_scrub": cpu_usage_scrub,
            "cpu_usage_llm": cpu_usage_llm,
            "network_latency": network_latency
        }

        # Log to CSV
        log_to_csv(log_data)

        return {
            "scrubbed_input": scrubbed_input,
            "detected_entities": detected_entities,
            "response": response_text,
            "log": log_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", port=5000, reload=True)

