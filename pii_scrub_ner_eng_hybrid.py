# Partha Pratim Ray, parthapratimray1986@gmail.com, 19/12/2024

## NER for PII Scrubbing using Ollama API for Local and Remote LLM
 
## curl call example:

#  curl -X POST "http://localhost:5000/process" -H "Content-Type: application/json" -d '{"text": "John Doe lives in New York and his email is johndoe@example.com.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "John Doe lives in New York and his email is johndoe@example.com.",  "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "My IP address is 192.168.1.1, and my SSN is 123-45-6789.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "My credit card number is 4111-1111-1111-1111 and the password: mysecurepassword.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{ "text": "Alice from California has the email alice123@gmail.com, IP 10.0.0.1, and SSN 987-65-4321.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "The user handle @cooluser is from San Francisco, CA.", "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

# curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{ "text": "Here is my password: super_secret123 and IP address 172.16.254.1.",   "model": "qwen2.5:0.5b-instruct-q4_K_M"}'

################
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
################


### For openAI make sure the API is set before calling

# export OPENAI_API_KEY="your_api_key_here"

# export OPENAI_API_KEY="your_openai_key" curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "John Doe has an email johndoe@example.com", "model": "gpt-4"}'




### For Anthropic make sure the API is set before calling

# export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# export ANTHROPIC_API_KEY="your_anthropic_key" curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"text": "Alice from California, email alice123@gmail.com", "model": "claude-2"}'





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
import os

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI()

# Load spaCy's pre-trained NER model
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("The spaCy model 'en_core_web_sm' is not installed. Install it using 'python -m spacy download en_core_web_sm'.")

nlp = load_spacy_model()

# PII patterns via regex for custom detection
PII_PATTERNS = {
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "PASSWORD": re.compile(r"(?i)(password:\s*\S+)")
}

# Additional spaCy entity labels to treat as PII if desired
PII_ENTITY_LABELS = [
    "PERSON", "GPE", "ORG", "DATE", "TIME", "MONEY", 
    "NORP", "LOC", "FAC"
]

class RequestData(BaseModel):
    text: str
    model: str

DATABASE_URL = "sqlite:///./ollama_log.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class LogEntry(Base):
    __tablename__ = "requests"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    prompt = Column(Text)
    pii_scrub = Column(Text)
    total_duration = Column(Float)
    load_duration = Column(Float)
    prompt_eval_count = Column(Integer)
    prompt_eval_duration = Column(Float)
    eval_count = Column(Integer)
    eval_duration = Column(Float)
    response = Column(Text)
    tokens_per_second = Column(Float)
    pii_scrub_duration = Column(Integer)
    cpu_usage_scrub = Column(Float)
    cpu_usage_llm = Column(Float)
    network_latency = Column(Integer)
    model = Column(String)   # New column to store the model name

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def scrub_pii(text):
    start_time = time.perf_counter_ns()
    initial_cpu = psutil.cpu_percent(interval=None)

    doc = nlp(text)
    scrubbed_text = text
    entities = []

    # Detect PII using spaCy NER
    for ent in doc.ents:
        if ent.label_ in PII_ENTITY_LABELS:
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

def call_llm_endpoint(url, headers, payload):
    cpu_times_before = psutil.cpu_times()
    start_time = time.perf_counter_ns()

    response = requests.post(url, headers=headers, json=payload)

    end_time = time.perf_counter_ns()
    cpu_times_after = psutil.cpu_times()

    total_before = sum(cpu_times_before)
    total_after = sum(cpu_times_after)
    total_diff = total_after - total_before
    idle_diff = cpu_times_after.idle - cpu_times_before.idle
    if total_diff > 0:
        cpu_usage_llm = (1.0 - (idle_diff / total_diff)) * 100.0
    else:
        cpu_usage_llm = 0.0
    cpu_usage_llm = round(cpu_usage_llm, 1)

    network_latency = end_time - start_time

    if response.status_code == 200:
        resp_json = response.json()
        return cpu_usage_llm, network_latency, resp_json
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

def send_to_llm(prompt, model):
    # Determine which endpoint to call based on the model name:
    if model.startswith("gpt-"):
        # OpenAI ChatCompletion
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        cpu_usage_llm, network_latency, resp_json = call_llm_endpoint(url, headers, payload)
        response_text = resp_json["choices"][0]["message"]["content"] if resp_json.get("choices") else ""
        total_duration = resp_json.get("usage", {}).get("total_tokens", 0)
        load_duration = 0
        prompt_eval_count = 0
        prompt_eval_duration = 0
        eval_count = resp_json.get("usage", {}).get("total_tokens", 0)
        eval_duration = 1  # Avoid division by zero
        tokens_per_second = (eval_count / eval_duration) * 1e9

    elif model.startswith("claude-"):
        # Anthropic Claude
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            raise HTTPException(status_code=500, detail="Anthropic API key not configured.")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01"
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}]
        }

        cpu_usage_llm, network_latency, resp_json = call_llm_endpoint(url, headers, payload)
        response_text = resp_json["choices"][0]["message"]["content"] if resp_json.get("choices") else ""
        # Anthropic doesn't provide the same usage info:
        total_duration = 0
        load_duration = 0
        prompt_eval_count = 0
        prompt_eval_duration = 0
        eval_count = 0
        eval_duration = 0
        tokens_per_second = 0

    else:
        # Local LLM (Ollama)
        url = "http://0.0.0.0:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        cpu_usage_llm, network_latency, resp_json = call_llm_endpoint(url, headers, payload)
        response_text = resp_json.get("response", "")
        total_duration = resp_json.get("total_duration", 0)
        load_duration = resp_json.get("load_duration", 0)
        prompt_eval_count = resp_json.get("prompt_eval_count", 0)
        prompt_eval_duration = resp_json.get("prompt_eval_duration", 0)
        eval_count = resp_json.get("eval_count", 0)
        eval_duration = resp_json.get("eval_duration", 1)  # Avoid division by zero
        tokens_per_second = (eval_count / eval_duration) * 1e9

    return {
        "total_duration": total_duration,
        "load_duration": load_duration,
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
        "response": response_text,
        "tokens_per_second": tokens_per_second
    }, network_latency, cpu_usage_llm

def log_to_db(data):
    db = SessionLocal()
    entry = LogEntry(
        timestamp=datetime.now(),
        prompt=data["prompt"],
        pii_scrub=data["pii_scrub"],
        total_duration=data["total_duration"],
        load_duration=data["load_duration"],
        prompt_eval_count=data["prompt_eval_count"],
        prompt_eval_duration=data["prompt_eval_duration"],
        eval_count=data["eval_count"],
        eval_duration=data["eval_duration"],
        response=data["response"],
        tokens_per_second=data["tokens_per_second"],
        pii_scrub_duration=data["pii_scrub_duration"],
        cpu_usage_scrub=data["cpu_usage_scrub"],
        cpu_usage_llm=data["cpu_usage_llm"],
        network_latency=data["network_latency"],
        model=data["model"]    # Storing the model name
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    db.close()

@app.post("/process")
async def process_request(request_data: RequestData):
    try:
        # Scrub PII from the user input
        scrubbed_input, detected_entities, pii_scrub_duration, cpu_usage_scrub = scrub_pii(request_data.text)

        # Send the scrubbed prompt to chosen LLM
        response_data, network_latency, cpu_usage_llm = send_to_llm(scrubbed_input, request_data.model)

        total_duration = response_data.get("total_duration", 0)
        load_duration = response_data.get("load_duration", 0)
        prompt_eval_count = response_data.get("prompt_eval_count", 0)
        prompt_eval_duration = response_data.get("prompt_eval_duration", 0)
        eval_count = response_data.get("eval_count", 0)
        eval_duration = response_data.get("eval_duration", 1)
        response_text = response_data.get("response", "")
        tokens_per_second = response_data.get("tokens_per_second", 0.0)

        # Prepare data for logging
        pii_scrub_info = [f"{ent['type']}: {ent['entity']}" for ent in detected_entities]

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "prompt": request_data.text,
            "model": request_data.model,  # Include model
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

        # Log to database
        log_to_db(log_data)

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

