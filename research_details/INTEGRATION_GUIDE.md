# Integration & Adaptability Guide

Complete guide for integrating TextDiff into any project or system.

---

## 🔧 Core Integration Patterns

### 1. Web Applications

**Flask Example:**
```python
from flask import Flask, request, jsonify
import textdiff

app = Flask(__name__)
defense = textdiff.ControlDD()

@app.post("/api/chat")
def chat():
    user_message = request.json['message']
    result = defense.analyze_and_respond(user_message)
    
    if result['send_to_llm']:
        # Safe - process with your LLM
        response = your_llm.generate(result['llm_prompt'])
        return jsonify({"response": response})
    elif result['status'] == 'needs_clarification':
        return jsonify({
            "needs_confirmation": True,
            "message": result['message_to_user'],
            "suggested": result['cleaned_prompt']
        })
    else:
        return jsonify({"error": result['message_to_user']})

@app.post("/api/confirm")
def confirm_choice():
    data = request.json
    verification = defense.verify_and_proceed(
        data['choice'], data['original'], data['cleaned']
    )
    
    if verification['send_to_llm']:
        response = your_llm.generate(verification['prompt_to_use'])
        return jsonify({"response": response})
    else:
        return jsonify({"error": verification['message_to_user']})
```

**FastAPI Example:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import textdiff

app = FastAPI()
defense = textdiff.ControlDD()

class ChatRequest(BaseModel):
    message: str

class ConfirmRequest(BaseModel):
    choice: str
    original: str
    cleaned: str

@app.post("/chat")
async def chat(request: ChatRequest):
    result = defense.analyze_and_respond(request.message)
    return result

@app.post("/confirm")
async def confirm(request: ConfirmRequest):
    verification = defense.verify_and_proceed(
        request.choice, request.original, request.cleaned
    )
    return verification
```

---

### 2. Command-Line Tools

**Interactive CLI:**
```python
import textdiff

def main():
    defense = textdiff.ControlDD()
    print("TextDiff Safety Layer Active")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        
        result = defense.analyze_and_respond(user_input)
        
        if result['send_to_llm']:
            response = your_llm.generate(result['llm_prompt'])
            print(f"AI: {response}")
        elif result['status'] == 'needs_clarification':
            print(f"System: {result['message_to_user']}")
            choice = input("Use cleaned version? (yes/no): ")
            
            verification = defense.verify_and_proceed(
                'cleaned' if choice.lower() == 'yes' else 'original',
                result['original_prompt'],
                result['cleaned_prompt']
            )
            
            if verification['send_to_llm']:
                response = your_llm.generate(verification['prompt_to_use'])
                print(f"AI: {response}")
            else:
                print(f"System: {verification['message_to_user']}")
        else:
            print(f"System: {result['message_to_user']}")

if __name__ == "__main__":
    main()
```

---

### 3. Batch Processing

**File Processing:**
```python
import textdiff
import json

defense = textdiff.ControlDD()

def process_prompts_file(input_file, output_file):
    with open(input_file, 'r') as f:
        prompts = [line.strip() for line in f]
    
    results = []
    for prompt in prompts:
        clean = defense.get_clean_text_for_llm(prompt)
        results.append({
            "original": prompt,
            "cleaned": clean,
            "risk": defense.analyze_risk(prompt)
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

# Usage
process_prompts_file('user_prompts.txt', 'cleaned_prompts.json')
```

**Parallel Processing:**
```python
from concurrent.futures import ThreadPoolExecutor
import textdiff

defense = textdiff.ControlDD()

def clean_prompt(prompt):
    return defense.get_clean_text_for_llm(prompt)

# Process 1000 prompts in parallel
prompts = load_prompts()
with ThreadPoolExecutor(max_workers=10) as executor:
    cleaned = list(executor.map(clean_prompt, prompts))
```

---

### 4. API Middleware

**Generic Middleware Pattern:**
```python
import textdiff

class SafetyMiddleware:
    def __init__(self):
        self.defense = textdiff.ControlDD()
    
    def process(self, user_input):
        """Process user input before sending to LLM."""
        result = self.defense.analyze_and_respond(user_input)
        return {
            "safe": result['send_to_llm'],
            "prompt": result.get('llm_prompt'),
            "message": result.get('message_to_user'),
            "status": result['status']
        }
    
    def verify_user_choice(self, choice, original, cleaned):
        """Verify user's confirmation."""
        verification = self.defense.verify_and_proceed(choice, original, cleaned)
        return {
            "approved": verification['send_to_llm'],
            "prompt": verification.get('prompt_to_use'),
            "message": verification.get('message_to_user')
        }

# Usage
middleware = SafetyMiddleware()
result = middleware.process(user_input)
if not result['safe']:
    # Handle user confirmation
    confirmation = middleware.verify_user_choice('cleaned', original, cleaned)
```

---

### 5. Custom Threshold Configuration

**Educational Platform (More Permissive):**
```python
import textdiff

defense = textdiff.ControlDD()
defense.adaptive_thresholds.set_mode('permissive')
defense.adaptive_thresholds.set_context('educational')

# Now safer for educational queries
result = defense.analyze_and_respond(student_question)
```

**Safety-Critical Application (Stricter):**
```python
defense = textdiff.ControlDD()
defense.adaptive_thresholds.set_mode('conservative')
defense.adaptive_thresholds.set_context('safety_critical')

# Much stricter filtering
result = defense.analyze_and_respond(user_input)
```

---

## 🤖 LLM Integration Examples

### OpenAI GPT
```python
import textdiff
import openai

defense = textdiff.ControlDD()

def safe_chat(user_message):
    clean = defense.get_clean_text_for_llm(user_message)
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": clean}]
    )
    
    return response.choices[0].message.content
```

### Anthropic Claude
```python
import textdiff
import anthropic

defense = textdiff.ControlDD()

def safe_chat(user_message):
    clean = defense.get_clean_text_for_llm(user_message)
    
    client = anthropic.Client(api_key="your-key")
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": clean}]
    )
    
    return response.content[0].text
```

### HuggingFace Transformers
```python
import textdiff
from transformers import pipeline

defense = textdiff.ControlDD()
llm = pipeline("text-generation", model="gpt2")

def safe_generate(user_input):
    clean = defense.get_clean_text_for_llm(user_input)
    result = llm(clean, max_length=100)
    return result[0]['generated_text']
```

### Local LLMs (Ollama)
```python
import textdiff
import requests

defense = textdiff.ControlDD()

def safe_ollama(user_input):
    clean = defense.get_clean_text_for_llm(user_input)
    
    response = requests.post('http://localhost:11434/api/generate',
        json={"model": "llama2", "prompt": clean})
    
    return response.json()['response']
```

### LM Studio
```python
import textdiff
from openai import OpenAI

defense = textdiff.ControlDD()

# LM Studio provides OpenAI-compatible API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

def safe_lm_studio(user_input):
    clean = defense.get_clean_text_for_llm(user_input)
    
    completion = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": clean}]
    )
    
    return completion.choices[0].message.content
```

---

## 🌐 Platform-Specific Integrations

### AWS Lambda (Serverless)
```python
import textdiff

# Initialize outside handler for reuse
defense = textdiff.ControlDD()

def lambda_handler(event, context):
    user_input = event['body']['message']
    clean = defense.get_clean_text_for_llm(user_input)
    
    # Call your LLM
    response = invoke_llm(clean)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'response': response})
    }
```

### Google Cloud Functions
```python
import textdiff
import functions_framework

defense = textdiff.ControlDD()

@functions_framework.http
def safe_chat(request):
    data = request.get_json()
    clean = defense.get_clean_text_for_llm(data['message'])
    
    response = your_llm.generate(clean)
    return {"response": response}
```

### Azure Functions
```python
import textdiff
import azure.functions as func

defense = textdiff.ControlDD()

def main(req: func.HttpRequest) -> func.HttpResponse:
    user_input = req.get_json()['message']
    clean = defense.get_clean_text_for_llm(user_input)
    
    response = your_llm.generate(clean)
    return func.HttpResponse(json.dumps({"response": response}))
```

---

## 📱 Mobile Backend Integration

### REST API for Mobile Apps
```python
from flask import Flask
import textdiff

app = Flask(__name__)
defense = textdiff.ControlDD()

@app.post("/api/v1/safe-chat")
def mobile_chat():
    data = request.json
    
    # Simple mode for mobile
    clean = defense.get_clean_text_for_llm(data['message'])
    response = llm.generate(clean)
    
    return jsonify({
        "message": response,
        "safe": True
    })
```

---

## 🔌 Message Queue Integration

### RabbitMQ Consumer
```python
import textdiff
import pika

defense = textdiff.ControlDD()

def callback(ch, method, properties, body):
    user_input = body.decode()
    clean = defense.get_clean_text_for_llm(user_input)
    
    # Process with LLM
    response = your_llm.generate(clean)
    
    # Publish response
    ch.basic_publish(exchange='',
                     routing_key='llm_responses',
                     body=response)

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.basic_consume(queue='user_prompts', on_message_callback=callback)
channel.start_consuming()
```

### Apache Kafka
```python
import textdiff
from kafka import KafkaConsumer, KafkaProducer

defense = textdiff.ControlDD()
consumer = KafkaConsumer('user-prompts')
producer = KafkaProducer()

for message in consumer:
    user_input = message.value.decode()
    clean = defense.get_clean_text_for_llm(user_input)
    
    response = your_llm.generate(clean)
    producer.send('llm-responses', response.encode())
```

---

## 💻 Desktop Application Integration

### Electron/Web-based Desktop App
```python
# Python backend for Electron
import textdiff
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
defense = textdiff.ControlDD()

@app.post("/clean")
def clean_text():
    user_input = request.json['text']
    clean = defense.get_clean_text_for_llm(user_input)
    return jsonify({"cleaned": clean})
```

### PyQt Application
```python
from PyQt5.QtWidgets import QApplication, QMainWindow
import textdiff

class SafeChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.defense = textdiff.ControlDD()
    
    def on_send_message(self, user_input):
        clean = self.defense.get_clean_text_for_llm(user_input)
        response = self.llm.generate(clean)
        self.display_message(response)
```

---

## 📊 Analytics & Monitoring

### Track Safety Metrics
```python
import textdiff

class MonitoredDefense:
    def __init__(self):
        self.defense = textdiff.ControlDD()
        self.stats = {
            "total": 0,
            "approved": 0,
            "suggested": 0,
            "rejected": 0,
            "risks": []
        }
    
    def process(self, user_input):
        self.stats["total"] += 1
        result = self.defense.analyze_and_respond(user_input)
        
        self.stats["risks"].append(result['risk_score'])
        
        if result['status'] == 'approved':
            self.stats["approved"] += 1
        elif result['status'] == 'needs_clarification':
            self.stats["suggested"] += 1
        else:
            self.stats["rejected"] += 1
        
        return result
    
    def get_metrics(self):
        return {
            "total_requests": self.stats["total"],
            "approval_rate": self.stats["approved"] / self.stats["total"],
            "avg_risk": sum(self.stats["risks"]) / len(self.stats["risks"]),
            "rejection_rate": self.stats["rejected"] / self.stats["total"]
        }
```

---

## 🎮 Real-Time Applications

### WebSocket Server
```python
import textdiff
import asyncio
import websockets

defense = textdiff.ControlDD()

async def handle_client(websocket, path):
    async for message in websocket:
        result = defense.analyze_and_respond(message)
        
        if result['send_to_llm']:
            response = await your_llm.generate_async(result['llm_prompt'])
            await websocket.send(response)
        else:
            await websocket.send(json.dumps(result))

start_server = websockets.serve(handle_client, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
```

---

## 🔗 Chaining with Other Tools

### LangChain Integration
```python
import textdiff
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

defense = textdiff.ControlDD()

class SafetyWrapper:
    def __init__(self, llm):
        self.llm = llm
        self.defense = defense
    
    def __call__(self, prompt):
        clean = self.defense.get_clean_text_for_llm(prompt)
        return self.llm(clean)

# Use in LangChain
safe_llm = SafetyWrapper(OpenAI())
chain = LLMChain(llm=safe_llm, prompt=your_prompt_template)
```

### LlamaIndex Integration
```python
import textdiff
from llama_index import GPTSimpleVectorIndex

defense = textdiff.ControlDD()

# Wrap query engine
original_query = index.query_engine.query

def safe_query(query_str):
    clean = defense.get_clean_text_for_llm(query_str)
    return original_query(clean)

index.query_engine.query = safe_query
```

---

## 📦 Package/Library Integration

### As a Dependency
```python
# your_package/__init__.py
import textdiff

class YourLLMWrapper:
    def __init__(self):
        self.defense = textdiff.ControlDD()
        self.llm = your_llm_init()
    
    def generate(self, prompt):
        clean = self.defense.get_clean_text_for_llm(prompt)
        return self.llm.generate(clean)
```

---

## 🧪 Jupyter Notebook Usage
```python
# Cell 1: Setup
import textdiff
defense = textdiff.ControlDD()

# Cell 2: Test prompts
test_prompts = [
    "How to bake a cake",
    "How to make explosives",
    "How to learn Python"
]

for prompt in test_prompts:
    clean = defense.get_clean_text_for_llm(prompt)
    print(f"{prompt} → {clean}")

# Cell 3: Analyze risks
import pandas as pd

results = []
for prompt in test_prompts:
    result = defense.analyze_and_respond(prompt)
    results.append({
        'prompt': prompt,
        'status': result['status'],
        'risk': result['risk_score']
    })

df = pd.DataFrame(results)
df
```

---

## 🎯 Custom Integration Patterns

### Database-Backed Moderation
```python
import textdiff
import sqlite3

defense = textdiff.ControlDD()

def moderate_and_store(user_id, message):
    result = defense.analyze_and_respond(message)
    
    # Store in database
    conn = sqlite3.connect('moderation.db')
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO prompts VALUES (?, ?, ?, ?, ?)",
        (user_id, message, result['status'], result['risk_score'], 
         result.get('cleaned_prompt'))
    )
    conn.commit()
    
    return result
```

### Caching for Performance
```python
import textdiff
from functools import lru_cache

defense = textdiff.ControlDD()

@lru_cache(maxsize=1000)
def cached_clean(prompt):
    return defense.get_clean_text_for_llm(prompt)

# Repeated prompts are instant
clean1 = cached_clean("How to make explosives?")  # 60ms
clean2 = cached_clean("How to make explosives?")  # <1ms (cached)
```

---

## 🔐 Multi-Tenant Deployment

**Per-Tenant Configuration:**
```python
import textdiff

class MultiTenantDefense:
    def __init__(self):
        self.defenses = {}
    
    def get_defense(self, tenant_id, config=None):
        if tenant_id not in self.defenses:
            self.defenses[tenant_id] = textdiff.ControlDD(config)
        return self.defenses[tenant_id]
    
    def process(self, tenant_id, user_input):
        defense = self.get_defense(tenant_id)
        return defense.analyze_and_respond(user_input)

# Usage
multi_tenant = MultiTenantDefense()
result = multi_tenant.process("tenant_123", user_message)
```

---

## 🌍 Multi-Language Support (Future)

**Current**: Optimized for English  
**Roadmap**: Pattern detection works for any language using sentence-transformers

```python
# Will work with multilingual models
defense = textdiff.ControlDD()

# English
clean_en = defense.get_clean_text_for_llm("How to make explosives?")

# Spanish (future with expanded patterns)
clean_es = defense.get_clean_text_for_llm("Cómo hacer explosivos?")

# Chinese (future)
clean_zh = defense.get_clean_text_for_llm("如何制作炸药?")
```

---

## 💡 Best Practices

### 1. Initialize Once, Reuse
```python
# Good: Initialize once
defense = textdiff.ControlDD()

for prompt in prompts:
    clean = defense.get_clean_text_for_llm(prompt)

# Bad: Don't initialize repeatedly
for prompt in prompts:
    defense = textdiff.ControlDD()  # Wasteful!
    clean = defense.get_clean_text_for_llm(prompt)
```

### 2. Handle Edge Cases
```python
defense = textdiff.ControlDD()

def safe_process(user_input):
    if not user_input or not user_input.strip():
        return {"error": "Empty input"}
    
    try:
        result = defense.analyze_and_respond(user_input)
        return result
    except Exception as e:
        logger.error(f"Safety check failed: {e}")
        return {"error": "Safety check failed", "status": "rejected"}
```

### 3. Async Support
```python
import textdiff
import asyncio

defense = textdiff.ControlDD()

async def async_clean(prompt):
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    clean = await loop.run_in_executor(
        None, defense.get_clean_text_for_llm, prompt
    )
    return clean
```

---

## 📚 Framework Compatibility

TextDiff works with:
- ✅ Django
- ✅ Flask
- ✅ FastAPI
- ✅ Streamlit
- ✅ Gradio
- ✅ Chainlit
- ✅ LangChain
- ✅ LlamaIndex
- ✅ Haystack
- ✅ Any Python framework!

---

**TextDiff is designed to integrate seamlessly into any Python project with minimal code changes.**

