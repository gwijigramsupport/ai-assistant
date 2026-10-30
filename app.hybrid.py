import os, requests
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import numpy as np

app = Flask(__name__)

API_BASE_URL = os.environ.get("ANY_API_BASE_URL", "https://api.openai.com/v1")
API_SECRET_KEY = os.environ.get("ANY_API_KEY", os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))
TARGET_MODEL = os.environ.get("TARGET_MODEL", "gpt-4o-mini")

# FIXED: use your local file or a real raw github link
DATASET_URL = os.environ.get("DATASET_URL", "")  # leave empty to use local dataset.txt

client = OpenAI(base_url=API_BASE_URL, api_key=API_SECRET_KEY)
KNOWLEDGE_BASE = []

def get_embedding(text):
    try:
        r = client.embeddings.create(model="text-embedding-3-small", input=[text])
        return r.data[0].embedding
    except:
        h = [0.0]*1536
        for i,ch in enumerate(text[:1536]): h[i]=float(ord(ch))/255.0
        return h

def load_dataset():
    global KNOWLEDGE_BASE
    KNOWLEDGE_BASE=[]
    text_data=""
    if DATASET_URL:
        print(f"Fetching remote dataset from: {DATASET_URL}")
        try:
            resp=requests.get(DATASET_URL, timeout=15)
            if resp.status_code==200: text_data=resp.text
            else: print(f"Failed fetch {resp.status_code}")
        except Exception as e: print(f"Network error: {e}")
    else:
        # fallback to local dataset.txt
        try:
            with open("dataset.txt","r",encoding="utf-8") as f: text_data=f.read()
            print(f"Loaded local dataset.txt {len(text_data)} chars")
        except Exception as e: print(f"No dataset.txt: {e}")

    if text_data:
        for chunk in text_data.split("\n\n"):
            t=chunk.strip()
            if t: KNOWLEDGE_BASE.append({"text":t,"vector":get_embedding(t)})
        print(f"Loaded {len(KNOWLEDGE_BASE)} blocks")

load_dataset()

def retrieve_relevant_context(q, top_k=1):
    if not KNOWLEDGE_BASE: return ""
    qv=get_embedding(q)
    scores=[]
    for item in KNOWLEDGE_BASE:
        dot=np.dot(qv,item["vector"]); nq=np.linalg.norm(qv); ni=np.linalg.norm(item["vector"])
        sim=dot/(nq*ni) if nq and ni else 0
        scores.append((sim,item["text"]))
    scores.sort(key=lambda x:x[0], reverse=True)
    return "\n".join([t for s,t in scores[:top_k]])

HTML_TEMPLATE = '''<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI Core</title>
<style>body{font-family:system-ui;background:#0b0f19;margin:0;display:flex;justify-content:center;align-items:center;height:100vh;color:#fff}
.chat-container{width:480px;height:620px;background:#111827;border-radius:12px;display:flex;flex-direction:column;overflow:hidden;border:1px solid #1f2937}
.chat-header{background:linear-gradient(135deg,#ec4899,#8b5cf6);padding:20px;text-align:center;font-weight:bold}
.chat-messages{flex:1;padding:20px;overflow-y:auto;display:flex;flex-direction:column;gap:12px}
.message{padding:12px 16px;border-radius:8px;max-width:80%;line-height:1.4;word-wrap:break-word}
.user-message{background:#2563eb;align-self:flex-end;border-bottom-right-radius:2px}
.ai-message{background:#1f2937;align-self:flex-start;border:1px solid #374151;border-bottom-left-radius:2px}
.chat-input-area{display:flex;padding:15px;border-top:1px solid #1f2937;background:#111827}
.chat-input{flex:1;padding:12px;background:#0b0f19;border:1px solid #374151;border-radius:6px;color:white;outline:none}
.send-btn{background:#ec4899;color:white;border:none;padding:12px 20px;margin-left:10px;border-radius:6px;cursor:pointer;font-weight:bold}</style></head><body>
<div class="chat-container"><div class="chat-header">Remote URL Knowledge Core</div><div class="chat-messages" id="chat-messages"><div class="message ai-message">System online. Ask me anything!</div></div>
<div class="chat-input-area"><input id="chat-input" class="chat-input" placeholder="Query..." onkeypress="if(event.key==='Enter')sendMessage()"><button class="send-btn" onclick="sendMessage()">Send</button></div></div>
<script>const cm=document.getElementById('chat-messages'),ci=document.getElementById('chat-input');function add(t,c){const d=document.createElement('div');d.className=`message ${c}`;d.innerText=t;cm.appendChild(d);cm.scrollTop=cm.scrollHeight}
async function sendMessage(){const m=ci.value.trim();if(!m)return;add(m,'user-message');ci.value='';try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});const d=await r.json();add(d.response||"Error: "+d.error,'ai-message')}catch{add("Failed to reach API",'ai-message')}}</script></body></html>'''

@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat_route():
    um=request.json.get("message","")
    if not um: return jsonify({"error":"Empty"}),400
    try:
        ctx=retrieve_relevant_context(um, top_k=1)
        sys_prompt=f"You are expert assistant. Answer based on:\n\nURL DATASET CONTEXT:\n{ctx if ctx else 'No matching context.'}"
        comp=client.chat.completions.create(model=TARGET_MODEL, messages=[{"role":"system","content":sys_prompt},{"role":"user","content":um}], temperature=0.3)
        return jsonify({"response": comp.choices[0].message.content})
    except Exception as e: return jsonify({"error": str(e)}),500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
