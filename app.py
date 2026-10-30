from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import os

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"))

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Chat</title><style>
body{font-family:Arial;background:#f4f7f6;margin:0;display:flex;justify-content:center;align-items:center;height:100vh}
.chat-container{width:450px;height:600px;background:white;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,.1);display:flex;flex-direction:column;overflow:hidden}
.chat-header{background:#075e54;color:white;padding:15px;text-align:center;font-weight:bold}
.chat-messages{flex:1;padding:20px;overflow-y:auto;display:flex;flex-direction:column;gap:10px}
.message{padding:10px 15px;border-radius:15px;max-width:75%;word-wrap:break-word}
.user-message{background:#dcf8c6;align-self:flex-end}
.ai-message{background:#f1f0f0;align-self:flex-start}
.chat-input-area{display:flex;padding:15px;border-top:1px solid #ddd}
.chat-input{flex:1;padding:10px;border:1px solid #ccc;border-radius:20px;outline:none;padding-left:15px}
.send-btn{background:#075e54;color:white;border:none;padding:10px 20px;margin-left:10px;border-radius:20px;cursor:pointer;font-weight:bold}
</style></head><body>
<div class="chat-container">
<div class="chat-header">AI Chat Assistant</div>
<div class="chat-messages" id="chat-messages"><div class="message ai-message">Hello! How can I help you today?</div></div>
<div class="chat-input-area"><input id="chat-input" class="chat-input" placeholder="Type a message..." onkeypress="if(event.key==='Enter')sendMessage()"><button class="send-btn" onclick="sendMessage()">Send</button></div>
</div>
<script>
const chatMessages=document.getElementById('chat-messages');const chatInput=document.getElementById('chat-input');
function appendMessage(t,c){const d=document.createElement('div');d.className=`message ${c}`;d.innerText=t;chatMessages.appendChild(d);chatMessages.scrollTop=chatMessages.scrollHeight}
async function sendMessage(){const m=chatInput.value.trim();if(!m)return;appendMessage(m,'user-message');chatInput.value='';try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m})});const data=await r.json();appendMessage(data.response||"Error: "+data.error,'ai-message')}catch{appendMessage("Failed to connect",'ai-message')}}
</script></body></html>
'''

@app.route('/')
def home(): return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message","")
    if not user_message: return jsonify({"error":"Empty message"}),400
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"You are a customer assistant trained on custom data. Respond clearly."},{"role":"user","content":user_message}], temperature=0.7)
        return jsonify({"response": resp.choices[0].message.content})
    except Exception as e: return jsonify({"error": str(e)}),500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
