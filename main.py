from openai import OpenAI
import openai

# 발급받은 API 키 설정
OPENAI_API_KEY = 'YOUR API KEY'

client = OpenAI(api_key = OPENAI_API_KEY)

pre_prompt = """"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

import json
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def default():
    return 'Hello World!'

@app.route('/create', methods=['OPTIONS', 'POST'])
def create():
    if request.method == 'OPTIONS':
        return json.dumps({'message': 'Preflight request handled'}), 200
        
    params = json.loads(request.get_data())
    line = params['prompt']
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."},
            {"role": "user",
            "content": f"{pre_prompt} {line}"}],
    )

    output_text = response.choices[0].message.content
    image = pipe(output_text).images[0]
    image.save("output.png")
    return send_file('output.png', mimetype='image/png'), 200
    
if __name__ == '__main__':
    app.run()
