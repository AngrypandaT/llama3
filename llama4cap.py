import ollama
import json

text_path = '/root/autodl-tmp/all_captions.txt'
text_loader = {}
with open(text_path, 'r') as f:
    for line in f:
        if line.strip():
            line_data = json.loads(line.strip())
            text_loader.update(line_data)
text_loader = dict(sorted(text_loader.items()))
video_name = {}
answer_list = []
for item in text_loader:
    for sentence in text_loader[item]:
        sentence = sentence
        labels = 'Normal,Abuse,Arrest,Arson,Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism'
        messages = [{'role': 'user',
                     'content': f'we have 14 labels :{labels},please analyze this sentence:{sentence} ,You must answer the category of this sentence using one of the 14 tags, provide only one word answer, without any other answer'}]
        stream = ollama.chat(
            model='llama3:latest',  # 使用你提供的模型名称
            messages=messages,
            stream=True,  # 开启流式响应
        )
        for chunk in stream:
            if 'message' in chunk:
                # print(chunk['message']['content'], end='')
                answer = chunk['message']['content']
                answer_list.append(answer)
                print(answer_list)
