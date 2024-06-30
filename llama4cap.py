import ollama
import json
from tqdm import tqdm

text_path = '/root/autodl-tmp/all_captions.txt'
text_loader = {}
with open(text_path, 'r') as f:
    for line in f:
        if line.strip():
            line_data = json.loads(line.strip())
            text_loader.update(line_data)
text_loader = dict(sorted(text_loader.items()))
result_dict = {}
progress_bar = tqdm(total=len(text_loader), desc='Processing items')

for item in text_loader:
    answer_list = []
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

        buffer = ""
        for chunk in stream:
            if 'message' in chunk:
                buffer += chunk['message']['content']
                words = buffer.split(" ")
                # 保留最后一个单词在缓冲区中，因为它可能是不完整的
                buffer = words.pop() if words[-1] != '' else ""
                for word in words:
                    if word:  # 过滤掉空字符串
                        answer_list.append(word.strip())
        # 输出最后剩余的完整单词
        if buffer:
            answer_list.append(buffer.strip())
    result_dict[item] = answer_list
    progress_bar.update(1)
progress_bar.close()
with open('result.txt', 'w') as f:
        f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))
