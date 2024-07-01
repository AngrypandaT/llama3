import ollama
import json
from tqdm import tqdm

text_path = '/root/autodl-tmp/xd_all_captions.txt'
text_loader = {}
with open(text_path, 'r') as f:
    for line in f:
        if line.strip():
            line_data = json.loads(line.strip())
            text_loader.update(line_data)
text_loader = dict(sorted(text_loader.items()))
phrases = ['car accident']
result_dict = {}
progress_bar = tqdm(total=len(text_loader), desc='Processing items')

for item in text_loader:
    answer_list = []
    for sentence in text_loader[item]:
        sentence = sentence
        labels = 'normal, fighting, shooting, riot, abuse, car accident, explosion'
        messages = [{'role': 'user',
                     'content': f'You have 7 labels :{labels}, please analyze this sentence:{sentence} , you must select the closest label from the labels, provide only one word answer and stop analyze without any other answer, and do not allow the use of words other than those provided with the labels. Keep lowercase.'}]
        stream = ollama.chat(
            model='llama3:latest',  # 使用你提供的模型名称
            messages=messages,
            stream=True,  # 开启流式响应
        )

        buffer = ""
        for chunk in stream:
            if 'message' in chunk:
                buffer += chunk['message']['content']
                for phrase in phrases:
                    if phrase in buffer:
                        answer_list.append(phrase)
                        buffer = buffer.replace(phrase, '').strip()

                words = buffer.split()
                # 保留最后一个单词在缓冲区中，因为它可能是不完整的
                buffer = words.pop() if words else ""
                for word in words:
                    if word:  # 过滤掉空字符串
                        answer_list.append(word.strip())
        # 输出最后剩余的完整单词
        if buffer and not buffer.endswith((' ', '.', ',', '!', '?', ';')):
            answer_list.append(buffer.strip())
    print(answer_list)
    result_dict[item] = answer_list
    progress_bar.update(1)
progress_bar.close()
with open('/root/autodl-tmp/xd_llama3_swin_cap2label.txt', 'w') as f:
    for item, answers in result_dict.items():
        json_object = json.dumps({item: answers}, ensure_ascii=False)
        f.write(f"{json_object}\n")
