from openai import OpenAI

class Conversation:
    def __init__(self,model,prompt,num_rounds=3):
        self.num_rounds = num_rounds
        self.client = OpenAI(api_key="sk-lcjxpijtmznhqojajxzespzrudcyrzadosrcvcsphzfdtbpe", 
                        base_url="https://api.siliconflow.cn/v1")
        self.model = model
        self.messages = []
        self.messages.append({'role': 'system', 'content': prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )

    def ask(self,question):
        self.messages.append({'role': 'user', 'content': question})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )
        if len(self.messages) >= self.num_rounds*2+1:
            del self.messages[1:3]
        new_mes=''
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_mes+=chunk.choices[0].delta.content
            if chunk.choices[0].delta.reasoning_content:
                print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
        print('\n')
        self.messages.append({'role': 'assistant', 'content': new_mes})



if __name__ == '__main__':
    promt = "你是一个会回答问题的人工智能"
    model = 'Qwen/Qwen2.5-7B-Instruct'
    conversation = Conversation(model,promt)
    conversation.ask("1+1=？")
    conversation.ask("我的第一个问题是什么？")