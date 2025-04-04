from openai import OpenAI
import gradio as gr
import os
import json
from typing import List, Tuple

# 对话类，使用message传递对话内容，使用OpenAI API进行对话,num_rounds为对话轮数，每次对话双方内容都会存储在messages列表中
class Conversation:
    # 初始化，确定模型，发送初始prompt
    def __init__(self, model, prompt, max_num_rounds=10):
        self.BaseUrl = "https://api.siliconflow.cn/v1"
        self.APIkey = "sk-mscsqwnzejryevexgdojppyuhhwwyiobmtehhtcjmgfofpxf"

        self.num_rounds = max_num_rounds
        self.client = OpenAI(api_key=self.APIkey, base_url=self.BaseUrl)
        self.model = model
        self.messages = []
        self.messages.append({'role': 'system', 'content': prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )

    def ask(self, question, temp=1.0):
        self.messages.append({'role': 'user', 'content': question})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=temp,
                stream=True
            )
            if len(self.messages) >= self.num_rounds * 2 + 1:
                del self.messages[1:3]
            new_mes = ''
            for chunk in response:
                if not chunk.choices:
                    continue
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    new_mes += chunk.choices[0].delta.content
                if chunk.choices[0].delta.reasoning_content:
                    print(chunk.choices[0].delta.reasoning_content, end="",
                          flush=True)
            print('\n')
            self.messages.append({'role': 'assistant', 'content': new_mes})

        except Exception as e:
            print("ERROR: 发生错误")

    def interact_chat(self, chatbot: List[Tuple[str, str]], user_input: str,
                      temp=1.0) -> List[Tuple[str, str]]:
        """
        * 参数:
          - user_input: 每轮对话中的用户输入
          - temp: 模型的温度参数。温度用于控制聊天机器人的输出。温度越高，响应越具创造性。
        """
        try:
            messages = []
            for input_text, response_text in chatbot:
                messages.append({'role': 'user', 'content': input_text})
                messages.append({'role': 'assistant', 'content': response_text})

            messages.append({'role': 'user', 'content': user_input})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # 包含用户的输入和对话历史
                temperature=temp,  # 使用温度参数控制创造性
            )
            chatbot.append((user_input, response.choices[0].message.content))

        except Exception as e:
            print(f"发生错误：{e}")
            chatbot.append((user_input, f"抱歉，发生了错误：{e}"))
        return chatbot

    def reset(self):
        self.messages = []
        return []

    def export_chat(self, description: str) -> None:
        """
        * 参数:
          - description: 此任务的描述
        """
        target = {"chatbot": self.messages, "description": description}
        file_path = 'files/dialogue_history.json'
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(file_path, "w", encoding="utf-8") as file:  # 修改为 file_path
                json.dump(target, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"导出对话时发生错误：{e}")


if __name__ == '__main__':
    character_for_chatbot = 'assistant'

    prompt_for_dialogue = "你是一个有用的AI助手，接下来请回答我的问题"
    model = 'Pro/Qwen/Qwen2.5-VL-7B-Instruct'
    # model = 'deepseek-ai/DeepSeek-V3'
    conversation = Conversation(model, prompt_for_dialogue)

    # 进行第一次对话
    first_dialogue = conversation.interact_chat([], prompt_for_dialogue)

    # 生成 Gradio 的UI界面
    with gr.Blocks() as demo:
        gr.Markdown(
            f"# 我是你的AI助手，试着问我一些问题！")
        chatbot = gr.Chatbot(value=first_dialogue)
        description_textbox = gr.Textbox(label="机器人扮演的角色",
                                         interactive=False,
                                         value=f"{character_for_chatbot}")
        input_textbox = gr.Textbox(label="输入", value="")

        with gr.Column():
            gr.Markdown(
                "# 温度调节\n温度用于控制聊天机器人的输出。温度越高，响应越具创造性。")
            temperature_slider = gr.Slider(0.0, 1.9, 1.0, step=0.1,
                                           label="温度")

        with gr.Row():
            sent_button = gr.Button(value="发送")
            reset_button = gr.Button(value="重置")

        with gr.Column():
            gr.Markdown("# 保存结果\n当你对结果满意后，点击导出按钮保存结果。")
            export_button = gr.Button(value="导出")

        # JavaScript代码来监听回车键事件
        js_code = """
            input_textbox.addEventListener('keydown', function(event){
                if (event.key === 'Enter') {
                    sent_button.click();
                }
            });
            """
        gr.HTML(js_code)

        # 连接按钮与函数
        sent_button.click(conversation.interact_chat,
                          inputs=[chatbot, input_textbox,
                                  temperature_slider],
                          outputs=[chatbot])
        reset_button.click(conversation.reset, outputs=[chatbot])
        export_button.click(conversation.export_chat,
                            inputs=[chatbot])

        # 监听回车键事件
        input_textbox.submit(conversation.interact_chat,
                             inputs=[chatbot, input_textbox,
                                     temperature_slider],
                             outputs=[description_textbox])

        # 启动 Gradio 界面
        demo.launch(debug=True)

    '''
    * 参数：
      -debug=True 在调试模式下启动，显示详细的报错信息，更改代码后无需手动重启
    '''
