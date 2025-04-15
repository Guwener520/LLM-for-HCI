from openai import OpenAI
import gradio as gr
import os
import json
import re
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
        self.all_mess = {'Pro/Qwen/Qwen2.5-VL-7B-Instruct': [],
                         'THUDM/chatglm3-6b': [],
                         'deepseek-ai/DeepSeek-V3': []}
        self.messages = []
        self.messages.append({'role': 'system', 'content': prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=True
        )

    def switch_model(self, new_model):
        clean_model = process_model_choice(new_model)
        self.all_mess[self.model] = self.messages
        self.messages = self.all_mess[clean_model]
        self.model = clean_model
        return []

    def interact_chat(self, chatbot: List[Tuple[str, str]], user_input: str,
                      temp=1.0) -> Tuple[List[Tuple[str, str]], str]:
        """
        * 参数:
          - user_input: 每轮对话中的用户输入
          - temp: 模型的温度参数。温度用于控制聊天机器人的输出。温度越高，响应越具创造性。
        """
        try:
            messages = self.messages
            messages.append({'role': 'user', 'content': user_input})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # 包含用户的输入和对话历史
                temperature=temp,  # 使用温度参数控制创造性
            )
            messages.append({'role': 'assistant',
                             'content': response.choices[0].message.content})
            chatbot.append((user_input, response.choices[0].message.content))
            self.messages = messages

        except Exception as e:
            print(f"发生错误：{e}")
            chatbot.append((user_input, f"抱歉，发生了错误：{e}"))
        return chatbot, ""

    def reset(self):
        self.all_mess['Pro/Qwen/Qwen2.5-VL-7B-Instruct'] = []
        self.all_mess['THUDM/chatglm3-6b'] = []
        self.all_mess['deepseek-ai/DeepSeek-V3'] = []

        self.messages = []
        return []

    def export_chat(self, description: str) -> None:
        """
        * 参数:
          - model: 选择的模型
          - messages: 对话记录
        """
        target = [{"model": 'Qwen2.5-VL-7B-Instruct',
                   "description": self.all_mess[
                       'Pro/Qwen/Qwen2.5-VL-7B-Instruct']},
                  {"model": 'chatglm3-6b',
                   "description": self.all_mess['THUDM/chatglm3-6b']},
                  {"model": 'deepseek-ai/DeepSeek-V3',
                   "description": self.all_mess[
                       'deepseek-ai/DeepSeek-V3']}]
        file_path = 'files/dialogue_history.json'
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        try:
            with open(file_path, "w",
                      encoding="utf-8") as file:  # 修改为 file_path
                json.dump(target, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"导出对话时发生错误：{e}")


def process_model_choice(selected_model):
    pattern = r'#(.*?)#'
    matches = re.findall(pattern, selected_model)
    selected_model = selected_model.replace(f'#{matches[0]}#', '')
    return selected_model


if __name__ == '__main__':
    character_for_chatbot = 'assistant'
    prompt_for_dialogue = "你是一个有用的AI助手，接下来请回答我的问题"

    model1 = 'Pro/Qwen/Qwen2.5-VL-7B-Instruct'
    model2 = 'THUDM/chatglm3-6b'
    model3 = 'deepseek-ai/DeepSeek-V3'

    models = [model1, model2, model3]

    model_description = [model1 + '#全面的AI助手，善于推理、编程#',
                         model2 + '#开源模型，免费调用，对话流畅#',
                         model3 + '#深度思考，回答准确，反应较慢#']

    conversation = Conversation(model1, prompt_for_dialogue)

    # 进行第一次对话
    first_dialogue = conversation.interact_chat([], prompt_for_dialogue)

    # 生成 Gradio 的UI界面
    with gr.Blocks() as demo:
        gr.Markdown(
            f"# 我是你的AI助手，试着问我一些问题！")
        chatbot = gr.Chatbot(value=first_dialogue[0])
        input_textbox = gr.Textbox(label="输入", value="")

        with gr.Row():
            sent_button = gr.Button(value="发送")
            reset_button = gr.Button(value="重置")

        with gr.Column():
            gr.Markdown("# 选择模型")
            model_dropdown = gr.Dropdown(choices=model_description,
                                         label="选择模型",
                                         value=model_description[0])

        with gr.Column():
            gr.Markdown(
                "# 温度调节\n温度用于控制聊天机器人的输出。温度越高，响应越具创造性。")
            temperature_slider = gr.Slider(0.0, 1.9, 1.0, step=0.1,
                                           label="温度")

        with gr.Column():
            gr.Markdown("# 保存结果\n当你对结果满意后，点击导出按钮保存结果。")
            export_button = gr.Button(value="导出")

        # 连接按钮与函数
        sent_button.click(conversation.interact_chat,
                          inputs=[chatbot, input_textbox,
                                  temperature_slider],
                          outputs=[chatbot, input_textbox])
        model_dropdown.change(conversation.switch_model, inputs=model_dropdown)
        reset_button.click(conversation.reset, outputs=[chatbot])
        export_button.click(conversation.export_chat,
                            inputs=[chatbot])

        # submit函数监听回车键事件
        input_textbox.submit(conversation.interact_chat,
                             inputs=[chatbot, input_textbox,
                                     temperature_slider],
                             outputs=[chatbot, input_textbox])

        # 启动 Gradio 界面
        demo.launch(debug=True)
        '''
        * 参数：
          -debug=True 在调试模式下启动，显示详细的报错信息，更改代码后无需手动重启
        '''
