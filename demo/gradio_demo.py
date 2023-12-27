"""
利用 Gradio 构建 UI 操作界面
"""
import os
import sys

import gradio as gr
import torch
from TTS.api import TTS


def run_tts(
    text: str,
    model_name: str,
    speaker_wav_path: str | None = None,
    language: str | None = None,
) -> str:
    """
    运行 TTS
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO
    # Text to speech list of amplitude values as output
    # return tts.tts(text=text, speaker_wav=speaker_wav_path, language=language)

    match model_name:
        case "tts_models/multilingual/multi-dataset/xtts_v2":
            if sys.platform == "win32":
                localappdata_path: str = os.getenv("LOCALAPPDATA")
                model_path: str = os.path.join(
                    localappdata_path,
                    "tts",
                    "tts_models--multilingual--multi-dataset--xtts_v2",
                )
                config_path: str = os.path.join(model_path, "config.json")
                # 当前模型使用 model_path 加载要快于 model_name 的方式
                tts: TTS = TTS(model_path=model_path, config_path=config_path).to(device)
                tts.tts_to_file(text=text, speaker_wav=speaker_wav_path, language=language, file_path="output.wav")
        case "tts_models/zh-CN/baker/tacotron2-DDC-GST":
            tts: TTS = TTS(model_name=model_name).to(device)
            tts.tts_to_file(text=text, file_path="output.wav")
    return "output.wav"


if __name__ == "__main__":
    # Gradio 组件
    # Inputs
    text_box = gr.Textbox(label="输入", placeholder="输入用于语音合成的文本内容")
    model_name_box = gr.Dropdown(
        label="模型",
        info="选择需要使用的模型",
        choices=[
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/zh-CN/baker/tacotron2-DDC-GST",
        ],
    )
    speaker_wav_box = gr.Audio(label="语言克隆文件", type="filepath")
    language_box = gr.Radio(
        label="语言",
        choices=["en", "zh"],
    )
    # Outputs
    output_audio_box = gr.Audio(label="语言合成结果")

    # UI 界面
    demo = gr.Interface(
        fn=run_tts,
        inputs=[
            text_box,
            model_name_box,
            speaker_wav_box,
            language_box,
        ],
        outputs=output_audio_box,
        allow_flagging="never",
    )  # TODO: queue

    demo.launch(show_api=False)
