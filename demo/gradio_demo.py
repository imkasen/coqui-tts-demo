"""
利用 Gradio 构建基本 UI 操作界面
"""
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
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

    Gradio 中使用 tts() 输出音质似乎不如 tts_to_file()
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tts: TTS
    wav: list[int]

    match model_name:
        case "tts_models/multilingual/multi-dataset/xtts_v2":
            if sys.platform == "win32":
                localappdata_path: str = os.getenv("LOCALAPPDATA")
            elif sys.platform == "linux":
                localappdata_path: str = Path.home().joinpath(".local/share")
            model_path: str = os.path.join(
                localappdata_path,
                "tts",
                "tts_models--multilingual--multi-dataset--xtts_v2",
            )
            config_path: str = os.path.join(model_path, "config.json")

            # 当前多语言模型使用 model_path 加载要快于 model_name 的方式
            if os.path.isdir(model_path):
                tts = TTS(model_path=model_path, config_path=config_path).to(device)
            else:
                tts = TTS(model_name=model_name).to(device)

            # tts.tts_to_file(text=text, speaker_wav=speaker_wav_path, language=language, file_path="output.wav")
            wav = tts.tts(text=text, speaker_wav=speaker_wav_path, language=language)

        case "tts_models/zh-CN/baker/tacotron2-DDC-GST":
            tts = TTS(model_name=model_name).to(device)
            # tts.tts_to_file(text=text, file_path="output.wav")
            wav = tts.tts(text=text)

    # return "output.wav"
    return tts.synthesizer.output_sample_rate, np.array(wav)


if __name__ == "__main__":
    # Gradio Interface
    # Inputs
    text_box = gr.Textbox(label="输入", placeholder="输入用于语音合成的文本内容", lines=5)
    model_name_box = gr.Dropdown(
        label="模型",
        info="选择需要使用的模型",
        choices=[
            "tts_models/multilingual/multi-dataset/xtts_v2",
            "tts_models/zh-CN/baker/tacotron2-DDC-GST",
        ],
    )
    speaker_wav_box = gr.Audio(label="语言克隆文件（可选）", type="filepath")
    language_box = gr.Radio(
        label="语言（可选）",
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
        title="Coqui.ai TTS Interface Demo",
        article="_注：只有多语言模型需要**上传语音克隆文件**和**选择语言**。_",
    ).queue()

    demo.launch(show_api=False)
