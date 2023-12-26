import torch
from TTS.api import TTS
from pathlib import Path
import os
import time


def print_all_models():
    """
    列出可用的 TTS 模型
    命令行：tts --list_models
    """
    for model in TTS().list_models().list_models():
        print(model)


def init_tts(model_name: str) -> TTS:
    """
    初始化 TTS 模型，从网上下载模型到默认路径，并从默认安装路径中加载模型
    
    注意：使用 model_dir_path 方式存在 bug，但可能会快于 model_name 的方式

    :param model_name: 模型名
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    return TTS(model_name=model_name).to(device)


def run_tts(tts: TTS, text: str, *, speaker_wav_path: str | None = None, language: str | None = None, output_path: str | None = None):
    """
    执行语音合成

    :param tts: TTS 模型
    :param text: 需要生成的文本
    :param speaker_wav_path: 用于语音克隆的参考音频文件的路径
    :param language: 使用语言
    :param output_path: 输出文件路径，默认为 None
    """
    # Text to speech list of amplitude values as output
    # if speaker_wav_path and language:
    #     wav: list[int] = tts.tts(text=text, speaker_wav=speaker_wav_path, language=language)

    # Text to speech to a file
    if speaker_wav_path and language:
        tts.tts_to_file(text=text, speaker_wav=speaker_wav_path, language=language, file_path=output_path)
    if speaker_wav_path is None and language is None:
        tts.tts_to_file(text=text, file_path=output_path)
    

if __name__ == "__main__":
    t1: float = time.perf_counter()
    
    PROJ_ROOT: Path = Path().resolve()
    OUTPUT_DIR_PATH: str = os.path.join(PROJ_ROOT, "output")
    
    if not (os.path.exists(OUTPUT_DIR_PATH) and os.path.isdir(OUTPUT_DIR_PATH)):
        os.makedirs(OUTPUT_DIR_PATH)
    
    REF_FILE_PATH: str = os.path.join(PROJ_ROOT, "input", "jarvis.mp3")
    EN_TXT: str = "Hello, I'am Jarvis, your personal assistant! How can I help you today?"
    ZH_TXT: str = "你好，我是贾维斯，你的个人助手！今天我能为您做些什么？"
    
    # ========
    
    # print_all_models()
    
    # ==== Running a multi-speaker and multi-lingual model ====
    t3: float = time.perf_counter()
    tts: TTS = init_tts(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    t4: float = time.perf_counter()
    
    # Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    t5: float = time.perf_counter()
    run_tts(tts, 
            EN_TXT, 
            speaker_wav_path=REF_FILE_PATH, 
            language="en", 
            output_path=os.path.join(OUTPUT_DIR_PATH, "output-en.wav"))
    # run_tts(tts, 
    #         ZH_TXT, 
    #         speaker_wav_path=REF_FILE_PATH, 
    #         language="zh", 
    #         output_path=os.path.join(OUTPUT_DIR_PATH, "output-zh.wav"))
    t6: float = time.perf_counter()

    # ==== Running a single speaker model ====
    # t3: float = time.perf_counter()
    # tts: TTS = init_tts(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST")
    # t4: float = time.perf_counter()
    
    # t5: float = time.perf_counter()
    # run_tts(tts, text=ZH_TXT, output_path=os.path.join(OUTPUT_DIR_PATH, "output-zh2.wav"))
    # t6: float = time.perf_counter()
    
    # =========

    t2: float = time.perf_counter()
    print(f"Total time: {(t2 - t1):.2f}s, init time: {(t4 - t3):.2f}s, conversion time: {(t6 - t5):.2f}s")
