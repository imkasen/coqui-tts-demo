"""
API 请求
"""
import re

import numpy as np
import requests

# TODO: UI 中设置 URL，gradio 全局变量
# TODO：无服务端时 requests 的 timeout 问题
SERVER_URL: str = "http://127.0.0.1:8000"


def get_xttsv2_speakers():
    """
    获得 XTTS V2 发言者列表
    """
    return requests.get(url=f"{SERVER_URL}/xttsv2/speakers", timeout=5).json()


def get_xttsv2_languages():
    """
    获得 XTTS V2 支持语音列表
    """
    return requests.get(url=f"{SERVER_URL}/xttsv2/languages", timeout=5).json()


def send_put_xttsv2_tts(text: str, language: str, speaker: str | None):
    """
    发送 put 请求调用 XTTS v2 进行语音合成
    """
    data: dict[str, str] = {
        "text": text,
        "language": language,
        "speaker": speaker,
    }
    response = requests.put(url=f"{SERVER_URL}/xttsv2/tts", timeout=60, json=data)
    sample_rate, wav = response.json()
    return sample_rate, np.array(wav)


def send_put_tacotron2_tts(text: str):
    """
    发送 put 请求调用 Tacotron2 进行语音合成
    """
    # 添加中文句号手动截断语句，否则影响合成效果
    pattern = r"[\u3002\uFF01\uFF1F\uFF0C\uFF1B\uff1a]"  # 。！？，；：
    if not bool(re.search(pattern, text[-1])):
        text += "\u3002"

    response = requests.put(url=f"{SERVER_URL}/tacotron2/tts", timeout=60, json={"text": text})
    sample_rate, wav = response.json()
    return sample_rate, np.array(wav)
