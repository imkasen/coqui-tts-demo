"""
API 请求
"""
import re

import numpy as np
import requests


def get_xttsv2_speakers(url: str):
    """
    获得 XTTS V2 发言者列表
    """
    return requests.get(url=f"{url}/xttsv2/speakers", timeout=5).json()


def get_xttsv2_languages(url: str):
    """
    获得 XTTS V2 支持语音列表
    """
    return requests.get(url=f"{url}/xttsv2/languages", timeout=5).json()


def send_put_xttsv2_tts(url: str, text: str, language: str, speaker: str | None):
    """
    发送 put 请求调用 XTTS v2 进行语音合成
    """
    data: dict[str, str] = {
        "text": text,
        "language": language,
        "speaker": speaker,
    }
    response = requests.put(url=f"{url}/xttsv2/tts", timeout=60, json=data)
    sample_rate, wav = response.json()
    return sample_rate, np.array(wav)


def send_put_tacotron2_tts(url: str, text: str):
    """
    发送 put 请求调用 Tacotron2 进行语音合成
    """
    # 添加中文句号手动截断语句，否则影响合成效果
    pattern = r"[\u3002\uFF01\uFF1F\uFF0C\uFF1B\uff1a]"  # 。！？，；：
    if not bool(re.search(pattern, text[-1])):
        text += "\u3002"

    response = requests.put(url=f"{url}/tacotron2/tts", timeout=60, json={"text": text})
    sample_rate, wav = response.json()
    return sample_rate, np.array(wav)
