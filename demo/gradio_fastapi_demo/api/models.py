"""
TTS Models
"""
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from TTS.api import TTS

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class XTTSV2Model:
    """
    tts_models/multilingual/multi-dataset/xtts_v2
    """

    def __init__(self) -> None:
        self.model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"

        if sys.platform == "win32":
            localappdata_path: str = os.getenv("LOCALAPPDATA")
        elif sys.platform == "linux":
            localappdata_path: str = Path.home().joinpath(".local/share")

        self.model_path: str = os.path.join(
            localappdata_path,
            "tts",
            "tts_models--multilingual--multi-dataset--xtts_v2",
        )
        self.config_path: str = os.path.join(self.model_path, "config.json")

        # 当前多语言模型使用 model_path 加载要快于 model_name 的方式
        if os.path.isdir(self.model_path):
            self.tts: TTS = TTS(model_path=self.model_path, config_path=self.config_path).to(DEVICE)
        else:
            self.tts = TTS(model_name=self.model_name).to(DEVICE)

    def get_speakers(self) -> list[str]:
        """
        获得可用的发言者
        """
        # return self.tts.speakers  # AttributeError: 'TTS' object has no attribute 'speakers'
        return list(self.tts.synthesizer.tts_model.speaker_manager.name_to_id)

    def get_languages(self) -> list[str]:
        """
        获得支持的语言列表
        """
        return self.tts.languages

    def text_to_speech(
        self,
        text: str,
        language: str,
        speaker: str | None = None,
        speaker_wav_path: str | None = None,
    ):
        """
        语言合成
        """
        # TODO: emotion
        wav: list[int] = self.tts.tts(text=text, language=language, speaker=speaker, speaker_wav=speaker_wav_path)
        if os.path.isfile(speaker_wav_path):  # delete the temp voice cloning file
            os.remove(speaker_wav_path)
        # wav 先转为 ndarray，再转为 list。直接返回会导致错误
        return self.tts.synthesizer.output_sample_rate, np.array(wav).tolist()


class ZhTacotron2Model:
    """
    tts_models/zh-CN/baker/tacotron2-DDC-GST
    """

    def __init__(self) -> None:
        self.model_name: str = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
        self.tts: TTS = TTS(model_name=self.model_name).to(DEVICE)

    def text_to_speech(self, text: str):
        """
        语音合成
        """
        wav: list[int] = self.tts.tts(text=text)
        # 这里直接返回 list 也可以，但这里选择与上面保持样式一致。
        return self.tts.synthesizer.output_sample_rate, np.array(wav).tolist()


class XTTSV2Factory:
    """
    XTTS v2 单例模式工厂类
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_instance = XTTSV2Model()

    def get_model(self):
        """
        获得 XTTS v2 模型
        """
        return self.model_instance


class ZhTacotron2Factory:
    """
    Tacotron2 单例模式工厂类
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_instance = ZhTacotron2Model()

    def get_model(self):
        """
        获得 Tacotron2 模型
        """
        return self.model_instance
