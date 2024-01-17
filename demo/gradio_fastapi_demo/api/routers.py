"""
FastAPI 路由文件
"""

import os
import tempfile
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from scipy.io import wavfile

from .models import XTTSV2Factory, XTTSV2Model, ZhTacotron2Factory, ZhTacotron2Model

api = APIRouter()

xttsv2_factory = XTTSV2Factory()
zhtacotron2_factory = ZhTacotron2Factory()


class UploadContent(BaseModel):
    """
    tts 所需要的参数信息
    """

    text: str
    language: str | None = None
    speaker: str | None = None
    sample_rate: int | None = None
    wav_list: list[Any] | None = None
    dtype_name: str | None = None


# Routers
@api.get(path="/xttsv2/speakers", response_model=list[str])
async def get_xttsv2_speakers(model: XTTSV2Model = Depends(xttsv2_factory.get_model)) -> list[str]:
    """
    获得 XTTS v2 模型所支持的发言者列表
    """
    return model.get_speakers()


@api.get(path="/xttsv2/languages", response_model=list[str])
async def get_xttsv2_languages(model: XTTSV2Model = Depends(xttsv2_factory.get_model)) -> list[str]:
    """
    获得 XTTS v2 模型所支持的语言列表
    """
    return model.get_languages()


@api.put(path="/xttsv2/tts")
async def xttsv2_tts(content: UploadContent, model: XTTSV2Model = Depends(xttsv2_factory.get_model)):
    """
    语音合成
    """
    tmp_wav_path: str = os.path.join(tempfile.gettempdir(), "temp_voice_cloning_file.wav")

    if content.sample_rate and content.wav_list and content.dtype_name:
        wavfile.write(
            filename=tmp_wav_path,
            rate=content.sample_rate,
            data=np.array(content.wav_list).astype(content.dtype_name),
        )

    return model.text_to_speech(
        text=content.text,
        language=content.language,
        speaker=content.speaker,
        speaker_wav_path=tmp_wav_path,
    )


@api.put(path="/tacotron2/tts")
async def tacotron2_tts(content: UploadContent, model: ZhTacotron2Model = Depends(zhtacotron2_factory.get_model)):
    """
    语音合成
    """
    return model.text_to_speech(text=content.text)
