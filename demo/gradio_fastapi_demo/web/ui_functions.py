"""
Gradio 组件所需的方法
"""
import gradio as gr
from numpy import ndarray

from .api_requests import get_xttsv2_languages, get_xttsv2_speakers, send_post_tacotron2_tts, send_post_xttsv2_tts


def xttsv2_list_languages(url: str) -> gr.Dropdown:
    """
    为 XTTS V2 语音下拉栏获取选项列表
    """
    list_cnt: list[str] = get_xttsv2_languages(url)
    return gr.Dropdown(choices=list_cnt)


def xttsv2_list_speakers(url: str) -> gr.Dropdown:
    """
    为 XTTS V2 发言者下拉栏获取选项列表
    """
    list_cnt: list[str] = get_xttsv2_speakers(url)
    return gr.Dropdown(choices=list_cnt)


def xttsv2_submit(
    url: str,
    text: str | None,
    language: list | str | None,
    speaker: list | str | None,
    wav_audio: tuple[int, ndarray] | None,
):
    """
    发送 post 请求
    """
    if not text:
        raise gr.Error("文本不能为空！")
    if not language:
        raise gr.Error("语音不能为空！")
    if not speaker and not wav_audio:
        raise gr.Error("发言者和语音克隆文件必须选择一项！")
    if speaker and wav_audio:
        speaker = None
        gr.Warning("使用语音克隆文件处理语音合成，不使用发言者选项。")
    return send_post_xttsv2_tts(url, text, language, speaker, wav_audio)


def tacotron2_submit(url: str, text: str | None):
    """
    发送 post 请求
    """
    if not text:
        raise gr.Error("文本不能为空！")
    return send_post_tacotron2_tts(url, text)
