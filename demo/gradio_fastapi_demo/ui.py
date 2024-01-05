"""
UI 界面
"""
import gradio as gr
from api_requests import get_xttsv2_languages, get_xttsv2_speakers, send_put_tacotron2_tts, send_put_xttsv2_tts
from numpy import ndarray


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
    发送 put 请求
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
    return send_put_xttsv2_tts(url, text, language, speaker, wav_audio)


def tacotron2_submit(url: str, text: str | None):
    """
    发送 put 请求
    """
    if not text:
        raise gr.Error("文本不能为空！")
    return send_put_tacotron2_tts(url, text)


# ========
# Gradio UI
with gr.Blocks(title="Coqui.ai TTS Interface Demo") as demo:
    url_text = gr.Textbox(
        label="API 地址",
        placeholder="输入后端 API 地址",
        value="http://127.0.0.1:8000",
        interactive=True,
    )

    with gr.Tab("XTTS V2"):
        with gr.Row():
            with gr.Column(variant="panel"):
                xttsv2_input_text = gr.Textbox(label="输入文本", placeholder="输入用于语音合成的文本内容。", lines=5)
                languages_dropdown = gr.Dropdown(label="语音", info="选择所需要使用的语言。", filterable=False)
                speakers_dropdown = gr.Dropdown(label="发言人（可选）", info="选择使用不同的输出声音。", filterable=False)
                speaker_wav_audio = gr.Audio(label="上传语音克隆文件（可选）")
                with gr.Row():
                    xttsv2_clear_button = gr.ClearButton(value="清空")
                    xttsv2_submit_button = gr.Button(value="提交", variant="primary")
            with gr.Column():
                xttsv2_output_audio = gr.Audio(label="语音合成结果")

    # 下拉栏事件
    languages_dropdown.focus(  # pylint: disable=E1101
        fn=xttsv2_list_languages,
        inputs=url_text,
        outputs=languages_dropdown,
        show_progress=False,
    )
    speakers_dropdown.focus(  # pylint: disable=E1101
        fn=xttsv2_list_speakers,
        inputs=url_text,
        outputs=speakers_dropdown,
        show_progress=False,
    )
    # 按钮事件
    xttsv2_clear_button.add(
        [
            xttsv2_input_text,
            languages_dropdown,
            speakers_dropdown,
            speaker_wav_audio,
            xttsv2_output_audio,
        ]
    )
    xttsv2_submit_button.click(  # pylint: disable=E1101
        fn=xttsv2_submit,
        inputs=[
            url_text,
            xttsv2_input_text,
            languages_dropdown,
            speakers_dropdown,
            speaker_wav_audio,
        ],
        outputs=xttsv2_output_audio,
    )

    # ========

    with gr.Tab("tacotron2-DDC-GST"):
        with gr.Row():
            with gr.Column(variant="panel"):
                tacotron2_input_text = gr.Textbox(label="输入文本", placeholder="输入用于语音合成的文本内容。", lines=5)
                with gr.Row():
                    tacotron2_clear_button = gr.ClearButton(value="清空")
                    tacotron2_submit_button = gr.Button(value="提交", variant="primary")
            with gr.Column():
                tacotron2_output_audio = gr.Audio(label="语音合成结果")

    tacotron2_clear_button.add(
        [
            tacotron2_input_text,
            tacotron2_output_audio,
        ]
    )
    tacotron2_submit_button.click(  # pylint: disable=E1101
        fn=tacotron2_submit,
        inputs=[url_text, tacotron2_input_text],
        outputs=tacotron2_output_audio,
    )


if __name__ == "__main__":
    demo.queue().launch(show_api=False)
