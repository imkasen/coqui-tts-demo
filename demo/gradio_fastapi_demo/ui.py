"""
UI 界面
"""
import gradio as gr
from api_requests import get_xttsv2_languages, get_xttsv2_speakers, send_put_tacotron2_tts, send_put_xttsv2_tts


def list_languages() -> gr.Dropdown:
    """
    为 XTTS V2 语音下拉栏获取选项列表
    """
    list_cnt: list[str] = get_xttsv2_languages()
    return gr.Dropdown(choices=list_cnt)


def list_speakers() -> gr.Dropdown:
    """
    为 XTTS V2 发言者下拉栏获取选项列表
    """
    list_cnt: list[str] = get_xttsv2_speakers()
    return gr.Dropdown(choices=list_cnt)


# Gradio UI
with gr.Blocks(title="Coqui.ai TTS Interface Demo") as demo:
    with gr.Tab("XTTS V2"):
        with gr.Row():
            with gr.Column(variant="panel"):
                xttsv2_input_text = gr.Textbox(label="输入文本", placeholder="输入用于语音合成的文本内容。", lines=5)
                # TODO: 为空时跳出错误信息
                languages_dropdown = gr.Dropdown(label="语音", info="选择所需要使用的语言。")
                # TODO: 与 wav_audio 互斥
                speakers_dropdown = gr.Dropdown(label="发言人", info="选择使用不同的输出声音。")
                # TODO: 上传克隆文件，与 speakers 互斥
                with gr.Row():
                    xttsv2_clear_button = gr.ClearButton(value="清空")
                    xttsv2_submit_button = gr.Button(value="提交", variant="primary")
            with gr.Column():
                xttsv2_output_audio = gr.Audio(label="语音合成结果")

    # 下拉栏事件
    languages_dropdown.focus(  # pylint: disable=E1101
        fn=list_languages,
        outputs=languages_dropdown,
        show_progress=False,
    )
    speakers_dropdown.focus(  # pylint: disable=E1101
        fn=list_speakers,
        outputs=speakers_dropdown,
        show_progress=False,
    )
    # 按钮事件
    xttsv2_clear_button.add(
        [
            xttsv2_input_text,
            languages_dropdown,
            speakers_dropdown,
            xttsv2_output_audio,
        ]
    )
    xttsv2_submit_button.click(  # pylint: disable=E1101
        fn=send_put_xttsv2_tts,
        inputs=[
            xttsv2_input_text,
            languages_dropdown,
            speakers_dropdown,
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
        fn=send_put_tacotron2_tts,
        inputs=tacotron2_input_text,
        outputs=tacotron2_output_audio,
    )


if __name__ == "__main__":
    demo.queue().launch(show_api=False)
