"""
运行 Gradio UI 操作页面
"""

from web import demo

if __name__ == "__main__":
    demo.queue().launch(show_api=False)
