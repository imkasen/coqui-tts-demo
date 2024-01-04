"""
以 API 服务的形式运行 TTS
"""


from fastapi import FastAPI, applications
from fastapi.openapi.docs import get_swagger_ui_html
from models import xtts_v2_model, zh_tacotron2_model
from pydantic import BaseModel

# ===== FastAPI =====


# CDN
def swagger_monkey_patch(*args, **kwargs):
    """
    重写方法将 CDN 地址替换为国内网址
    """
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url="https://cdn.staticfile.org/swagger-ui/5.6.2/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.staticfile.org/swagger-ui/5.6.2/swagger-ui.css",
    )


applications.get_swagger_ui_html = swagger_monkey_patch

# 创建 FastAPI 对象，并将 swagger 文档从默认 '/docs' 改为 '/'，关闭 redoc 文档
app = FastAPI(title="TTS Server", description="A Coqui.ai TTS Server", docs_url="/", redoc_url=None)


class UploadContent(BaseModel):
    """
    tts 所需要的参数信息
    """

    text: str
    language: str | None = None
    speaker: str | None = None


# Routers
@app.get(path="/xttsv2/speakers", response_model=list[str])
async def get_xttsv2_speakers() -> list[str]:
    """
    获得 XTTS v2 模型所支持的发言者列表
    """
    return xtts_v2_model.get_speakers()


@app.get(path="/xttsv2/languages", response_model=list[str])
async def get_xttsv2_languages() -> list[str]:
    """
    获得 XTTS v2 模型所支持的语言列表
    """
    return xtts_v2_model.get_languages()


@app.put(path="/xttsv2/tts")
async def xttsv2_tts(content: UploadContent):
    """
    语音合成
    """
    return xtts_v2_model.text_to_speech(
        text=content.text,
        language=content.language,
        speaker=content.speaker,
    )


@app.put(path="/tacotron2/tts")
async def tacotron2_tts(content: UploadContent):
    """
    语音合成
    """
    return zh_tacotron2_model.text_to_speech(text=content.text)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app="api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
