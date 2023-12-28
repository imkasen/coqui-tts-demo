# Coqui.ai/TTS Demo

## 安装

1. 从 Pytoch [官网](https://download.pytorch.org/whl/)下载对应版本的 torch、torchvision、torchaudio 等，或者 `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`
2. 创建虚拟环境，安装依赖：`pip install -r requirements.txt`

## 模型

模型信息：[.models.json](https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json)

查看模型列表：

``` Python
tts --list_models
tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
```

### 默认下载路径

- Linux：`~/.local/share/tts`
- Windows：`C:\Users\<UserName>\AppData\Local\tts`

具体实现详见 `TTS/utils/generic_utils.py` 中 `get_user_data_dir` 方法和 `TTS/utils/manage.py` 中 `ModelManager` 的 `__init__` 方法。
