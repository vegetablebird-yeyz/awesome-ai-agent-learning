# 第9—14章配套示例

这些示例与对应章节一起阅读。Python 3.11 或更高版本最省心。

## 安装

在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r examples/requirements.txt
cp .env.example .env
```

Windows PowerShell 激活命令：

```powershell
.venv\Scripts\Activate.ps1
```

## 示例索引

| 章节 | 文件 | 是否需要网络或密钥 |
|------|------|--------------------|
| 第9章 LangChain | [`09_langchain_quickstart.py`](09_langchain_quickstart.py) | 需要兼容 OpenAI 的模型 API |
| 第10章 LangGraph | [`10_langgraph_review.py`](10_langgraph_review.py) | 不需要 |
| 第11章 Dify | [`11_dify_client.py`](11_dify_client.py) | 需要已发布的 Dify 应用 |
| 第12章 RAG | [`12_local_rag.py`](12_local_rag.py) | 不需要 |
| 第13章多 Agent | [`13_multi_agent.py`](13_multi_agent.py) | 不需要 |
| 第14章毕业项目 | [`14_capstone_research_assistant.py`](14_capstone_research_assistant.py) | 不需要 |

## 环境变量

第9章使用：

```dotenv
API_KEY=your-api-key
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4o-mini
```

第11章使用：

```dotenv
DIFY_BASE_URL=https://your-dify.example.com
DIFY_API_KEY=app-xxxxxxxx
```

不要提交 `.env`。远程 Dify 地址必须使用 HTTPS；只有 `localhost` 和 `127.0.0.1` 允许 HTTP。

## 逐个运行

```bash
python examples/09_langchain_quickstart.py
python examples/10_langgraph_review.py
python examples/11_dify_client.py
python examples/12_local_rag.py
python examples/13_multi_agent.py
python examples/14_capstone_research_assistant.py
```

后三个离线示例采用小数据与确定性逻辑，方便看清流程和编写测试。它们是教学基线，不应直接当成生产检索或安全沙箱。
