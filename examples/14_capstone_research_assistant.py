"""第14章毕业项目：分层、可观测、可测试、可恢复的离线研究助手。

运行演示：
    python examples/14_capstone_research_assistant.py

运行自检：
    python examples/14_capstone_research_assistant.py --self-test

程序只使用标准库。演示运行目录放在系统临时目录，退出后自动清理，
不会在仓库中留下日志或检查点。真实服务应把 RuntimePaths 指向持久卷。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


MAX_FILE_BYTES = 200_000
MAX_FILES = 50
MAX_TOTAL_BYTES = 2_000_000
MAX_QUESTION_CHARS = 500
ALLOWED_SUFFIXES = {".md", ".txt"}
REFUSAL = "资料不足：本地知识库中没有找到足够证据。"
SECRET_KEYS = {"api_key", "authorization", "password", "token", "secret"}


class RunStatus(str, Enum):
    CREATED = "created"
    PLANNED = "planned"
    RETRIEVED = "retrieved"
    DRAFTED = "drafted"
    APPROVED = "approved"
    REFUSED = "refused"
    FAILED = "failed"


@dataclass(frozen=True)
class Document:
    source: str
    content: str
    sha256: str


@dataclass(frozen=True)
class Evidence:
    source: str
    excerpt: str
    score: float


@dataclass
class ResearchState:
    run_id: str
    question: str
    status: RunStatus = RunStatus.CREATED
    subquestions: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    answer: str = ""
    review_issues: list[str] = field(default_factory=list)
    error: str = ""
    step: int = 0
    started_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"


class DocumentRepository(Protocol):
    def load(self) -> list[Document]:
        """读取允许的文档。"""


class SearchService(Protocol):
    def search(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int = 4,
    ) -> list[Evidence]:
        """检索证据。"""


class FileDocumentRepository:
    """基础设施层：只负责安全读取，不负责检索和生成。"""

    def __init__(
        self,
        root: Path,
        max_file_bytes: int = MAX_FILE_BYTES,
        max_files: int = MAX_FILES,
        max_total_bytes: int = MAX_TOTAL_BYTES,
    ) -> None:
        self.root = root.resolve()
        self.max_file_bytes = max_file_bytes
        self.max_files = max_files
        self.max_total_bytes = max_total_bytes

    def _is_inside_root(self, path: Path) -> bool:
        try:
            path.relative_to(self.root)
            return True
        except ValueError:
            return False

    def load(self) -> list[Document]:
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"资料目录不存在：{self.root}")

        documents: list[Document] = []
        total_bytes = 0
        for candidate in sorted(self.root.iterdir()):
            if len(documents) >= self.max_files:
                break
            resolved = candidate.resolve()
            if not self._is_inside_root(resolved):
                continue
            if resolved.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            if not resolved.is_file() or resolved.is_symlink():
                continue
            size = resolved.stat().st_size
            if size > self.max_file_bytes:
                continue
            if total_bytes + size > self.max_total_bytes:
                break
            try:
                raw = resolved.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            total_bytes += size
            clean = normalize_text(raw)
            digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
            documents.append(Document(resolved.name, clean, digest))
        return documents


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def terms(text: str) -> set[str]:
    english = re.findall(r"[a-z0-9_]{2,}", text.lower())
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    single = list(chinese)
    bigrams = [chinese[index : index + 2] for index in range(len(chinese) - 1)]
    return set(english + single + bigrams)


class LocalSearchService:
    """领域层：可替换为向量数据库，但返回 Evidence 契约不变。"""

    INJECTION_PATTERNS = (
        r"忽略.{0,8}(系统|之前|以上).{0,8}(指令|规则)",
        r"ignore previous",
        r"(输出|泄露).{0,8}(密钥|密码|token)",
        r"(执行|运行).{0,8}(shell|命令)",
    )

    @classmethod
    def suspicious(cls, text: str) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in cls.INJECTION_PATTERNS)

    def search(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int = 4,
    ) -> list[Evidence]:
        query_terms = terms(query)
        if not query_terms:
            return []
        hits: list[Evidence] = []
        for document in documents:
            paragraphs = re.split(r"\n\s*\n", document.content)
            for number, paragraph in enumerate(paragraphs, 1):
                paragraph = paragraph.strip()
                if not paragraph or self.suspicious(paragraph):
                    continue
                overlap = query_terms & terms(paragraph)
                score = len(overlap) / len(query_terms)
                if score >= 0.08:
                    hits.append(
                        Evidence(
                            source=f"{document.source}#p{number}",
                            excerpt=paragraph[:500],
                            score=score,
                        )
                    )
        return sorted(hits, key=lambda item: (-item.score, item.source))[:top_k]


class Planner:
    """应用角色：把问题拆成少量可检索子问题。"""

    def plan(self, question: str) -> list[str]:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("问题不能为空")
        if len(cleaned) > MAX_QUESTION_CHARS:
            raise ValueError(f"问题不能超过 {MAX_QUESTION_CHARS} 个字符")
        result = [cleaned]
        if "风险" in cleaned or "安全" in cleaned:
            result.extend(["工具参数怎样校验？", "哪些动作需要人工确认？"])
        return result[:3]


class Writer:
    """应用角色：抽取式写作，保证每条内容附近都有来源。"""

    def draft(self, question: str, evidence: Sequence[Evidence]) -> str:
        if not evidence:
            return REFUSAL
        lines = [f"# 研究答复", "", f"问题：{question}", "", "## 可核验结论"]
        for item in evidence:
            lines.append(f"- {item.excerpt} [{item.source}]")
        lines.extend(
            [
                "",
                "## 使用边界",
                "以上内容仅依据已列出的本地证据；证据之外不作推断。",
                "涉及写入、删除、发送、支付等高风险动作时，必须再次人工确认。",
            ]
        )
        return "\n".join(lines)


class Reviewer:
    """确定性审查层：LLM 审查可以增加，但不能替代硬校验。"""

    CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_.-]+#p\d+)\]")

    def review(self, answer: str, evidence: Sequence[Evidence]) -> list[str]:
        if answer == REFUSAL:
            return []
        issues: list[str] = []
        allowed = {item.source for item in evidence}
        citations = set(self.CITATION_PATTERN.findall(answer))
        if not citations:
            issues.append("回答缺少引用")
        if citations - allowed:
            issues.append(f"存在未知引用：{sorted(citations - allowed)}")
        missing = allowed - citations
        if missing:
            issues.append(f"召回证据未在答案中引用：{sorted(missing)}")
        if "证据之外不作推断" not in answer:
            issues.append("回答缺少证据边界")
        if "人工确认" not in answer:
            issues.append("回答缺少高风险动作确认边界")
        return issues


def redact(value: Any) -> Any:
    """递归脱敏，防止令牌和密码进入日志。"""
    if isinstance(value, Mapping):
        return {
            str(key): "***REDACTED***" if str(key).lower() in SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(
            r"(?i)(api[_-]?key|token|password)\s*[:=]\s*\S+",
            r"\1=***REDACTED***",
            value,
        )
    return value


class JsonlLogger:
    """基础设施层：一行一个事件，崩溃时最多损失最后一行。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, run_id: str, **fields: Any) -> None:
        record = {
            "timestamp": round(time.time(), 3),
            "event": event,
            "run_id": run_id,
            **redact(fields),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # 忽略崩溃时可能留下的不完整尾行。
                continue
        return records


class CheckpointStore:
    """每一步原子保存；恢复时从最后一个完整检查点继续。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        if not re.fullmatch(r"run-[0-9a-f]{12}", run_id):
            raise ValueError("非法 run_id")
        return self.root / f"{run_id}.json"

    def save(self, state: ResearchState) -> None:
        path = self._path(state.run_id)
        payload = asdict(state)
        payload["status"] = state.status.value
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def load(self, run_id: str) -> ResearchState:
        payload = json.loads(self._path(run_id).read_text(encoding="utf-8"))
        payload["status"] = RunStatus(payload["status"])
        payload["evidence"] = [Evidence(**item) for item in payload["evidence"]]
        return ResearchState(**payload)


class SimulatedCrash(RuntimeError):
    """只用于展示检查点恢复。"""


class ResearchApplication:
    """应用服务层：编排步骤，不包含文件解析或检索细节。"""

    TERMINAL = {RunStatus.APPROVED, RunStatus.REFUSED, RunStatus.FAILED}

    def __init__(
        self,
        repository: DocumentRepository,
        search: SearchService,
        logger: JsonlLogger,
        checkpoints: CheckpointStore,
    ) -> None:
        self.repository = repository
        self.search = search
        self.logger = logger
        self.checkpoints = checkpoints
        self.planner = Planner()
        self.writer = Writer()
        self.reviewer = Reviewer()

    @staticmethod
    def new_run_id(question: str) -> str:
        seed = f"{question}|{time.time_ns()}"
        return "run-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]

    def persist(self, state: ResearchState, event: str, **fields: Any) -> None:
        state.step += 1
        self.checkpoints.save(state)
        self.logger.write(
            event,
            state.run_id,
            status=state.status.value,
            step=state.step,
            **fields,
        )

    def create(self, question: str) -> ResearchState:
        question = question.strip()
        if not question or len(question) > MAX_QUESTION_CHARS:
            raise ValueError("问题为空或过长")
        state = ResearchState(run_id=self.new_run_id(question), question=question)
        self.persist(state, "run.created")
        return state

    def execute(
        self,
        state: ResearchState,
        crash_after: RunStatus | None = None,
    ) -> ResearchState:
        """幂等推进状态；已完成步骤不会重复执行。"""
        if state.status in self.TERMINAL:
            return state
        try:
            documents: list[Document] | None = None

            if state.status == RunStatus.CREATED:
                state.subquestions = self.planner.plan(state.question)
                state.status = RunStatus.PLANNED
                self.persist(state, "plan.completed", count=len(state.subquestions))
                if crash_after == state.status:
                    raise SimulatedCrash("模拟：规划后进程退出")

            if state.status == RunStatus.PLANNED:
                documents = self.repository.load()
                unique: dict[str, Evidence] = {}
                for subquestion in state.subquestions:
                    for item in self.search.search(subquestion, documents):
                        current = unique.get(item.source)
                        if current is None or item.score > current.score:
                            unique[item.source] = item
                state.evidence = sorted(
                    unique.values(),
                    key=lambda item: (-item.score, item.source),
                )[:5]
                state.status = RunStatus.RETRIEVED
                self.persist(
                    state,
                    "retrieval.completed",
                    document_count=len(documents),
                    evidence_count=len(state.evidence),
                    sources=[item.source for item in state.evidence],
                )
                if crash_after == state.status:
                    raise SimulatedCrash("模拟：检索后进程退出")

            if state.status == RunStatus.RETRIEVED:
                state.answer = self.writer.draft(state.question, state.evidence)
                state.status = (
                    RunStatus.REFUSED if state.answer == REFUSAL else RunStatus.DRAFTED
                )
                self.persist(state, "draft.completed", refused=state.status == RunStatus.REFUSED)
                if crash_after == state.status:
                    raise SimulatedCrash("模拟：起草后进程退出")

            if state.status == RunStatus.DRAFTED:
                state.review_issues = self.reviewer.review(state.answer, state.evidence)
                if state.review_issues:
                    state.status = RunStatus.FAILED
                    state.error = "；".join(state.review_issues)
                else:
                    state.status = RunStatus.APPROVED
                self.persist(
                    state,
                    "review.completed",
                    issue_count=len(state.review_issues),
                )
            return state
        except SimulatedCrash:
            # 模拟真实崩溃：不覆盖最后一个成功检查点。
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            state.status = RunStatus.FAILED
            state.error = f"{type(exc).__name__}: {exc}"
            self.persist(state, "run.failed", error=state.error)
            return state

    def start(
        self,
        question: str,
        crash_after: RunStatus | None = None,
    ) -> ResearchState:
        return self.execute(self.create(question), crash_after=crash_after)

    def resume(self, run_id: str) -> ResearchState:
        state = self.checkpoints.load(run_id)
        self.logger.write(
            "run.resumed",
            run_id,
            status=state.status.value,
            step=state.step,
        )
        return self.execute(state)


def build_application(data_dir: Path, runtime_dir: Path) -> ResearchApplication:
    paths = RuntimePaths(runtime_dir)
    return ResearchApplication(
        repository=FileDocumentRepository(data_dir),
        search=LocalSearchService(),
        logger=JsonlLogger(paths.logs),
        checkpoints=CheckpointStore(paths.checkpoints),
    )


def print_result(state: ResearchState) -> None:
    print(f"run_id={state.run_id}")
    print(f"status={state.status.value} step={state.step}")
    print(state.answer or "(没有生成回答)")
    if state.review_issues:
        print("审查问题：", state.review_issues)
    if state.error:
        print("错误：", state.error)


class FakeRepository:
    """测试替身：单元测试无需访问真实文件系统。"""

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)
        self.calls = 0

    def load(self) -> list[Document]:
        self.calls += 1
        return list(self.documents)


def make_test_document(text: str, source: str = "test.md") -> Document:
    return Document(
        source=source,
        content=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="capstone-test-") as directory:
        root = Path(directory)
        runtime = root / "runtime"
        data = root / "data"
        data.mkdir()
        (data / "safe.md").write_text(
            "工具调用前要校验名称、参数类型和范围。\n\n"
            "删除、发送和支付必须等待人工确认。",
            encoding="utf-8",
        )
        (data / "ignored.bin").write_bytes(b"\x00\x01")
        repository = FileDocumentRepository(data)
        documents = repository.load()
        assert [item.source for item in documents] == ["safe.md"]

        app = build_application(data, runtime)
        normal = app.start("怎样降低工具调用风险？")
        assert normal.status == RunStatus.APPROVED
        assert "[safe.md#p" in normal.answer
        assert not normal.review_issues

        unknown = app.start("木星咖啡店几点营业？")
        assert unknown.status == RunStatus.REFUSED
        assert unknown.answer == REFUSAL

        try:
            interrupted = app.create("工具调用怎样校验？")
            run_id = interrupted.run_id
            app.execute(interrupted, crash_after=RunStatus.RETRIEVED)
        except SimulatedCrash:
            recovered = app.resume(run_id)
        else:
            raise AssertionError("应触发模拟崩溃")
        assert recovered.status == RunStatus.APPROVED

        app.logger.write(
            "security.test",
            normal.run_id,
            api_key="should-not-appear",
            nested={"password": "should-not-appear"},
        )
        raw_log = app.logger.path.read_text(encoding="utf-8")
        assert "should-not-appear" not in raw_log
        assert "***REDACTED***" in raw_log

        reviewer = Reviewer()
        issues = reviewer.review(
            "没有引用但声称安全，证据之外不作推断，需要人工确认。",
            [Evidence("safe.md#p1", "证据", 1.0)],
        )
        assert "回答缺少引用" in issues
    print("SELF-TEST: 通过（分层、边界、拒答、引用、日志脱敏、崩溃恢复）")


def run_demo(data_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="capstone-demo-") as directory:
        runtime = Path(directory)
        app = build_application(data_dir, runtime)
        print("第一次执行：在检索后模拟崩溃")
        state = app.create("怎样降低 Agent 工具调用的风险？")
        try:
            app.execute(state, crash_after=RunStatus.RETRIEVED)
        except SimulatedCrash as exc:
            print(str(exc))

        print("\n从检查点恢复：")
        recovered = app.resume(state.run_id)
        print_result(recovered)

        events = app.logger.read_all()
        print(f"\n审计事件数：{len(events)}")
        for event in events:
            print(
                f"- step={event.get('step', '-')} "
                f"event={event['event']} status={event.get('status', '-')}"
            )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第14章毕业项目")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).with_name("capstone_data"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        self_test()
    else:
        run_demo(args.data_dir)


if __name__ == "__main__":
    main()
