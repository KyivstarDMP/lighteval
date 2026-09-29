# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Model client for a vLLM OpenAI-compatible server (``vllm serve``).

Unlike the LiteLLM backend, this client speaks to exactly one known server
kind, so it uses the ``openai`` SDK directly (no provider routing, no name
prefixes) and leans on vLLM-specific API extensions:

- generative requests forward sampling parameters LiteLLM drops
  (``top_k``, ``min_p``, ``repetition_penalty``) and ``chat_template_kwargs``;
- chat-templated loglikelihood (text and vision) scores each choice as an
  unterminated assistant reply via ``prompt_logprobs`` +
  ``continue_final_message``;
- plain-text loglikelihood and rolling perplexity use ``/v1/completions``
  with ``echo=True``.

Retry semantics are local-server-appropriate: a connection failure means the
server died (fail fast and abort), while overload (429/5xx/timeouts) backs off
and, when exhausted, degrades that one request.
"""

import asyncio
import contextlib
import logging
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import ClassVar, NamedTuple

import httpx
import openai
from openai import NOT_GIVEN, AsyncOpenAI
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset, LoglikelihoodDataset
from lighteval.models.abstract_model import LightevalModel, ModelConfig
from lighteval.models.endpoints.openai_scoring import (
    align_continuation_suffix,
    check_argmax,
    extract_prompt_logprob_entries,
    find_continuation_start,
    pieces_to_text,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.prompt_manager import PromptManager, image_url_parts
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached


logger = logging.getLogger(__name__)

_CONNECT_RETRIES = 2
_CONNECT_RETRY_SLEEP_S = 1.0
# Consecutive stalled generations (across requests) before the server is declared dead.
_STALL_ABORT_AFTER = 8
_DEFAULT_MAX_LENGTH = 4096
_DEFAULT_STALL_S = 600.0  # the stall bound when the config leaves `timeout` unset

# Adaptive concurrency: how the in-flight limit follows the server's load.
_LOAD_INTERVAL_S = 2.0
_START_CONCURRENCY = 64
_KV_GROW_BELOW = 0.85  # grow only while the KV cache has room; between this and _KV_SHED_AT, hold
_KV_SHED_AT = 0.95
_SHED_TO = 0.9  # shed to this fraction of what the server is running
_SATURATED = 0.9  # "all" with slack: the client fills its limit, the server runs what is in flight


class _ProgressClock:
    """When any stream last received a chunk: a request still queued behind others stalls only if this went quiet."""

    def __init__(self) -> None:
        self.last = time.monotonic()

    def tick(self) -> None:
        self.last = time.monotonic()


class _Quiet(Exception):
    pass


class _StreamStall(openai.APITimeoutError):
    """A request that stopped making progress; handled like the SDK's own timeout."""

    def __init__(self, detail: str, request: httpx.Request) -> None:
        openai.APIConnectionError.__init__(self, message=detail, request=request)


async def _await_unless_quiet(awaitable, *, quiet_for, stall_s: float):
    """Await ``awaitable``, raising ``_Quiet`` once ``quiet_for()`` reaches ``stall_s`` seconds before it finishes."""
    task = asyncio.ensure_future(awaitable)
    poll = min(15.0, stall_s / 4)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll)
            if done:
                return task.result()
            if quiet_for() >= stall_s:
                raise _Quiet
    finally:
        if not task.done():
            task.cancel()


async def _stream_chunks(stream, *, progress: _ProgressClock, stall_s: float):
    """Yield ``stream``'s chunks, raising ``_StreamStall`` when it stalls.

    Before its first chunk a request is queued behind others, not stuck: it stalls only when no stream anywhere has
    received a chunk for ``stall_s``. Once started, the gap between its own chunks is bounded by ``stall_s``.
    """
    iterator = stream.__aiter__()
    opened = last_chunk = time.monotonic()
    started = False

    def quiet_for() -> float:
        return time.monotonic() - (last_chunk if started else max(progress.last, opened))

    try:
        while True:
            try:
                chunk = await _await_unless_quiet(iterator.__anext__(), quiet_for=quiet_for, stall_s=stall_s)
            except StopAsyncIteration:
                return
            except _Quiet:
                source = "this stream" if started else "any stream"
                request = getattr(getattr(stream, "response", None), "request", None)
                raise _StreamStall(
                    f"no chunk from {source} for {stall_s:g}s", request or httpx.Request("POST", "http://vllm/stream")
                ) from None
            started = True
            last_chunk = time.monotonic()
            progress.tick()
            yield chunk
    finally:
        closer = getattr(stream, "close", None) or getattr(stream, "aclose", None)
        if closer is not None:
            with contextlib.suppress(Exception):
                await closer()


class _ServerLoad(NamedTuple):
    running: float
    waiting: float
    kv_usage: float
    preemptions: float  # cumulative


_LOAD_SERIES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
    "vllm:kv_cache_usage_perc": "kv_usage",
    "vllm:num_preemptions_total": "preemptions",
}
_SERIES_LINE = re.compile(r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)")


def _metrics_url(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1") + "/metrics"


def _parse_server_load(text: str) -> _ServerLoad | None:
    """Read the series the limiter steers by from a Prometheus exposition; ``None`` if the server lacks any of them.

    Several engines (data parallel) add up, except the KV usage, where the fullest one counts.
    """
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = _SERIES_LINE.match(line)
        if match is None or (field := _LOAD_SERIES.get(match["name"])) is None:
            continue
        try:
            value = float(match["value"])
        except ValueError:
            continue
        totals[field] = max(totals.get(field, 0.0), value) if field == "kv_usage" else totals.get(field, 0.0) + value
    return _ServerLoad(**totals) if len(totals) == len(_LOAD_SERIES) else None


def _next_limit(limit: int, *, ceiling: int, in_flight: int, load: _ServerLoad, preempted: float) -> int:
    """One control step.

    Shed below what the server is running once its KV cache is exhausted or it just preempted, by at most half, and
    not again until the last shed has taken effect (requests already sent keep running, and keep the server
    preempting, until they finish). Grow only while the client fills its limit and the server is running what the
    client has in flight, with KV to spare: requests the server has not scheduled yet are invisible to its metrics,
    and a limiter cannot recall what it has sent. Otherwise hold.
    """
    if in_flight > limit:
        return limit
    if load.kv_usage >= _KV_SHED_AT or preempted > 0:
        return max(1, min(limit, max(int(load.running * _SHED_TO), limit // 2)))
    saturating = in_flight >= _SATURATED * limit
    seen = load.waiting == 0 and load.running >= _SATURATED * in_flight
    if saturating and seen and load.kv_usage < _KV_GROW_BELOW:
        return min(ceiling, limit + max(1, limit // 2))
    return limit


class _AdaptiveLimiter:
    """An in-flight gate with a movable limit, used like a semaphore (``async with``); waiters are served in order.

    Lowering the limit evicts nothing: requests already in flight finish, and new ones wait until there is room.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_flight = 0
        self._waiters: deque[asyncio.Future] = deque()

    async def __aenter__(self) -> None:
        if self.in_flight < self.limit and not self._waiters:
            self.in_flight += 1
            return
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        try:
            await waiter  # `_wake` counts the slot before resolving it
        except BaseException:
            if waiter.done() and not waiter.cancelled():
                self.in_flight -= 1  # granted a slot but cancelled before using it
                self._wake()
            else:
                with contextlib.suppress(ValueError):
                    self._waiters.remove(waiter)
            raise

    async def __aexit__(self, *exc_info) -> None:
        self.in_flight -= 1
        self._wake()

    def set_limit(self, limit: int) -> None:
        self.limit = limit
        self._wake()

    def _wake(self) -> None:
        while self._waiters and self.in_flight < self.limit:
            waiter = self._waiters.popleft()
            if not waiter.done():
                self.in_flight += 1
                waiter.set_result(None)


async def _steer(
    limiter: _AdaptiveLimiter,
    *,
    ceiling: int,
    fetch: Callable[[], Awaitable[str | None]],
    interval: float = _LOAD_INTERVAL_S,
) -> None:
    """Move ``limiter`` with the server's load until cancelled.

    A server that has never reported its load is driven at the ceiling; one that did and now misses a reading (a
    fetch that timed out under load, say) keeps the current limit.
    """
    last_preemptions: float | None = None
    warned = False
    while True:
        await asyncio.sleep(interval)
        text = await fetch()
        load = _parse_server_load(text) if text else None
        if load is None:
            if last_preemptions is not None:
                continue
            if not warned:
                logger.warning(f"Adaptive concurrency: the server reports no load; sending up to {ceiling} in flight.")
                warned = True
            limiter.set_limit(ceiling)
            continue
        preempted = 0.0 if last_preemptions is None else max(0.0, load.preemptions - last_preemptions)
        last_preemptions = load.preemptions
        limit = _next_limit(
            limiter.limit, ceiling=ceiling, in_flight=limiter.in_flight, load=load, preempted=preempted
        )
        if limit != limiter.limit:
            logger.info(
                f"Adaptive concurrency {limiter.limit} -> {limit} (in flight={limiter.in_flight}; server "
                f"running={load.running:.0f} waiting={load.waiting:.0f} kv={load.kv_usage:.0%} preempted={preempted:.0f})"
            )
            limiter.set_limit(limit)


async def _collect_chat_stream(
    stream, num_samples: int, *, progress: _ProgressClock | None = None, stall_s: float = _DEFAULT_STALL_S
):
    """Fold a chat-completion stream into the shape ``greedy_until`` reads (``choices[i].message``).

    Streaming is what makes the client timeout a stall bound rather than a
    generation-length bound. Chunks carry ``choice.index`` for ``n`` samples;
    vLLM's parsed reasoning arrives as ``delta.reasoning_content``.
    """
    content: list[list[str]] = [[] for _ in range(num_samples)]
    reasoning: list[list[str]] = [[] for _ in range(num_samples)]
    finish: list[str | None] = [None] * num_samples
    chunks = stream if progress is None else _stream_chunks(stream, progress=progress, stall_s=stall_s)
    async for chunk in chunks:
        for choice in getattr(chunk, "choices", None) or []:
            index = getattr(choice, "index", 0) or 0
            while index >= len(content):
                content.append([]), reasoning.append([]), finish.append(None)
            delta = getattr(choice, "delta", None)
            if delta is not None:
                if piece := getattr(delta, "content", None):
                    content[index].append(piece)
                # vLLM names the parsed channel `reasoning` (0.19+); older servers said `reasoning_content`.
                if thought := getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None):
                    reasoning[index].append(thought)
            if reason := getattr(choice, "finish_reason", None):
                finish[index] = reason
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=i,
                finish_reason=finish[i],
                message=SimpleNamespace(content="".join(content[i]), reasoning_content="".join(reasoning[i]) or None),
            )
            for i in range(len(content))
        ]
    )


async def _collect_text_stream(
    stream, num_samples: int, *, progress: _ProgressClock | None = None, stall_s: float = _DEFAULT_STALL_S
):
    """Fold a text-completion stream into ``choices[i].text``."""
    text: list[list[str]] = [[] for _ in range(num_samples)]
    finish: list[str | None] = [None] * num_samples
    chunks = stream if progress is None else _stream_chunks(stream, progress=progress, stall_s=stall_s)
    async for chunk in chunks:
        for choice in getattr(chunk, "choices", None) or []:
            index = getattr(choice, "index", 0) or 0
            while index >= len(text):
                text.append([]), finish.append(None)
            if piece := getattr(choice, "text", None):
                text[index].append(piece)
            if reason := getattr(choice, "finish_reason", None):
                finish[index] = reason
    return SimpleNamespace(
        choices=[SimpleNamespace(index=i, finish_reason=finish[i], text="".join(text[i])) for i in range(len(text))]
    )


def _fit_open_files(connections: int) -> None:
    """Raise the soft open-files limit to fit ``connections`` sockets: containers commonly start it at 1024."""
    try:
        import resource
    except ImportError:  # no such limit on Windows
        return
    wanted = connections + 1024  # headroom for the process's own files
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= wanted:
        return
    target = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    if target < wanted:
        logger.warning(
            f"The open-files limit ({hard}) cannot fit {connections} connections: "
            f"requests past ~{target - 1024} in flight will fail to connect"
        )


class VLLMOpenAIModelConfig(ModelConfig):
    """Configuration for a vLLM OpenAI-compatible server client.

    Attributes:
        model_name (str):
            The served model name, exactly as the server knows it (no provider
            prefix) — what ``vllm serve <model>`` was started with.
        base_url (str):
            The server's OpenAI-compatible root, e.g. ``http://localhost:8000/v1``.
            Required: this backend always talks to an explicit server.
        api_key (str | None):
            Bearer token, only if the server enforces one.
        use_chat_template (bool):
            ``True`` (default) sends chat messages to ``/chat/completions`` and
            scores loglikelihood through the chat template. ``False`` is for
            base / completion-only models: plain-text prompts against
            ``/v1/completions`` for both generation and loglikelihood.
        concurrent_requests (int):
            Client-side request concurrency. Default 10.
        adaptive_concurrency (bool):
            Off by default. When on, ``concurrent_requests`` is only a ceiling:
            generation starts at 64 in flight and follows the server's own load
            (``/metrics``) — it grows while the server takes everything offered
            with KV cache to spare, and sheds below what the server is running
            once its KV cache is exhausted or it preempts, so a KV-bound model
            is not over-admitted. A server without ``/metrics`` is driven at
            the ceiling. Loglikelihood requests keep the static concurrency.
        timeout (float | None):
            Stall bound in seconds (600 when unset). Generations are streamed,
            so it is never the length of a generation: a request that has
            started stalls when its own chunks stop for this long, while one
            still queued behind others stalls only when no stream at all has
            received a chunk for this long (a busy server is not a dead one,
            however long its queue). A stalled request is not retried;
            ``_STALL_ABORT_AFTER`` consecutive stalls abort the evaluation.
            Loglikelihood requests are not streamed; for them it is the whole
            request, which is short.
        max_model_length (int | None):
            The invocation's context budget, held per request whatever window
            the server was started with: generation prompts are left-truncated
            by the server (``truncate_prompt_tokens``) and log-likelihood token
            ids by the client, keeping the last tokens that leave room for the
            answer — the in-process backend's rule. Docs with images are never
            truncated. When unset, nothing is truncated and ``max_length`` is
            probed once from ``/models``.
        client_tokenization (bool):
            ``True`` (default) scores loglikelihood from client-side token ids:
            the context is rendered locally (chat template with a generation
            prompt when ``use_chat_template``, plain text otherwise), context
            and choices are split into ids exactly like the in-process backend
            (``tok_encode_pair``), and the ids go to ``/v1/completions`` with
            ``echo=True`` — the continuation is the last ``len(ids)`` positions,
            no text alignment. ``False`` keeps the server-templated
            ``prompt_logprobs`` chat route / the text echo route. Docs with
            images always take the server-templated chat route.
        tokenizer (str | None):
            Tokenizer to load for client tokenization when it differs from
            ``model_name``.
        revision (str | None):
            Model repo revision (a commit SHA pins it) the client tokenizer
            loads, matching the server's ``--revision``. Applies to
            ``model_name``; ignored when ``tokenizer`` names another repo.
        trust_remote_code (bool):
            Let the client tokenizer run code from the model repo; default
            ``False``. Only for repos whose tokenizer is not a transformers class.
        add_special_tokens (bool | None):
            BOS handling for client tokenization and for the plain-text prompts
            the server tokenizes on generation. ``None`` (default) adds them
            for plain-text prompts and not for chat-templated ones, which carry
            their own.
        pairwise_tokenization (bool):
            Tokenize context and continuation separately instead of jointly
            (``tok_encode_pair``); default ``False``, as in-process.
        api_max_retry / api_retry_sleep / api_retry_multiplier:
            Backoff schedule for transient failures (429/5xx).

    Operational fields (connection/retry knobs) are excluded from the sample
    cache key via ``CACHE_KEY_EXCLUDE`` — notably ``base_url``: the serving
    pipeline allocates a fresh port per run, which must not invalidate cached
    samples.
    """

    model_name: str
    base_url: str
    api_key: str | None = None
    use_chat_template: bool = True
    concurrent_requests: int = 10
    adaptive_concurrency: bool = False
    timeout: float | None = None
    max_model_length: int | None = None

    api_max_retry: int = 5
    api_retry_sleep: float = 1.0
    api_retry_multiplier: float = 2.0

    client_tokenization: bool = True
    tokenizer: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    add_special_tokens: bool | None = None
    pairwise_tokenization: bool = False

    CACHE_KEY_EXCLUDE: ClassVar[frozenset[str]] = frozenset(
        {
            "base_url",
            "api_key",
            "tokenizer",
            "trust_remote_code",
            "concurrent_requests",
            "adaptive_concurrency",
            "timeout",
            "api_max_retry",
            "api_retry_sleep",
            "api_retry_multiplier",
        }
    )


class VLLMOpenAIClient(LightevalModel):
    _consecutive_stalls = 0  # reset by any completed request
    _progress: _ProgressClock | None = None
    adaptive_concurrency = False
    # Client-side tokenization state (class defaults so tests can build instances without __init__).
    client_tokenization = True
    pairwise_tokenization = False
    _tokenizer = None
    _tokenizer_id: str | None = None
    _trust_remote_code = False
    _context_budget: int | None = None
    _add_special_tokens: bool | None = None

    def __init__(self, config: VLLMOpenAIModelConfig) -> None:
        self.config = config
        self.model = config.model_name
        self.base_url = config.base_url
        self.api_key = config.api_key
        self.generation_parameters = config.generation_parameters
        self.concurrent_requests = config.concurrent_requests
        self.adaptive_concurrency = config.adaptive_concurrency
        self.timeout = config.timeout
        self._max_length = config.max_model_length
        self._context_budget = config.max_model_length
        self.client_tokenization = config.client_tokenization
        self.pairwise_tokenization = config.pairwise_tokenization
        self._tokenizer_id = config.tokenizer
        self._revision = None if config.tokenizer else config.revision
        self._trust_remote_code = config.trust_remote_code
        self._add_special_tokens = config.add_special_tokens

        self.API_MAX_RETRY = config.api_max_retry
        self.API_RETRY_SLEEP = config.api_retry_sleep
        self.API_RETRY_MULTIPLIER = config.api_retry_multiplier

        self.prompt_manager = PromptManager(
            use_chat_template=config.use_chat_template,
            tokenizer=None,
            system_prompt=config.system_prompt,
            chat_template_kwargs=config.chat_template_kwargs,
            image_placement=config.image_placement,
        )

        self._cache = SampleCache(config)

    # ------------------------------------------------------------------
    # Client / request plumbing
    # ------------------------------------------------------------------

    def _make_client(self) -> AsyncOpenAI:
        """One SDK client per event-loop scope (each public method runs its own loop).

        Its connection pool holds ``concurrent_requests``: past the SDK's default of 1000, requests queue inside
        httpx, whose pool handling grows costlier with the queue until it starves the event loop.
        """
        _fit_open_files(self.concurrent_requests)
        limits = httpx.Limits(
            max_connections=self.concurrent_requests, max_keepalive_connections=self.concurrent_requests
        )
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "EMPTY",
            timeout=self.timeout if self.timeout is not None else NOT_GIVEN,
            max_retries=0,  # retry classification is ours, not the SDK's
            http_client=openai.DefaultAsyncHttpxClient(limits=limits),
        )

    async def _fetch_load(self) -> str | None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                return (await http.get(_metrics_url(self.base_url))).text
        except httpx.HTTPError:
            return None

    @contextlib.asynccontextmanager
    async def _gate(self):
        """The in-flight gate for one generation run.

        Yields:
            A semaphore, or a limiter steered by the server's load when ``adaptive_concurrency`` is on.
        """
        if not self.adaptive_concurrency:
            yield asyncio.Semaphore(self.concurrent_requests)
            return
        limiter = _AdaptiveLimiter(min(self.concurrent_requests, _START_CONCURRENCY))
        steering = asyncio.create_task(_steer(limiter, ceiling=self.concurrent_requests, fetch=self._fetch_load))
        try:
            yield limiter
        finally:
            steering.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await steering

    def _clock(self) -> _ProgressClock:
        if self._progress is None:
            self._progress = _ProgressClock()
        return self._progress

    def _stall_s(self) -> float:
        return self.timeout if self.timeout is not None else _DEFAULT_STALL_S

    def _stream_timeout(self) -> httpx.Timeout:
        # No read timeout: `_stream_chunks` tells a queued request from a stuck one, which a per-read timer cannot.
        return httpx.Timeout(self._stall_s(), read=None)

    async def _open_stream(self, create):
        """Await the SDK call that opens a stream under the same queued-versus-stuck rule as its chunks."""
        opened = time.monotonic()
        progress = self._clock()
        try:
            return await _await_unless_quiet(
                create, quiet_for=lambda: time.monotonic() - max(progress.last, opened), stall_s=self._stall_s()
            )
        except _Quiet:
            raise _StreamStall(
                f"no chunk from any stream for {self._stall_s():g}s while opening the request",
                httpx.Request("POST", f"{self.base_url}/completions"),
            ) from None

    async def _request(self, call, *, label: str):
        """Run ``await call()`` with local-server retry semantics.

        - Connection failures (server process gone): a couple of quick retries,
          then **raise** — the serving pipeline is responsible for the server
          being up, and every subsequent request would fail the same way.
        - Timeouts (a stalled stream, see ``_stream_chunks``): no retry — a
          resample of the same prompt stalls the same way — degrade this one
          request to ``None``; after ``_STALL_ABORT_AFTER`` consecutive stalls
          the server is dead, **raise**.
        - 429 / 5xx (transient overload), and an error the server reports
          inside a stream (a generation the server could not parse, say):
          exponential backoff up to ``api_max_retry``, then degrade this one
          request to ``None``.
        - Other 4xx (bad request, e.g. context overflow): no retry, degrade to
          ``None`` so one oversized doc doesn't abort a long run.
        """
        connect_failures = 0
        for attempt in range(self._retry_upper_bound()):
            try:
                result = await call()
                self._consecutive_stalls = 0
                return result
            except (openai.APITimeoutError, httpx.TimeoutException) as e:
                self._consecutive_stalls += 1
                if self._consecutive_stalls >= _STALL_ABORT_AFTER:
                    raise RuntimeError(
                        f"vLLM server at {self.base_url} stalled on {self._consecutive_stalls} consecutive "
                        f"requests ({e}); aborting the evaluation."
                    ) from e
                logger.error(f"Stall on {label}: {e}. Not retrying; degrading this request.")
                return None
            except openai.APIConnectionError as e:
                connect_failures += 1
                if connect_failures > _CONNECT_RETRIES:
                    raise RuntimeError(
                        f"vLLM server at {self.base_url} is unreachable ({e}); aborting the evaluation."
                    ) from e
                logger.warning(f"Connection failure on {label}: {e} — retrying in {_CONNECT_RETRY_SLEEP_S}s")
                await asyncio.sleep(_CONNECT_RETRY_SLEEP_S)
            except openai.APIStatusError as e:
                if e.status_code == 429 or e.status_code >= 500:
                    wait_time = self._backoff(attempt)
                    logger.warning(f"HTTP {e.status_code} on {label} — backing off {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"HTTP {e.status_code} on {label}: {e}. Not retrying; degrading this request.")
                    return None
            except openai.APIError as e:
                wait_time = self._backoff(attempt)
                logger.warning(f"Server error in the {label} stream: {e} — retrying in {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        logger.error(f"{label} failed after {self.API_MAX_RETRY} attempts; degrading this request.")
        return None

    def _backoff(self, attempt: int) -> float:
        return min(64.0, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))

    def _retry_upper_bound(self) -> int:
        # Connection retries are counted separately from backoff attempts; the
        # loop bound covers whichever path is taken.
        return self.API_MAX_RETRY + _CONNECT_RETRIES + 1

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _sampling_params(self) -> tuple[dict, dict]:
        """Split generation parameters into standard OpenAI params and vLLM extras.

        Standard params ride as SDK keyword arguments; vLLM-only knobs travel in
        ``extra_body`` — plain body fields the vLLM server accepts. LiteLLM
        silently dropped these (``drop_params=True``).
        """
        gp = self.generation_parameters
        standard = {
            "temperature": gp.temperature,
            "top_p": gp.top_p,
            "seed": gp.seed,
            "frequency_penalty": gp.frequency_penalty,
            "presence_penalty": gp.presence_penalty,
        }
        extra = {
            "top_k": gp.top_k,
            "min_p": gp.min_p,
            "repetition_penalty": gp.repetition_penalty,
            "skip_special_tokens": gp.skip_special_tokens,
            "min_tokens": gp.min_new_tokens,
        }
        return (
            {k: v for k, v in standard.items() if v is not None},
            {k: v for k, v in extra.items() if v is not None},
        )

    def _stop_for(self, stop_sequence: list[str] | None) -> list[str] | None:
        # The task's stop sequences win; configured stop_tokens are the fallback.
        return stop_sequence or self.generation_parameters.stop_tokens or None

    def _prepare_context(self, doc: Doc) -> list[dict] | str:
        """Request payload for a doc: multimodal messages, chat messages, or plain text."""
        if doc.images:
            return self.prompt_manager.prepare_prompt_api_multimodal(doc)
        if self.prompt_manager.use_chat_template:
            return self.prompt_manager.prepare_prompt_api(doc)
        return self.prompt_manager.prepare_plain_text(doc)

    # ------------------------------------------------------------------
    # Generative routes
    # ------------------------------------------------------------------

    def _prompt_room(self, reserve: int | None) -> int | None:
        """Prompt tokens the context budget leaves after ``reserve``; None when there is nothing to hold."""
        if self._context_budget is None or not reserve:
            return None
        room = self._context_budget - reserve
        return room if room > 0 else None

    async def _call_api_chat_generative(
        self, client: AsyncOpenAI, messages, max_new_tokens, num_samples, *, prompt_room: int | None = None
    ):
        standard, extra_body = self._sampling_params()
        if prompt_room is not None:
            extra_body["truncate_prompt_tokens"] = prompt_room
        if self.prompt_manager.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(self.prompt_manager.chat_template_kwargs)

        async def call():
            # No stop sequences on the chat route: chat-templated models
            # terminate via EOS (mirrors the in-process VLLMModel, which sets
            # stop_tokens = [] under a chat template). Task stop sequences are
            # completion-style markers ("\n", ...) that would truncate
            # legitimate output — a reasoning model opening with "<think>\n"
            # dies after one token otherwise.
            stream = await self._open_stream(
                client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    n=num_samples,
                    max_tokens=max_new_tokens if max_new_tokens else NOT_GIVEN,
                    stream=True,
                    timeout=self._stream_timeout(),
                    extra_body=extra_body or None,
                    **standard,
                )
            )
            return await _collect_chat_stream(stream, num_samples, progress=self._clock(), stall_s=self._stall_s())

        return await self._request(call, label="chat completion")

    async def _call_api_text_generative(
        self, client: AsyncOpenAI, prompt: str, max_new_tokens, num_samples, stop, *, prompt_room: int | None = None
    ):
        standard, extra_body = self._sampling_params()
        extra_body["add_special_tokens"] = self.add_special_tokens
        if prompt_room is not None:
            extra_body["truncate_prompt_tokens"] = prompt_room

        async def call():
            stream = await self._open_stream(
                client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    n=num_samples,
                    max_tokens=max_new_tokens if max_new_tokens else NOT_GIVEN,
                    stop=self._stop_for(stop) or NOT_GIVEN,
                    stream=True,
                    timeout=self._stream_timeout(),
                    extra_body=extra_body or None,
                    **standard,
                )
            )
            return await _collect_text_stream(stream, num_samples, progress=self._clock(), stall_s=self._stall_s())

        return await self._request(call, label="text completion")

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        """Generates responses using a greedy decoding strategy until certain ending conditions are met."""
        dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)

        async def process_splits() -> list[ModelResponse]:
            results = []
            async with self._make_client() as client, self._gate() as semaphore:
                for split in tqdm(
                    dataset.splits_iterator(),
                    total=dataset.num_dataset_splits,
                    desc="Splits",
                    position=0,
                    disable=self.disable_tqdm,
                ):
                    use_chat_template = self.prompt_manager.use_chat_template
                    if not use_chat_template and any(doc.images for doc in split):
                        raise ValueError("Multimodal prompts are only supported with chat template format.")
                    max_new_tokens = self.generation_parameters.max_new_tokens or split[0].generation_size
                    num_samples = split[0].num_samples
                    stop_sequence = split[0].stop_sequences

                    if num_samples > 1 and self.generation_parameters.temperature == 0:
                        raise ValueError(
                            "num_samples > 1 is not supported with temperature=0, "
                            "please set temperature > 0 or use non sampling metrics."
                        )

                    async def bounded_call(doc):
                        # Context prep (incl. image base64-encoding) happens under the
                        # semaphore so it overlaps with in-flight requests instead of
                        # running serially for the whole split up front.
                        async with semaphore:
                            context = self._prepare_context(doc)
                            prompt_room = None if doc.images else self._prompt_room(max_new_tokens)
                            if use_chat_template:
                                response = await self._call_api_chat_generative(
                                    client, context, max_new_tokens, num_samples, prompt_room=prompt_room
                                )
                            else:
                                response = await self._call_api_text_generative(
                                    client,
                                    context,
                                    max_new_tokens,
                                    num_samples,
                                    stop_sequence,
                                    prompt_room=prompt_room,
                                )
                            return context, response

                    pairs = await asyncio.gather(*[bounded_call(doc) for doc in split])

                    for context, response in pairs:
                        if response is None or not getattr(response, "choices", None):
                            results.append(ModelResponse(text=[""], reasonings=[None], input=context))
                            continue
                        if use_chat_template:
                            result = [choice.message.content for choice in response.choices]
                            reasonings = [
                                getattr(choice.message, "reasoning_content", None) for choice in response.choices
                            ]
                        else:
                            result = [getattr(choice, "text", None) for choice in response.choices]
                            reasonings = [None for _ in result]

                        results.append(
                            ModelResponse(
                                # In empty responses, the model should return an empty string instead of None
                                text=result if result and result[0] else [""],
                                reasonings=reasonings,
                                input=context,
                            )
                        )

            return results

        return dataset.get_original_order(asyncio.run(process_splits()))

    # ------------------------------------------------------------------
    # Loglikelihood — /v1/completions echo route (plain-text contexts)
    # ------------------------------------------------------------------

    async def _call_api_text_completion_async(self, client: AsyncOpenAI, prompt: str | list[int]):
        """Score ``prompt`` (text, or token ids) via ``echo=True`` + ``logprobs=1`` (deterministic)."""

        async def call():
            return await client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=1,  # generate exactly 1 token (echo gives prompt logprobs)
                echo=True,  # return prompt tokens with their log-probabilities
                logprobs=1,  # top-1 logprob per position for argmax check
                temperature=0.0,  # deterministic scoring
                seed=self.generation_parameters.seed if self.generation_parameters.seed is not None else NOT_GIVEN,
            )

        return await self._request(call, label="echo loglikelihood")

    async def _process_doc_loglikelihood_async(
        self,
        doc: Doc,
        context_str: str,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
    ) -> ModelResponse:
        """Compute logprobs for all choices of a single doc concurrently.

        Returns a ``ModelResponse`` with ``logprobs`` and
        ``argmax_logits_eq_gold`` populated per-choice, matching the VLLMModel
        data contract exactly.
        """
        # Whitespace at the context/continuation boundary is scored as part of
        # the continuation, exactly like ``tok_encode_pair``.
        context_for_alignment = context_str.rstrip()

        async def bounded_call(choice):
            async with semaphore:
                return await self._call_api_text_completion_async(client, context_str + choice)

        responses = await asyncio.gather(*[bounded_call(choice) for choice in doc.choices])

        logprobs_per_choice: list[float] = []
        argmax_per_choice: list[bool] = []

        for choice, response in zip(doc.choices, responses):
            if response is None or not getattr(response, "choices", None):
                logprobs_per_choice.append(float("-inf"))
                argmax_per_choice.append(False)
                continue

            lp_obj = getattr(response.choices[0], "logprobs", None)
            if lp_obj is None or not getattr(lp_obj, "token_logprobs", None):
                logprobs_per_choice.append(float("-inf"))
                argmax_per_choice.append(False)
                continue

            tokens: list[str] = list(lp_obj.tokens or [])
            token_logprobs: list = list(lp_obj.token_logprobs or [])
            top_logprobs: list = list(lp_obj.top_logprobs or [])

            # vLLM always returns text_offset on /v1/completions, so alignment
            # is exact; there is no local-tokenizer fallback to fall back to.
            cont_start = find_continuation_start(getattr(lp_obj, "text_offset", None), context_for_alignment)

            # token_logprobs[cont_start:-1] isolates the continuation slice.
            # The -1 excludes the single token generated by max_tokens=1 which
            # is appended at the very end of the echoed sequence.
            cont_lp_slice = token_logprobs[cont_start:-1]
            valid_lp = [v for v in cont_lp_slice if v is not None]
            total_logprob = sum(valid_lp) if valid_lp else float("-inf")

            is_argmax = check_argmax(tokens, top_logprobs, cont_start)

            logprobs_per_choice.append(total_logprob)
            argmax_per_choice.append(is_argmax)

        return ModelResponse(
            input=context_str,
            logprobs=logprobs_per_choice,
            argmax_logits_eq_gold=argmax_per_choice,
        )

    # ------------------------------------------------------------------
    # Loglikelihood — client token ids over the echo route (the in-process
    # contract: same tokenizer, same context/continuation split)
    # ------------------------------------------------------------------

    async def _process_doc_token_loglikelihood_async(
        self,
        doc: Doc,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
    ) -> ModelResponse:
        """Score every choice of ``doc`` from ids the client tokenized.

        The server echoes logprobs for exactly the ids it was given, so the
        continuation is the last ``len(continuation_ids)`` positions — the same
        boundary convention as ``VLLMModel._loglikelihood_tokens``, with no
        text-offset alignment in between.
        """
        self.tokenizer  # noqa: B018 — loads lazily and arms the prompt manager's chat template
        context = self.prompt_manager.prepare_prompt(doc)
        context_ids, continuation_ids = self.tok_encode_pair(context, doc.choices, pairwise=self.pairwise_tokenization)
        prompts = [list(ctx) + list(cont) for ctx, cont in zip(context_ids, continuation_ids)]
        if (room := self._prompt_room(1)) is not None:
            prompts = [p[-room:] if len(cont) < room else p for p, cont in zip(prompts, continuation_ids)]

        async def bounded_call(prompt_ids):
            async with semaphore:
                return await self._call_api_text_completion_async(client, prompt_ids)

        responses = await asyncio.gather(*[bounded_call(prompt_ids) for prompt_ids in prompts])

        logprobs_per_choice: list[float] = []
        argmax_per_choice: list[bool] = []
        for prompt_ids, cont_ids, response in zip(prompts, continuation_ids, responses):
            lp_obj = (
                getattr(response.choices[0], "logprobs", None)
                if response and getattr(response, "choices", None)
                else None
            )
            if lp_obj is None or not getattr(lp_obj, "token_logprobs", None):
                logprobs_per_choice.append(float("-inf"))
                argmax_per_choice.append(False)
                continue
            token_logprobs = list(lp_obj.token_logprobs or [])
            cont_start = len(prompt_ids) - len(cont_ids)
            # [cont_start, len(prompt_ids)) — the generated token sits at len(prompt_ids).
            cont_slice = token_logprobs[cont_start : len(prompt_ids)]
            valid = [v for v in cont_slice if v is not None]
            if cont_ids and not valid:
                logprobs_per_choice.append(float("-inf"))
                argmax_per_choice.append(False)
                continue
            logprobs_per_choice.append(float(sum(valid)))
            argmax_per_choice.append(
                check_argmax(list(lp_obj.tokens or []), list(lp_obj.top_logprobs or []), cont_start)
            )

        return ModelResponse(
            input=context,
            input_tokens=[list(ids) for ids in context_ids],
            output_tokens=[list(ids) for ids in continuation_ids],
            logprobs=logprobs_per_choice,
            argmax_logits_eq_gold=argmax_per_choice,
        )

    # ------------------------------------------------------------------
    # Loglikelihood — vLLM chat prompt_logprobs route (server-templated
    # contexts: vision, and text when client_tokenization is off)
    # ------------------------------------------------------------------

    def _build_ll_context_messages(self, doc: Doc) -> list[dict]:
        """Chat context messages shared by every choice of ``doc``.

        Each choice is scored appended as an (unterminated) assistant reply:
        the server renders its chat template with
        ``continue_final_message=True``, so the scored prompt ends exactly with
        the continuation. Multimodal docs use the same message structure as
        ``PromptManager.prepare_prompt_multimodal`` (via
        ``prepare_multimodal_messages``); text docs reuse
        ``prepare_prompt_api``. Built once per doc so images are only
        base64-encoded once, not per choice.
        """
        if doc.images:
            return self.prompt_manager.prepare_multimodal_messages(doc, image_url_parts(doc.images))
        return self.prompt_manager.prepare_prompt_api(doc)

    async def _call_api_chat_prompt_logprobs_async(self, client: AsyncOpenAI, messages: list[dict]):
        """Score a chat prompt through vLLM's ``prompt_logprobs`` extension.

        ``continue_final_message=True`` + ``add_generation_prompt=False`` make
        the server template the final assistant message as an unterminated
        reply, i.e. the continuation ends the scored prompt. The
        ``prompt_logprobs`` response field is a vLLM extension outside the
        OpenAI schema; the SDK's response models carry it as an extra
        attribute (pydantic ``extra="allow"``).
        """
        extra_body = {
            "add_generation_prompt": False,
            "continue_final_message": True,
            "prompt_logprobs": 1,
        }
        if self.prompt_manager.chat_template_kwargs:
            extra_body["chat_template_kwargs"] = dict(self.prompt_manager.chat_template_kwargs)

        async def call():
            return await client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                seed=self.generation_parameters.seed if self.generation_parameters.seed is not None else NOT_GIVEN,
                extra_body=extra_body,
            )

        return await self._request(call, label="chat loglikelihood")

    @staticmethod
    def _response_prompt_logprobs(response) -> list | None:
        """Pull vLLM's ``prompt_logprobs`` extra field off an SDK response object."""
        if response is None:
            return None
        raw = getattr(response, "prompt_logprobs", None)
        if raw is None and getattr(response, "choices", None):
            raw = getattr(response.choices[0], "prompt_logprobs", None)
        return raw

    async def _process_doc_chat_loglikelihood_async(
        self,
        doc: Doc,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
    ) -> ModelResponse:
        """Compute chat-templated logprobs for all choices of a single doc concurrently."""
        context_messages = self._build_ll_context_messages(doc)

        async def bounded_call(choice):
            async with semaphore:
                return await self._call_api_chat_prompt_logprobs_async(
                    client, [*context_messages, {"role": "assistant", "content": choice}]
                )

        responses = await asyncio.gather(*[bounded_call(choice) for choice in doc.choices])

        logprobs_per_choice: list[float] = []
        argmax_per_choice: list[bool] = []

        for choice, response in zip(doc.choices, responses):
            entries = extract_prompt_logprob_entries(self._response_prompt_logprobs(response))
            if not entries:
                logprobs_per_choice.append(float("-inf"))
                argmax_per_choice.append(False)
                continue

            pieces = [entry[0] for entry in entries]
            cont_start = align_continuation_suffix(pieces, choice)
            if cont_start is None:
                logger.warning(
                    f"Could not align continuation {choice!r} with the server's prompt tokens for doc '{doc.id}'. "
                    "Scoring the trailing tokens covering its length instead."
                )
                cont_start = len(pieces)
                covered = 0
                while cont_start > 0 and covered < len(choice):
                    cont_start -= 1
                    covered += len(pieces_to_text([pieces[cont_start]]))

            scored = entries[cont_start:]
            if scored:
                logprobs_per_choice.append(sum(logprob for _, logprob, _ in scored))
                argmax_per_choice.append(all(rank == 1 for _, _, rank in scored))
            else:
                logprobs_per_choice.append(0.0)
                argmax_per_choice.append(True)

        return ModelResponse(
            input=str(context_messages),
            logprobs=logprobs_per_choice,
            argmax_logits_eq_gold=argmax_per_choice,
        )

    # ------------------------------------------------------------------
    # Public loglikelihood API
    # ------------------------------------------------------------------

    async def _loglikelihood_async(self, docs: list[Doc], client: AsyncOpenAI) -> list[ModelResponse]:
        """Async coordinator: process every doc in parallel, bounded by the semaphore.

        ``asyncio.gather`` preserves input order, so the returned list aligns
        1-to-1 with ``docs``.
        """
        semaphore = asyncio.Semaphore(self.concurrent_requests)

        def route(doc: Doc):
            if self.client_tokenization and not doc.images:
                return self._process_doc_token_loglikelihood_async(doc=doc, client=client, semaphore=semaphore)
            if self.prompt_manager.use_chat_template:
                return self._process_doc_chat_loglikelihood_async(doc=doc, client=client, semaphore=semaphore)
            return self._process_doc_loglikelihood_async(
                doc=doc,
                context_str=self.prompt_manager.prepare_plain_text(doc),
                client=client,
                semaphore=semaphore,
            )

        return list(await asyncio.gather(*[route(doc) for doc in docs]))

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """Compute log-likelihoods for MCQ-style tasks against the vLLM server.

        With ``client_tokenization`` (default) text docs are tokenized here —
        chat template rendered locally when ``use_chat_template`` — and scored
        from ids over ``/v1/completions`` ``echo=True``, the in-process
        contract. Docs with images (and everything when client tokenization is
        off) go through the server-templated routes: vLLM's ``prompt_logprobs``
        chat extension, or the text echo route aligned via ``text_offset``.
        """
        if not self.prompt_manager.use_chat_template:
            docs_with_images = [doc for doc in docs if doc.images]
            if docs_with_images:
                raise ValueError(
                    "Multimodal loglikelihood requires use_chat_template=True (chat-templated route); "
                    f"got {len(docs_with_images)} doc(s) with images with use_chat_template=False."
                )

        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)

        async def process_splits() -> list[ModelResponse]:
            results = []
            async with self._make_client() as client:
                for split in tqdm(
                    dataset.splits_iterator(),
                    total=dataset.num_dataset_splits,
                    desc="Loglikelihood splits",
                    position=0,
                    disable=self.disable_tqdm,
                ):
                    results.extend(await self._loglikelihood_async(list(split), client))
            return results

        return dataset.get_original_order(asyncio.run(process_splits()))

    # ------------------------------------------------------------------
    # Rolling loglikelihood (perplexity)
    # ------------------------------------------------------------------

    async def _process_doc_rolling_async(
        self,
        doc: Doc,
        client: AsyncOpenAI,
        semaphore: asyncio.Semaphore,
    ) -> ModelResponse:
        """Compute per-token log-probabilities for the entire document text.

        Sends the full document as the prompt with ``echo=True`` and collects
        one logprob per token (skipping the leading null and the trailing
        generated token appended by ``max_tokens=1``).
        """
        doc_text = self.prompt_manager.prepare_plain_text(doc)
        async with semaphore:
            response = await self._call_api_text_completion_async(client, doc_text)

        if response is None or not getattr(response, "choices", None):
            return ModelResponse(input=doc_text, logprobs=[float("-inf")])

        lp_obj = getattr(response.choices[0], "logprobs", None)
        if lp_obj is None or not getattr(lp_obj, "token_logprobs", None):
            return ModelResponse(input=doc_text, logprobs=[float("-inf")])

        token_logprobs: list = list(lp_obj.token_logprobs or [])

        # token_logprobs[0]  → always None  (first token has no prior context)
        # token_logprobs[1:-1] → per-token log-probs for the full document
        # token_logprobs[-1] → the 1 newly generated token from max_tokens=1 (discard)
        rolling_logprobs = [v for v in token_logprobs[1:-1] if v is not None]

        return ModelResponse(input=doc_text, logprobs=rolling_logprobs)

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """Compute rolling log-likelihoods for perplexity-style evaluation."""
        dataset = LoglikelihoodDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)

        async def process_splits() -> list[ModelResponse]:
            results = []
            async with self._make_client() as client:
                semaphore = asyncio.Semaphore(self.concurrent_requests)
                for split in tqdm(
                    dataset.splits_iterator(),
                    total=dataset.num_dataset_splits,
                    desc="Loglikelihood rolling splits",
                    position=0,
                    disable=self.disable_tqdm,
                ):
                    results.extend(
                        await asyncio.gather(
                            *[self._process_doc_rolling_async(doc, client, semaphore) for doc in split]
                        )
                    )
            return results

        return dataset.get_original_order(asyncio.run(process_splits()))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tokenizer(self):
        """Loaded on first use for client tokenization; ``None`` when that is off (API-style paths never tokenize)."""
        if self._tokenizer is None and self.client_tokenization:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_id or self.model, revision=self._revision, trust_remote_code=self._trust_remote_code
            )
        if self._tokenizer is not None:
            self.prompt_manager.tokenizer = self._tokenizer
        return self._tokenizer

    @property
    def add_special_tokens(self) -> bool:
        if self._add_special_tokens is not None:
            return self._add_special_tokens
        return not self.prompt_manager.use_chat_template  # a chat template carries its own BOS

    @property
    def max_length(self) -> int:
        """Server context window: configured value, else probed once from ``/models``."""
        if self._max_length is not None:
            return self._max_length

        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = httpx.get(f"{self.base_url.rstrip('/')}/models", headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json().get("data") or []
            max_len = data[0].get("max_model_len") if data else None
            if max_len:
                self._max_length = int(max_len)
                return self._max_length
        except Exception as e:  # noqa: BLE001 — probe is best-effort
            logger.warning(f"Could not probe max_model_len from {self.base_url}/models: {e}")

        logger.warning(f"Falling back to default max_length={_DEFAULT_MAX_LENGTH}.")
        self._max_length = _DEFAULT_MAX_LENGTH
        return self._max_length
