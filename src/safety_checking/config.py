"""Model and experiment configs.

Everything about how a model is served and sampled lives in a YAML model config: the repo and
revision, parallelism, max length, parser names, sampling parameters and chat template kwargs.
Nothing about a model is hard-coded in the Slurm script or the runner.

The config's content hash is part of the adapter name, and so part of every run id: change a
parser, a sampling parameter or the pinned vLLM version and the runs are new runs.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .canonical import canonical_json, short_hash

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_local_url(url: str | None) -> bool:
    """True for a base_url on this machine. Such runs are free and need no --confirm-paid."""
    if not url:
        return False
    return (urlparse(url).hostname or "") in LOCAL_HOSTS


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServeConfig(_Model):
    """How ``vllm serve`` is started. Read by ``sc serve-args``, used by the Slurm script."""

    vllm_version: str
    tensor_parallel_size: int = 1
    max_model_len: int
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int | None = None
    seed: int = 0
    tool_call_parser: str
    reasoning_parser: str | None = None
    enable_prefix_caching: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    startup_timeout_s: int = 1500
    #: anything else, passed through verbatim
    extra_args: list[str] = Field(default_factory=list)
    #: environment variables for the server process. They belong here, not in the job script,
    #: when they change how the model is served (a sampler or attention implementation), so
    #: that they are hashed into the run id like every other serving choice.
    env: dict[str, str] = Field(default_factory=dict)


class RequestConfig(_Model):
    """What every chat-completions request carries."""

    #: standard OpenAI parameters (temperature, top_p, presence_penalty, max_tokens, ...)
    sampling: dict[str, Any] = Field(default_factory=dict)
    #: non-standard parameters, sent through the SDK's ``extra_body`` (top_k, min_p,
    #: chat_template_kwargs, ...)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    #: per-request seed is base_seed + sample_index, so samples differ but are reproducible.
    #: None leaves the seed to the server.
    base_seed: int | None = 0


class LogprobFormat(_Model):
    """How this model's tool-call format spells the start of a call (SPEC.md 3.10).

    A property of the tool parser's wire format, so it lives with the model. It is left out
    of the config hash: it changes nothing about how sampled runs are served, and the probe
    spec is hashed into every logprob record id instead.
    """

    #: text a call starts with, up to where the tool name begins
    call_opener: str
    #: what follows the tool name. Scored together with the name, so that a name which is a
    #: prefix of another, or merges with the next character at a token boundary, is handled.
    name_terminator: str


class ModelConfig(_Model):
    id: str
    hf_repo: str
    revision: str | None = None
    served_model_name: str
    serve: ServeConfig
    request: RequestConfig = Field(default_factory=RequestConfig)
    logprob: LogprobFormat | None = None
    notes: str = ""

    @property
    def base_url(self) -> str:
        return f"http://{self.serve.host}:{self.serve.port}/v1"

    @property
    def content_hash(self) -> str:
        return short_hash(self.model_dump(mode="json", exclude={"notes", "logprob"}), 12)

    @property
    def adapter_name(self) -> str:
        """Goes into every run id: the config id plus a hash of everything in it."""
        return f"vllm:{self.id}#{self.content_hash}"

    def request_params(self) -> dict[str, Any]:
        """The params dict the runner hashes into the run id and sends with each request."""
        params = dict(self.request.sampling)
        if self.request.extra_body:
            params["extra_body"] = dict(self.request.extra_body)
        return params

    def serve_argv(self) -> list[str]:
        """The full ``vllm serve`` command line, as a list."""
        s = self.serve
        argv = ["vllm", "serve", self.hf_repo]
        if self.revision:
            argv += ["--revision", self.revision]
        argv += [
            "--served-model-name",
            self.served_model_name,
            "--host",
            s.host,
            "--port",
            str(s.port),
            "--tensor-parallel-size",
            str(s.tensor_parallel_size),
            "--max-model-len",
            str(s.max_model_len),
            "--dtype",
            s.dtype,
            "--gpu-memory-utilization",
            str(s.gpu_memory_utilization),
            "--seed",
            str(s.seed),
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            s.tool_call_parser,
        ]
        if s.reasoning_parser:
            argv += ["--reasoning-parser", s.reasoning_parser]
        if s.max_num_seqs is not None:
            argv += ["--max-num-seqs", str(s.max_num_seqs)]
        argv.append(
            "--enable-prefix-caching" if s.enable_prefix_caching else "--no-enable-prefix-caching"
        )
        return argv + list(s.extra_args)

    def serve_command(self) -> str:
        return shlex.join(self.serve_argv())


class ProbeCall(_Model):
    tool: str
    #: string values may use $recipient_first_name, $recipient_id and $target_document
    args: dict[str, Any] = Field(default_factory=dict)


class ProbePoint(_Model):
    """A state at which logprob mode reads the model's next-call distribution.

    ``calls`` is a scripted continuation of the decision segment, executed against the world
    like a history episode. Empty means the decision start itself.
    """

    id: str
    calls: list[ProbeCall] = Field(default_factory=list)


class ExperimentConfig(_Model):
    experiment: str
    #: path to the model config, relative to this file
    model: str
    scenarios: list[str]
    lengths: list[int]
    patterns: list[str] = Field(default_factory=lambda: ["none"])
    n_samples: int
    arm: str = "baseline"
    max_steps: int = 8
    #: how many samples of one prefix are in flight at once. The first sample of a cell always
    #: goes alone, so the rest hit a warm prefix cache.
    concurrency: int = 1
    #: logprob mode only; ignored by sampled runs and absent from their run ids
    probe_points: list[ProbePoint] = Field(default_factory=lambda: [ProbePoint(id="start")])
    #: Sampling parameters that replace the model config's for this experiment only. Keys are
    #: the model config's ``request.sampling`` keys, plus ``extra_body``, which is merged one
    #: level deep (so overriding ``top_k`` keeps ``chat_template_kwargs``). The merged params are
    #: what the run id hashes, so an override means new runs and an empty one changes nothing.
    #: Sampled runs only: logprob mode always sends neutral sampling.
    sampling_override: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_sampling_override(self) -> ExperimentConfig:
        if "seed" in self.sampling_override:
            raise ValueError("sampling_override: the seed is base_seed + sample_index, not a param")
        extra_body = self.sampling_override.get("extra_body", {})
        if not isinstance(extra_body, dict):
            raise ValueError("sampling_override: extra_body must be a mapping")
        if self.sampling_override and self.arm == "baseline":
            # the analysis groups by arm, and "baseline" means the model config as written
            raise ValueError("sampling_override needs an arm name other than 'baseline'")
        return self

    def request_params(self, model: ModelConfig) -> dict[str, Any]:
        """The model's request params with this experiment's override applied."""
        params = model.request_params()
        for key, value in self.sampling_override.items():
            if key == "extra_body":
                params["extra_body"] = {**params.get("extra_body", {}), **value}
            else:
                params[key] = value
        return params


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping")
    return raw


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(_load_yaml(Path(path)))


def load_experiment_config(path: str | Path) -> tuple[ExperimentConfig, ModelConfig]:
    """Load an experiment config and the model config it points at."""
    path = Path(path)
    experiment = ExperimentConfig.model_validate(_load_yaml(path))
    return experiment, load_model_config(path.parent / experiment.model)


def load_any(path: str | Path) -> tuple[ExperimentConfig | None, ModelConfig]:
    """Accept either kind of YAML: an experiment config (which names its model) or a model."""
    path = Path(path)
    if "experiment" in _load_yaml(path):
        return load_experiment_config(path)
    return None, load_model_config(path)


def describe(config: ModelConfig) -> str:
    """Canonical JSON of a model config, for logs."""
    return canonical_json(config.model_dump(mode="json"))
