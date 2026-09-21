"""The model set: every model YAML, and running one experiment config against any of them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from safety_checking import cli
from safety_checking.config import (
    ModelConfig,
    load_any,
    load_experiment_config,
    load_model_config,
)

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "configs" / "models"
MODEL_YAMLS = sorted(MODELS_DIR.glob("*.yaml"))
QWEN_27B = MODELS_DIR / "qwen3.8-27b-nothink.yaml"
SMOKE_YAML = REPO / "configs" / "smoke.yaml"
GATE_YAML = REPO / "configs" / "gate.yaml"

# four families, a small and a medium model each (SPEC.md 3.9)
FAMILIES = {
    "qwen": ("qwen3.5-9b-nothink", "qwen3.8-27b-nothink"),
    "gemma": ("gemma-4-12b-nothink", "gemma-4-31b-nothink"),
    "mistral": ("ministral-3-8b", "ministral-3-14b"),
    "gpt-oss": ("gpt-oss-20b-low", "gpt-oss-120b-low"),
}


def model_ids() -> list[str]:
    return [path.stem for path in MODEL_YAMLS]


def test_the_model_set_is_four_families_of_two() -> None:
    assert sorted(model_ids()) == sorted(m for pair in FAMILIES.values() for m in pair)


@pytest.mark.parametrize("path", MODEL_YAMLS, ids=model_ids())
def test_every_model_config_is_complete_and_pinned(path: Path) -> None:
    config = load_model_config(path)
    assert config.id == path.stem  # the file name is how a job names its model
    assert re.fullmatch(r"[0-9a-f]{40}", config.revision or ""), "pin a commit, not a branch"
    assert config.serve.host == "127.0.0.1"  # never bound to the network
    # LS6 compute nodes have no nvcc, whichever GPU they carry (job 3458103)
    assert config.serve.env.get("VLLM_USE_FLASHINFER_SAMPLER") == "0"
    assert config.request.base_seed == 0
    assert config.request.sampling["max_tokens"] >= 1024
    # a rate over samples needs samples that differ
    assert config.request.sampling["temperature"] >= 0.5


def test_one_vllm_environment_serves_every_model() -> None:
    assert {load_model_config(p).serve.vllm_version for p in MODEL_YAMLS} == {"0.29.0"}


def test_ids_served_names_and_run_id_names_are_distinct() -> None:
    configs = [load_model_config(p) for p in MODEL_YAMLS]
    for values in (
        [c.id for c in configs],
        [c.served_model_name for c in configs],
        [c.adapter_name for c in configs],
    ):
        assert len(set(values)) == len(configs)


def test_models_that_can_stop_thinking_are_told_to() -> None:
    for name in (*FAMILIES["qwen"], *FAMILIES["gemma"]):
        config = load_model_config(MODELS_DIR / f"{name}.yaml")
        kwargs = config.request.extra_body["chat_template_kwargs"]
        assert kwargs == {"enable_thinking": False}, name
    for name in FAMILIES["gpt-oss"]:  # cannot stop: the lowest effort, and the id says so
        config = load_model_config(MODELS_DIR / f"{name}.yaml")
        assert config.request.sampling["reasoning_effort"] == "low"
        assert config.logprob is None  # the next token is reasoning, not a call (SPEC 3.10)


def test_adding_the_model_set_did_not_move_the_first_models_run_ids() -> None:
    # 30 smoke runs are logged under this name; `download` must stay out of the hash
    assert load_model_config(QWEN_27B).adapter_name == "vllm:qwen3.8-27b-nothink#e21313de06da"


def test_what_is_downloaded_is_not_part_of_the_run_id() -> None:
    raw = yaml.safe_load((MODELS_DIR / "gpt-oss-20b-low.yaml").read_text())
    assert raw["download"]["exclude"] == ["original/*", "metal/*"]
    with_excludes = ModelConfig.model_validate(raw)
    raw["download"] = {"exclude": []}
    assert ModelConfig.model_validate(raw).adapter_name == with_excludes.adapter_name


def test_ministral_is_served_in_hugging_face_format() -> None:
    # Mistral's format validates tool-call ids as 9 alphanumerics; ours are call_r001
    for name in FAMILIES["mistral"]:
        config = load_model_config(MODELS_DIR / f"{name}.yaml")
        text = " ".join(config.serve_argv())
        assert "--tokenizer-mode hf --config-format hf --load-format safetensors" in text
        assert config.download.exclude == ["consolidated.safetensors"]
        assert config.serve.reasoning_parser is None


# -- one experiment config, any model ------------------------------------------------


def test_an_experiment_runs_against_the_model_it_names_unless_told_otherwise() -> None:
    other = MODELS_DIR / "gemma-4-12b-nothink.yaml"
    experiment, named = load_experiment_config(GATE_YAML)
    same_experiment, override = load_experiment_config(GATE_YAML, other)
    assert named.id == "qwen3.8-27b-nothink"
    assert override.id == "gemma-4-12b-nothink"
    assert same_experiment == experiment

    assert load_any(SMOKE_YAML, other)[1].id == "gemma-4-12b-nothink"
    with pytest.raises(ValueError, match="model config already"):
        load_any(QWEN_27B, other)


@pytest.mark.parametrize("path", MODEL_YAMLS, ids=model_ids())
def test_every_model_dry_runs_the_smoke_config_and_fits_its_context(path, capsys) -> None:
    # status 2 would mean a cell whose estimated context exceeds serve.max_model_len
    args = ["run", "--config", str(SMOKE_YAML), "--model-config", str(path), "--dry-run"]
    assert cli.main(args) == 0
    assert "6 cells, 30 runs" in capsys.readouterr().out.replace("\n", " ")


def test_model_config_flag_needs_an_experiment_config() -> None:
    with pytest.raises(SystemExit, match="--model-config goes with --config"):
        cli.main(["run", "--model", "fake:always_check", "--model-config", str(QWEN_27B)])


def test_serve_args_follows_the_override_and_prints_lists_one_item_per_line(capsys) -> None:
    override = ["--model-config", str(MODELS_DIR / "gpt-oss-120b-low.yaml")]
    assert cli.main(["serve-args", str(SMOKE_YAML), *override]) == 0
    assert capsys.readouterr().out.startswith("vllm serve openai/gpt-oss-120b ")

    assert cli.main(["serve-args", str(SMOKE_YAML), *override, "--field", "experiment.arm"]) == 0
    assert capsys.readouterr().out == "baseline\n"

    # what scripts/download_weights.sh reads with `mapfile -t`
    field = ["--field", "download.exclude"]
    assert cli.main(["serve-args", str(SMOKE_YAML), *override, *field]) == 0
    assert capsys.readouterr().out == "original/*\nmetal/*\n"
    assert cli.main(["serve-args", str(QWEN_27B), *field]) == 0
    assert capsys.readouterr().out == ""


def test_the_job_script_passes_the_model_override_to_every_sc_call() -> None:
    script = (REPO / "scripts" / "serve_and_run.slurm").read_text()
    calls = [
        line
        for line in script.splitlines()
        if re.search(r'\$SC"? (serve-args|render|run|logprob) ', line) and "#" not in line[:1]
    ]
    assert len(calls) == 6
    for line in calls:
        assert '"${MODEL_ARGS[@]}"' in line, line
