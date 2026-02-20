#!/usr/bin/env python3
"""
tiny_LLM Quickstart Launcher

Interactive launcher for common repo workflows with:
- structured menus
- command preview
- explicit confirmation before execution
- cross-platform Python support
- PowerShell-script support when `pwsh`/`powershell` is available
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


ASCII_ART = r"""
 _______ _                   _      _      __  __
|__   __(_)                 | |    | |    |  \/  |
   | |   _ _ __  _   _      | |    | |    | \  / |
   | |  | | '_ \| | | |     | |    | |    | |\/| |
   | |  | | | | | |_| |     | |____| |____| |  | |
   |_|  |_|_| |_|\__, |     |______|______|_|  |_|
                   __/ |
                  |___/
"""


def on_windows() -> bool:
    return os.name == "nt"


def shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in argv)


def print_banner(root: Path, animate: bool, anim_speed: float) -> None:
    speed = max(0.2, float(anim_speed))
    if animate:
        line_delay = 0.05 / speed
        for ln in ASCII_ART.strip("\n").splitlines():
            print(ln)
            sys.stdout.flush()
            time.sleep(line_delay)
        print("")
        title = "tiny_LLM Quickstart Launcher"
        char_delay = 0.012 / speed
        for ch in title:
            print(ch, end="", flush=True)
            time.sleep(char_delay)
        print("")
    else:
        print(ASCII_ART)
        print("tiny_LLM Quickstart Launcher")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Repository root: {root}")


def prompt_str(label: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(f"{label}: ").strip()
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else default


def prompt_int(label: str, default: int, min_value: int = 1) -> int:
    while True:
        raw = prompt_str(label, str(default))
        try:
            v = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if v < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return v


def prompt_float(label: str, default: float, min_value: float = 0.0) -> float:
    while True:
        raw = prompt_str(label, str(default))
        try:
            v = float(raw)
        except ValueError:
            print("Please enter a valid number.")
            continue
        if v < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return v


def prompt_bool(label: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer yes or no.")


@dataclass
class CommandSpec:
    kind: str  # python | ps1 | program | shell
    args: List[str]
    cwd: Path
    note: str = ""


@dataclass
class Plan:
    title: str
    description: str
    commands: List[CommandSpec]
    warnings: List[str] = field(default_factory=list)


Builder = Callable[[Path], Optional[Plan]]


@dataclass
class Action:
    action_id: str
    name: str
    description: str
    builder: Builder


@dataclass
class Menu:
    name: str
    description: str
    items: List[Tuple[str, Union["Menu", Action]]] = field(default_factory=list)


def p_cmd(cwd: Path, *args: str) -> CommandSpec:
    return CommandSpec(kind="python", args=list(args), cwd=cwd)


def ps1_cmd(cwd: Path, script_rel: str, *args: str) -> CommandSpec:
    return CommandSpec(kind="ps1", args=[script_rel, *args], cwd=cwd)


def prog_cmd(cwd: Path, *args: str) -> CommandSpec:
    return CommandSpec(kind="program", args=list(args), cwd=cwd)


def shell_cmd(cwd: Path, command: str) -> CommandSpec:
    return CommandSpec(kind="shell", args=[command], cwd=cwd)


def resolve_ps_exec() -> Optional[str]:
    pwsh = shutil.which("pwsh")
    if pwsh:
        return pwsh
    if on_windows():
        ps = shutil.which("powershell")
        if ps:
            return ps
    return None


def has_powershell_runtime() -> bool:
    return resolve_ps_exec() is not None


@functools.lru_cache(maxsize=1)
def detect_torch_backends() -> Tuple[Optional[bool], Optional[bool]]:
    probe = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " cuda = bool(torch.cuda.is_available())\n"
        " mps_backend = getattr(torch.backends, 'mps', None)\n"
        " mps = bool(mps_backend and mps_backend.is_available())\n"
        " print(json.dumps({'cuda': cuda, 'mps': mps}))\n"
        "except Exception:\n"
        " print('{}')\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
            timeout=8,
        )
    except Exception:
        return None, None
    if int(proc.returncode) != 0:
        return None, None
    raw = (proc.stdout or "").strip()
    if not raw:
        return None, None
    try:
        data = json.loads(raw.splitlines()[-1].strip())
    except Exception:
        return None, None
    return bool(data.get("cuda", False)), bool(data.get("mps", False))


def has_cuda_runtime() -> bool:
    cuda, _ = detect_torch_backends()
    return bool(cuda)


def preferred_train_dtype() -> str:
    cuda, mps = detect_torch_backends()
    if cuda is True or mps is True:
        return "float16"
    return "auto"


def has_committed_files(root: Path, rel_path: str) -> bool:
    """
    Return True only when git tracks at least one file under rel_path.
    This prevents showing local-only folders not committed to the repository.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", rel_path],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    if int(proc.returncode) != 0:
        return False
    return any(line.strip() for line in proc.stdout.splitlines())


def resolve_argv(cmd: CommandSpec) -> Tuple[Optional[List[str]], Optional[str]]:
    if cmd.kind == "python":
        return [sys.executable, *cmd.args], None
    if cmd.kind == "program":
        return cmd.args, None
    if cmd.kind == "shell":
        if on_windows():
            return ["cmd", "/c", cmd.args[0]], None
        return ["/bin/sh", "-lc", cmd.args[0]], None
    if cmd.kind == "ps1":
        exe = resolve_ps_exec()
        if not exe:
            return None, "PowerShell runtime not found (`pwsh`/`powershell`)."
        script = cmd.cwd / cmd.args[0]
        if not script.exists():
            return None, f"PowerShell script not found: {script}"
        return [exe, "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *cmd.args[1:]], None
    return None, f"Unsupported command kind: {cmd.kind}"


def run_plan(plan: Plan, auto_yes: bool) -> int:
    print(f"\n== {plan.title} ==")
    print(plan.description)
    for w in plan.warnings:
        print(f"WARNING: {w}")
    print("\nCommands:")
    resolved: List[Tuple[CommandSpec, List[str]]] = []
    for idx, c in enumerate(plan.commands, start=1):
        argv, err = resolve_argv(c)
        print(f"  {idx}. cwd={c.cwd}")
        if c.note:
            print(f"     note: {c.note}")
        if err:
            print(f"     unresolved: {err}")
            return 2
        print(f"     {shell_join(argv)}")
        resolved.append((c, argv))

    if not auto_yes and not prompt_bool("Proceed?", default=False):
        print("Cancelled.")
        return 0

    for idx, (c, argv) in enumerate(resolved, start=1):
        print(f"\n[{idx}/{len(resolved)}] Running: {shell_join(argv)}")
        proc = subprocess.run(argv, cwd=str(c.cwd))
        if int(proc.returncode) != 0:
            print(f"FAILED with code {int(proc.returncode)}")
            return int(proc.returncode)
        print("OK")
    return 0


def plan_install_requirements(root: Path) -> Plan:
    return Plan(
        title="Install requirements",
        description="Install pip and dependencies from requirements.txt.",
        commands=[
            p_cmd(root, "-m", "pip", "install", "-U", "pip"),
            p_cmd(root, "-m", "pip", "install", "-r", "requirements.txt"),
        ],
    )


def plan_show_structure(root: Path) -> Plan:
    if on_windows():
        return Plan(
            title="Show repository structure",
            description="List root files/folders.",
            commands=[prog_cmd(root, "powershell", "-NoLogo", "-NoProfile", "-Command", "Get-ChildItem -Force")],
        )
    return Plan(title="Show repository structure", description="List root files/folders.", commands=[prog_cmd(root, "ls", "-la")])


def plan_mini_chat(root: Path) -> Plan:
    return Plan(
        title="mini_assistant chat",
        description="Grounded web assistant (default HF backend).",
        commands=[
            p_cmd(
                root,
                "-m",
                "mini_assistant.chat",
                "--backend",
                "hf",
                "--model_name",
                "Qwen/Qwen3-4B-Instruct-2507",
                "--embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--temperature",
                "0.0",
            )
        ],
    )


def plan_mini_chat_debug(root: Path) -> Plan:
    return Plan(
        title="mini_assistant chat (debug)",
        description="Grounded chat with routing debug.",
        commands=[
            p_cmd(
                root,
                "-m",
                "mini_assistant.chat",
                "--backend",
                "hf",
                "--show_debug",
                "--direct_confidence_threshold",
                "0.72",
            )
        ],
    )


def plan_mini_chat_url(root: Path) -> Optional[Plan]:
    url = prompt_str("Fixed URL", "https://en.wikipedia.org/wiki/Italy").strip()
    if not url:
        return None
    return Plan(
        title="mini_assistant chat (fixed URL)",
        description="Run chat pinned to one source URL.",
        commands=[p_cmd(root, "-m", "mini_assistant.chat", "--backend", "hf", "--url", url)],
    )


def plan_mini_direct_chat(root: Path) -> Plan:
    return Plan(
        title="mini_assistant direct chat",
        description="Direct chat without web retrieval.",
        commands=[p_cmd(root, "-m", "mini_assistant.direct_chat", "--backend", "hf")],
    )


def plan_api_server_default(root: Path) -> Plan:
    return Plan(
        title="model_api_server default",
        description="Start OpenAI-compatible local API server.",
        commands=[p_cmd(root, "model_api_server.py", "--host", "127.0.0.1", "--port", "8001", "--default_model", "tiny-llm-7b")],
    )


def plan_api_server_custom(root: Path) -> Plan:
    host = prompt_str("Host", "127.0.0.1")
    port = prompt_int("Port", 8001)
    default_model = prompt_str("Default model", "tiny-llm-7b")
    args = ["model_api_server.py", "--host", host, "--port", str(port), "--default_model", default_model]
    if not prompt_bool("Preload default model?", default=True):
        args.append("--no_preload_default")
    return Plan(title="model_api_server custom", description="Start server with custom parameters.", commands=[p_cmd(root, *args)])


def plan_api_smoke(root: Path) -> Plan:
    return Plan(
        title="API smoke test",
        description="Run scripts/api_smoke_test.ps1.",
        commands=[ps1_cmd(root, "scripts/api_smoke_test.ps1")],
        warnings=["Requires PowerShell (`pwsh`/`powershell`)."],
    )


def plan_eval_grounded(root: Path) -> Plan:
    return Plan(
        title="Eval grounded",
        description="Run eval.py grounded suite.",
        commands=[
            p_cmd(
                root,
                "eval.py",
                "--suite",
                "grounded",
                "--backend",
                "hf",
                "--model_name",
                "Qwen/Qwen3-4B-Instruct-2507",
                "--embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
        ],
    )


def plan_eval_chat(root: Path) -> Plan:
    return Plan(title="Eval chat", description="Run eval.py chat suite.", commands=[p_cmd(root, "eval.py", "--suite", "chat", "--backend", "hf")])


def plan_eval_both(root: Path) -> Plan:
    return Plan(title="Eval both", description="Run both suites.", commands=[p_cmd(root, "eval.py", "--suite", "both", "--backend", "hf")])


def plan_eval_conf_gate(root: Path) -> Plan:
    return Plan(
        title="Eval confidence gate",
        description="Run mini_assistant confidence-gate evaluation.",
        commands=[
            p_cmd(
                root,
                "-m",
                "mini_assistant.eval_confidence_gate",
                "--backend",
                "hf",
                "--model_name",
                "Qwen/Qwen3-4B-Instruct-2507",
                "--embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
                "--direct_confidence_threshold",
                "0.72",
            )
        ],
    )


def plan_unit_tests(root: Path) -> Plan:
    return Plan(
        title="Run unit tests",
        description="Run tiny-llm test suite.",
        commands=[p_cmd(root, "-m", "unittest", "discover", "-s", "tiny-llm/tests", "-p", "test_*.py")],
    )


def plan_check_script(root: Path) -> Plan:
    return Plan(
        title="Run scripts/check.ps1",
        description="PowerShell wrapper for test checks.",
        commands=[ps1_cmd(root, "scripts/check.ps1")],
        warnings=["Requires PowerShell (`pwsh`/`powershell`)."],
    )


def plan_download_05b(root: Path) -> Plan:
    return Plan(
        title="Download 0.5B base",
        description="tiny-llm/01_download_base.py preset.",
        commands=[
            p_cmd(
                root / "tiny-llm",
                "01_download_base.py",
                "--model_id",
                "Qwen/Qwen2.5-0.5B-Instruct",
                "--output_dir",
                "models/base",
                "--dtype",
                "auto",
            )
        ],
    )


def plan_download_3b(root: Path) -> Plan:
    return Plan(
        title="Download 3B base",
        description="tiny-llm/01_download_base.py 3B preset.",
        commands=[
            p_cmd(
                root / "tiny-llm",
                "01_download_base.py",
                "--model_id",
                "Qwen/Qwen2.5-3B-Instruct",
                "--output_dir",
                "models/base_3b",
                "--dtype",
                "auto",
            )
        ],
    )


def plan_download_7b_lora(root: Path) -> Plan:
    return Plan(
        title="Download 7B LoRA base",
        description="tiny-llm/03_download_lora_base.py 7B preset.",
        commands=[
            p_cmd(
                root / "tiny-llm",
                "03_download_lora_base.py",
                "--model_id",
                "Qwen/Qwen2.5-7B-Instruct",
                "--output_dir",
                "models/lora_base_7b",
                "--dtype",
                "auto",
            )
        ],
    )


def plan_train_base_05b(root: Path) -> Plan:
    return Plan(
        title="Train base 0.5B",
        description="Knowledge-heavy preset from README.",
        commands=[
            p_cmd(
                root / "tiny-llm",
                "02_train_base.py",
                "--model_dir",
                "models/base",
                "--output_dir",
                "models/base_trained",
                "--recipe",
                "knowledge-heavy",
                "--max_steps",
                "30000",
                "--repeat_sources",
                "--gradient_checkpointing",
            )
        ],
    )


def plan_train_base_3b(root: Path) -> Plan:
    return Plan(
        title="Train base 3B CPT",
        description="3B code-focused CPT preset.",
        commands=[
            p_cmd(
                root / "tiny-llm",
                "02_train_base.py",
                "--model_dir",
                "models/base_3b",
                "--output_dir",
                "models/base_3b_code_fast_16gb_v1",
                "--disable_local_data",
                "--hf_source",
                "codeparrot/github-code||train|code|800000",
                "--hf_code_languages",
                "python,typescript",
                "--hf_require_language_tag",
                "--repeat_sources",
                "--max_steps",
                "6000",
                "--learning_rate",
                "2e-5",
                "--warmup_ratio",
                "0.02",
                "--per_device_batch_size",
                "2",
                "--grad_accum",
                "12",
                "--block_size",
                "768",
                "--auto_tune_shape",
                "--auto_tune_batch_candidates",
                "1,2,3",
                "--auto_tune_block_candidates",
                "512,768,1024",
                "--gradient_checkpointing",
                "--dtype",
                "auto",
                "--logging_steps",
                "20",
                "--save_steps",
                "500",
                "--save_total_limit",
                "4",
                "--disable_sample_logging",
            )
        ],
        warnings=["Long-running and GPU-intensive."],
    )


def plan_train_base_custom(root: Path) -> Plan:
    model_dir = prompt_str("Model dir", "models/base")
    out_dir = prompt_str("Output dir", "models/base_custom")
    recipe = prompt_str("Recipe [tiny|standard|knowledge-heavy]", "standard")
    steps = prompt_int("Max steps", 3000)
    batch = prompt_int("Batch size", 1)
    accum = prompt_int("Grad accum", 8)
    lr = prompt_float("Learning rate", 2e-5)
    block = prompt_int("Block size", 1024, min_value=64)
    args = [
        "02_train_base.py",
        "--model_dir",
        model_dir,
        "--output_dir",
        out_dir,
        "--recipe",
        recipe,
        "--max_steps",
        str(steps),
        "--per_device_batch_size",
        str(batch),
        "--grad_accum",
        str(accum),
        "--learning_rate",
        str(lr),
        "--block_size",
        str(block),
    ]
    if prompt_bool("Repeat sources?", True):
        args.append("--repeat_sources")
    if prompt_bool("Gradient checkpointing?", True):
        args.append("--gradient_checkpointing")
    return Plan(title="Train base custom", description="Interactive custom base training.", commands=[p_cmd(root / "tiny-llm", *args)])


def plan_train_lora_7b(root: Path) -> Plan:
    use_4bit = has_cuda_runtime()
    args = [
        "04_train_lora.py",
        "--model_dir",
        "models/lora_base_7b",
        "--output_dir",
        "models/lora7b_seed_v1",
        "--disable_hf_data",
        "--validate_data",
        "--chat_format",
        "tokenizer",
        "--code_fence_hygiene",
        "normalize",
        "--reject_no_markdown_code_examples",
        "--fail_on_duplicate_examples",
        "--max_duplicate_example_ratio",
        "0.10",
        "--min_loaded_examples",
        "250",
        "--local_jsonl_glob",
        "samples/sft/repair_math_logic_coding.jsonl",
        "--local_jsonl_glob",
        "samples/sft/system_styles.jsonl",
        "--local_jsonl_glob",
        "samples/sft/chat_alignment_samples.jsonl",
        "--local_jsonl_glob",
        "samples/sft/formatting_code_fences.jsonl",
        "--local_jsonl_glob",
        "samples/sft/format_constraints_strict.jsonl",
        "--local_jsonl_glob",
        "samples/sft/math_reasoning_micro.jsonl",
        "--max_steps",
        "300",
        "--max_length",
        "1024",
        "--per_device_batch_size",
        "1",
        "--grad_accum",
        "16",
        "--learning_rate",
        "3e-5",
        "--gradient_checkpointing",
        "--dtype",
        preferred_train_dtype(),
        "--logging_steps",
        "20",
        "--save_steps",
        "100",
        "--save_total_limit",
        "6",
    ]
    warnings: List[str] = []
    if use_4bit:
        args.extend(["--use_4bit", "--bnb_4bit_quant_type", "nf4", "--bnb_4bit_compute_dtype", "auto"])
    else:
        warnings.append("CUDA not detected: running LoRA preset without 4-bit QLoRA flags.")
    return Plan(
        title="Train LoRA 7B",
        description="Official 7B seed preset (QLoRA enabled when CUDA is available).",
        commands=[p_cmd(root / "tiny-llm", *args)],
        warnings=warnings,
    )


def plan_train_lora_3b(root: Path) -> Plan:
    return Plan(
        title="Train LoRA 3B",
        description="3B seed LoRA preset.",
        commands=[
            p_cmd(
                root / "tiny-llm",
                "04_train_lora.py",
                "--model_dir",
                "models/base_3b_code_fast_16gb_v1",
                "--output_dir",
                "models/lora3b_code_review_seed_v1",
                "--disable_hf_data",
                "--validate_data",
                "--chat_format",
                "tokenizer",
                "--code_fence_hygiene",
                "normalize",
                "--reject_no_markdown_code_examples",
                "--fail_on_duplicate_examples",
                "--max_duplicate_example_ratio",
                "0.10",
                "--min_loaded_examples",
                "200",
                "--local_jsonl_glob",
                "samples/sft/code_review_seed.jsonl",
                "--local_jsonl_glob",
                "samples/sft/code_assistant_booster.jsonl",
                "--local_jsonl_glob",
                "samples/sft/code_review_synthetic.jsonl",
                "--max_steps",
                "400",
                "--max_length",
                "1024",
                "--per_device_batch_size",
                "1",
                "--grad_accum",
                "16",
                "--learning_rate",
                "4e-5",
                "--gradient_checkpointing",
                "--dtype",
                preferred_train_dtype(),
            )
        ],
    )


def plan_train_lora_custom(root: Path) -> Plan:
    model_dir = prompt_str("Model dir", "models/lora_base_7b")
    out_dir = prompt_str("Output dir", "models/lora_custom")
    recipe = prompt_str("Recipe [tiny|standard|heavy]", "heavy")
    steps = prompt_int("Max steps", 300)
    max_len = prompt_int("Max length", 1024, 64)
    batch = prompt_int("Batch size", 1)
    accum = prompt_int("Grad accum", 16)
    lr = prompt_float("Learning rate", 3e-5)
    args = [
        "04_train_lora.py",
        "--model_dir",
        model_dir,
        "--output_dir",
        out_dir,
        "--recipe",
        recipe,
        "--max_steps",
        str(steps),
        "--max_length",
        str(max_len),
        "--per_device_batch_size",
        str(batch),
        "--grad_accum",
        str(accum),
        "--learning_rate",
        str(lr),
        "--dtype",
        preferred_train_dtype(),
    ]
    warnings: List[str] = []
    if prompt_bool("Disable HF data?", True):
        args.append("--disable_hf_data")
    if prompt_bool("Validate data?", True):
        args.append("--validate_data")
    ask_4bit = prompt_bool("Use 4-bit QLoRA (CUDA only)?", has_cuda_runtime())
    if ask_4bit and has_cuda_runtime():
        args.extend(["--use_4bit", "--bnb_4bit_quant_type", "nf4", "--bnb_4bit_compute_dtype", "auto"])
    elif ask_4bit:
        warnings.append("CUDA not detected: skipped --use_4bit to avoid unsupported configuration.")
    globs = prompt_str(
        "Local JSONL globs comma-separated",
        "samples/sft/repair_math_logic_coding.jsonl,samples/sft/system_styles.jsonl,samples/sft/chat_alignment_samples.jsonl",
    )
    for g in [x.strip() for x in globs.split(",") if x.strip()]:
        args.extend(["--local_jsonl_glob", g])
    return Plan(
        title="Train LoRA custom",
        description="Interactive custom LoRA/QLoRA training.",
        commands=[p_cmd(root / "tiny-llm", *args)],
        warnings=warnings,
    )


def plan_eval_checkpoints(root: Path) -> Plan:
    base_model = prompt_str("Base model dir", "models/lora_base_7b")
    adapter = prompt_str("Adapter dir", "models/lora7b_seed_v1")
    max_ckpt = prompt_int("Max checkpoints", 6)
    out_json = prompt_str("Output JSON", f"{adapter}/checkpoint_eval_report.json")
    return Plan(
        title="Evaluate checkpoints",
        description="Run 05_eval_lora_checkpoints.py.",
        commands=[
            p_cmd(
                root / "tiny-llm",
                "05_eval_lora_checkpoints.py",
                "--base_model_dir",
                base_model,
                "--adapter_dir",
                adapter,
                "--max_checkpoints",
                str(max_ckpt),
                "--out_json",
                out_json,
            )
        ],
    )


def plan_regression_mock(root: Path) -> Plan:
    return Plan(title="Regression suite mock", description="Fast mock regression.", commands=[p_cmd(root / "tiny-llm", "regression_suite.py", "--backend", "mock")])


def plan_regression_hf(root: Path) -> Plan:
    model = prompt_str("Model dir", "models/lora_base_7b")
    adapter = prompt_str("Adapter dir (optional)", "models/lora7b_seed_v1/checkpoint-300").strip()
    args = ["regression_suite.py", "--backend", "hf", "--model_dir", model, "--device", "auto", "--chat_format", "tokenizer", "--max_new_tokens", "120"]
    if adapter:
        args.extend(["--adapter_dir", adapter])
    return Plan(title="Regression suite HF", description="HF regression with selected model.", commands=[p_cmd(root / "tiny-llm", *args)])


def plan_ps_script(root: Path, script_name: str, title: str, desc: str) -> Plan:
    return Plan(
        title=title,
        description=desc,
        commands=[ps1_cmd(root / "tiny-llm", f"scripts/{script_name}")],
        warnings=[
            "Requires PowerShell (`pwsh`/`powershell`).",
            "For cross-platform training/SFT use Python flows under tiny-llm Train Base / Train LoRA menus.",
        ],
    )


def plan_ps_release_windows_only(root: Path) -> Optional[Plan]:
    if not on_windows():
        print("release_lmstudio.ps1 is Windows-only by design.")
        print("Use Windows for release packaging; training/SFT remains cross-platform via Python commands.")
        return None
    return plan_ps_script(
        root,
        "release_lmstudio.ps1",
        "release_lmstudio.ps1",
        "PowerShell workflow (Windows-only release packaging).",
    )


def plan_rag_local(root: Path) -> Plan:
    return Plan(
        title="RAG router local",
        description="Interactive local router with memory.",
        commands=[
            p_cmd(
                root / "tiny-llm",
                "rag_memory_router.py",
                "--interactive",
                "--router",
                "local",
                "--knowledge_glob",
                "samples/base/*.txt",
                "--knowledge_glob",
                "samples/sft/*.jsonl",
                "--memory_file",
                "models/chat_memory/dev_session.jsonl",
                "--show_trace",
            )
        ],
    )


def plan_rag_auto(root: Path) -> Plan:
    return Plan(
        title="RAG router auto",
        description="Interactive local/cloud router.",
        commands=[p_cmd(root / "tiny-llm", "rag_memory_router.py", "--interactive", "--router", "auto", "--show_trace")],
        warnings=["Set OPENAI_API_KEY before using cloud route."],
    )


def plan_custom_python(root: Path) -> Optional[Plan]:
    raw = prompt_str("Python args after executable (e.g. -m mini_assistant.chat --backend hf)", "").strip()
    if not raw:
        return None
    args = shlex.split(raw, posix=not on_windows())
    cwd = (root / prompt_str("Working dir (repo-relative)", ".")).resolve()
    return Plan(title="Custom Python command", description="Run custom Python command.", commands=[p_cmd(cwd, *args)])


def plan_custom_shell(root: Path) -> Optional[Plan]:
    cmd = prompt_str("Shell command", "").strip()
    if not cmd:
        return None
    cwd = (root / prompt_str("Working dir (repo-relative)", ".")).resolve()
    return Plan(title="Custom shell command", description="Run custom shell command.", commands=[shell_cmd(cwd, cmd)], warnings=["Use with care."])


def build_actions(root: Path) -> Dict[str, Action]:
    a: Dict[str, Action] = {}

    def reg(action_id: str, name: str, desc: str, builder: Builder) -> Action:
        act = Action(action_id, name, desc, builder)
        a[action_id] = act
        return act

    reg("env.install", "Install requirements", "pip + requirements", plan_install_requirements)
    reg("env.structure", "Show structure", "list repo files/folders", plan_show_structure)
    reg("mini.chat", "Run mini_assistant chat", "grounded chat", plan_mini_chat)
    reg("mini.chat.debug", "Run mini_assistant chat (debug)", "show route debug", plan_mini_chat_debug)
    reg("mini.chat.url", "Run mini_assistant chat (fixed URL)", "chat with --url", plan_mini_chat_url)
    reg("mini.direct", "Run mini_assistant direct chat", "no web retrieval", plan_mini_direct_chat)
    reg("api.server.default", "Start API server default", "model_api_server.py", plan_api_server_default)
    reg("api.server.custom", "Start API server custom", "host/port/model prompt", plan_api_server_custom)
    reg("api.smoke", "Run API smoke test", "scripts/api_smoke_test.ps1", plan_api_smoke)
    reg("eval.grounded", "Run grounded eval", "eval.py --suite grounded", plan_eval_grounded)
    reg("eval.chat", "Run chat eval", "eval.py --suite chat", plan_eval_chat)
    reg("eval.both", "Run both evals", "eval.py --suite both", plan_eval_both)
    reg("eval.conf", "Run confidence-gate eval", "mini_assistant.eval_confidence_gate", plan_eval_conf_gate)
    reg("eval.tests", "Run tiny-llm unit tests", "unittest discover", plan_unit_tests)
    reg("eval.check", "Run scripts/check.ps1", "PowerShell check wrapper", plan_check_script)
    reg("tiny.download.05b", "Download base 0.5B", "01_download_base.py", plan_download_05b)
    reg("tiny.download.3b", "Download base 3B", "01_download_base.py 3B", plan_download_3b)
    reg("tiny.download.7b", "Download LoRA base 7B", "03_download_lora_base.py", plan_download_7b_lora)
    reg("tiny.train.base.05b", "Train base 0.5B", "knowledge-heavy preset", plan_train_base_05b)
    reg("tiny.train.base.3b", "Train base 3B", "CPT preset", plan_train_base_3b)
    reg("tiny.train.base.custom", "Train base custom", "custom base params", plan_train_base_custom)
    reg("tiny.train.lora.3b", "Train LoRA 3B", "3B seed preset", plan_train_lora_3b)
    reg("tiny.train.lora.7b", "Train LoRA 7B", "official 7B preset", plan_train_lora_7b)
    reg("tiny.train.lora.custom", "Train LoRA custom", "custom LoRA params", plan_train_lora_custom)
    reg("tiny.eval.ckpt", "Evaluate checkpoints", "05_eval_lora_checkpoints.py", plan_eval_checkpoints)
    reg("tiny.reg.mock", "Regression mock", "regression_suite mock", plan_regression_mock)
    reg("tiny.reg.hf", "Regression HF", "regression_suite hf", plan_regression_hf)
    reg("tiny.ps.sft", "run_lora_sft.ps1", "PowerShell workflow", lambda r: plan_ps_script(r, "run_lora_sft.ps1", "run_lora_sft.ps1", "PowerShell workflow"))
    reg("tiny.ps.sftq", "run_lora_sft_quality.ps1", "PowerShell workflow", lambda r: plan_ps_script(r, "run_lora_sft_quality.ps1", "run_lora_sft_quality.ps1", "PowerShell workflow"))
    reg("tiny.ps.repair", "run_lora_repair.ps1", "PowerShell workflow", lambda r: plan_ps_script(r, "run_lora_repair.ps1", "run_lora_repair.ps1", "PowerShell workflow"))
    reg("tiny.ps.targeted", "run_lora_targeted_repair.ps1", "PowerShell workflow", lambda r: plan_ps_script(r, "run_lora_targeted_repair.ps1", "run_lora_targeted_repair.ps1", "PowerShell workflow"))
    reg("tiny.ps.3b", "run_3b_programming_review.ps1", "PowerShell workflow", lambda r: plan_ps_script(r, "run_3b_programming_review.ps1", "run_3b_programming_review.ps1", "PowerShell workflow"))
    reg("tiny.ps.3bmax", "run_3b_code_assistant_max.ps1", "PowerShell workflow", lambda r: plan_ps_script(r, "run_3b_code_assistant_max.ps1", "run_3b_code_assistant_max.ps1", "PowerShell workflow"))
    reg("tiny.ps.release", "release_lmstudio.ps1", "Windows-only release workflow", plan_ps_release_windows_only)
    reg("rag.local", "Run RAG router local", "interactive local mode", plan_rag_local)
    reg("rag.auto", "Run RAG router auto", "interactive auto mode", plan_rag_auto)
    reg("util.custom.py", "Run custom Python command", "interactive custom command", plan_custom_python)
    reg("util.custom.sh", "Run custom shell command", "interactive custom command", plan_custom_shell)
    reg(
        "util.cross_platform_notes",
        "Show cross-platform notes",
        "Print OS support notes for training/SFT/release",
        lambda r: Plan(
            title="Cross-platform support notes",
            description="Training and SFT are cross-platform. Release is Windows-only.",
            commands=[],
            warnings=[
                "Cross-platform: download/train/eval flows based on Python scripts.",
                "Windows-only: tiny-llm/scripts/release_lmstudio.ps1",
                "QLoRA 4-bit paths require CUDA + bitsandbytes; on macOS/Linux without CUDA use standard LoRA.",
                "PowerShell wrappers may run with pwsh on Unix, but only release is intentionally Windows-only.",
            ],
        ),
    )
    return a


def maybe_experiment_menu(root: Path, actions: Dict[str, Action]) -> Optional[Menu]:
    exp_dir = root / "tiny-llm" / "experiment"
    if not exp_dir.exists():
        return None
    if not has_committed_files(root, "tiny-llm/experiment"):
        return None
    return Menu(
        "Experiments",
        "Local experiment workflows (if folder exists).",
        [
            ("start_api_tiny3b.ps1", Action("exp.start_api", "start_api_tiny3b.ps1", "experiment API preset", lambda r: Plan("start_api_tiny3b.ps1", "Start experiment API preset.", [ps1_cmd(r, "tiny-llm/experiment/start_api_tiny3b.ps1")], ["Requires PowerShell."]))),
            ("run_chat_100.ps1", Action("exp.chat100", "run_chat_100.ps1", "chat loop preset", lambda r: Plan("run_chat_100.ps1", "Run chat experiment preset.", [ps1_cmd(r, "tiny-llm/experiment/run_chat_100.ps1")], ["Requires PowerShell."]))),
            ("run_code_100.ps1", Action("exp.code100", "run_code_100.ps1", "code loop preset", lambda r: Plan("run_code_100.ps1", "Run code experiment preset.", [ps1_cmd(r, "tiny-llm/experiment/run_code_100.ps1")], ["Requires PowerShell."]))),
            ("run_triad_100.ps1", Action("exp.triad100", "run_triad_100.ps1", "triad loop preset", lambda r: Plan("run_triad_100.ps1", "Run triad experiment preset.", [ps1_cmd(r, "tiny-llm/experiment/run_triad_100.ps1")], ["Requires PowerShell."]))),
        ],
    )


def build_menu(root: Path, actions: Dict[str, Action]) -> Menu:
    root_menu = Menu("Main Menu", "Select a functionality area.")
    items: List[Tuple[str, Union[Menu, Action]]] = [
        ("Environment Setup", Menu("Environment Setup", "Bootstrap and inspect repository.", [("Install requirements", actions["env.install"]), ("Show structure", actions["env.structure"])])),
        ("Mini Assistant", Menu("Mini Assistant", "Grounded QA runtime.", [("Run grounded chat", actions["mini.chat"]), ("Run grounded chat (debug)", actions["mini.chat.debug"]), ("Run chat on fixed URL", actions["mini.chat.url"]), ("Run direct chat", actions["mini.direct"])])),
        ("Model API Server", Menu("Model API Server", "OpenAI-compatible local API.", [("Start server default", actions["api.server.default"]), ("Start server custom", actions["api.server.custom"]), ("Run API smoke test", actions["api.smoke"])])),
        ("Evaluation", Menu("Evaluation", "Regression and quality checks.", [("Grounded eval", actions["eval.grounded"]), ("Chat eval", actions["eval.chat"]), ("Both eval suites", actions["eval.both"]), ("Confidence gate eval", actions["eval.conf"]), ("Unit tests", actions["eval.tests"]), ("scripts/check.ps1", actions["eval.check"])])),
        ("tiny-llm Download", Menu("tiny-llm Download", "Download model artifacts.", [("Download base 0.5B", actions["tiny.download.05b"]), ("Download base 3B", actions["tiny.download.3b"]), ("Download LoRA base 7B", actions["tiny.download.7b"])])),
        ("tiny-llm Train Base", Menu("tiny-llm Train Base", "Base model training.", [("Train 0.5B preset", actions["tiny.train.base.05b"]), ("Train 3B preset", actions["tiny.train.base.3b"]), ("Train with your parameters", actions["tiny.train.base.custom"])])),
        ("tiny-llm Train LoRA", Menu("tiny-llm Train LoRA", "LoRA/QLoRA workflows.", [("Train 3B preset", actions["tiny.train.lora.3b"]), ("Train 7B preset", actions["tiny.train.lora.7b"]), ("Train with your parameters", actions["tiny.train.lora.custom"])])),
        ("tiny-llm Eval/Regression", Menu("tiny-llm Eval/Regression", "Checkpoint and regression tools.", [("Evaluate checkpoints", actions["tiny.eval.ckpt"]), ("Regression mock", actions["tiny.reg.mock"]), ("Regression HF", actions["tiny.reg.hf"])])),
        ("RAG Router", Menu("RAG Router", "RAG + memory router workflows.", [("Run local", actions["rag.local"]), ("Run auto local/cloud", actions["rag.auto"])])),
        ("Utilities", Menu("Utilities", "Custom runners for any repo option.", [("Cross-platform notes", actions["util.cross_platform_notes"]), ("Run custom Python command", actions["util.custom.py"]), ("Run custom shell command", actions["util.custom.sh"])])),
    ]
    if on_windows() or has_powershell_runtime():
        items.insert(
            8,
            (
                "tiny-llm PowerShell Workflows",
                Menu(
                    "tiny-llm PowerShell Workflows",
                    "End-to-end scripted workflows.",
                    [
                        ("run_lora_sft.ps1", actions["tiny.ps.sft"]),
                        ("run_lora_sft_quality.ps1", actions["tiny.ps.sftq"]),
                        ("run_lora_repair.ps1", actions["tiny.ps.repair"]),
                        ("run_lora_targeted_repair.ps1", actions["tiny.ps.targeted"]),
                        ("run_3b_programming_review.ps1", actions["tiny.ps.3b"]),
                        ("run_3b_code_assistant_max.ps1", actions["tiny.ps.3bmax"]),
                        ("release_lmstudio.ps1", actions["tiny.ps.release"]),
                    ],
                ),
            ),
        )
    root_menu.items = items
    exp_menu = maybe_experiment_menu(root, actions)
    if exp_menu is not None:
        root_menu.items.insert(len(root_menu.items) - 1, ("Experiments", exp_menu))
    return root_menu


def list_actions(actions: Dict[str, Action]) -> None:
    for k in sorted(actions.keys()):
        a = actions[k]
        print(f"{a.action_id}: {a.name} - {a.description}")


def menu_loop(root_menu: Menu, root: Path, auto_yes: bool) -> int:
    stack: List[Menu] = [root_menu]
    while stack:
        cur = stack[-1]
        print(f"\n{cur.name}\n{cur.description}\n")
        for i, (label, obj) in enumerate(cur.items, start=1):
            print(f"  {i}. {label}")
        if len(stack) > 1:
            print("  b. Back")
        print("  q. Quit")
        raw = input("\nSelect option: ").strip().lower()
        if raw == "q":
            return 0
        if raw == "b":
            if len(stack) > 1:
                stack.pop()
            continue
        if not raw.isdigit():
            print("Invalid selection.")
            continue
        idx = int(raw)
        if idx < 1 or idx > len(cur.items):
            print("Out of range.")
            continue
        _, obj = cur.items[idx - 1]
        if isinstance(obj, Menu):
            stack.append(obj)
            continue
        act = obj
        plan = act.builder(root)
        if plan is None:
            continue
        rc = run_plan(plan, auto_yes=auto_yes)
        if rc != 0:
            print(f"Action ended with exit code {rc}.")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="tiny_LLM quickstart launcher")
    ap.add_argument("--list-actions", action="store_true", help="List action IDs and exit.")
    ap.add_argument("--run", default="", help="Run one action ID and exit.")
    ap.add_argument("--yes", action="store_true", help="Auto-confirm command execution.")
    ap.add_argument("--no-anim", action="store_true", help="Disable animated banner.")
    ap.add_argument("--anim-speed", type=float, default=1.0, help="Animation speed multiplier (default: 1.0).")
    ap.add_argument("--root", default="", help="Optional repository root override.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent
    interactive = sys.stdout.isatty() and not args.list_actions and not args.run
    env_no_anim = os.environ.get("QUICKSTART_NO_ANIM", "").strip().lower() in {"1", "true", "yes", "on"}
    animate = bool(interactive and not args.no_anim and not env_no_anim)
    print_banner(root=root, animate=animate, anim_speed=float(args.anim_speed))
    actions = build_actions(root)

    if args.list_actions:
        list_actions(actions)
        return 0

    if args.run:
        act = actions.get(args.run)
        if act is None:
            print(f"Unknown action id: {args.run}")
            return 2
        plan = act.builder(root)
        if plan is None:
            return 1
        return run_plan(plan, auto_yes=bool(args.yes))

    menu = build_menu(root, actions)
    return menu_loop(menu, root, auto_yes=bool(args.yes))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
