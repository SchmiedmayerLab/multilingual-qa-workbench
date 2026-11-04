#
# This source file is part of the Multilingual QA Workbench open-source project
#
# SPDX-FileCopyrightText: 2026 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import requests
from requests.exceptions import HTTPError, ReadTimeout


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    temperature: float | None = 0.0
    max_tokens: int | None = None
    timeout_seconds: int = 180
    max_retries: int = 3
    concurrency: int = 4
    extra_body: dict[str, Any] = field(default_factory=dict)
    zero_retention_exception: str | None = None

    def request_signature(self, system_prompt: str | None) -> str:
        """Identify generation settings independently of retry and concurrency settings."""
        settings = {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": self.extra_body,
            "system_prompt": system_prompt,
        }
        if self.zero_retention_exception:
            settings["zero_retention_exception"] = self.zero_retention_exception
        return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class OpenRouterResult:
    text: str
    response_id: str = ""
    response_model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class BudgetReport:
    overview: pd.DataFrame
    recommended_budgets: pd.DataFrame
    model_cost_drivers: pd.DataFrame
    activity_summary: pd.DataFrame
    assumptions: pd.DataFrame
    scenario_summary: pd.DataFrame
    budget_iterations: pd.DataFrame
    per_model_estimates: pd.DataFrame
    current_approved_summary: pd.DataFrame
    current_approved_per_model: pd.DataFrame


DEFAULT_TRANSLATION_SYSTEM_PROMPT = (
    "Translate the question into natural, idiomatic language for the target locale. "
    "Use the way a local speaker would realistically ask it in everyday life. "
    "Do not make it overly formal, stiff, or more clinical than the original. "
    "Preserve the original meaning, urgency, ambiguity, and intent. "
    "Return only the translated question."
)
DEFAULT_FALLBACK_ANSWER_OUTPUT_TOKENS = 600.0
DEFAULT_OPENROUTER_LEGACY_PRICE_OVERRIDES: dict[str, dict[str, Any]] = {
    "anthropic/claude-3.5-sonnet": {
        "model_name": "Anthropic: Claude 3.5 Sonnet",
        "input_cost_per_million": 6.0,
        "output_cost_per_million": 30.0,
        "pricing_source": "openrouter_legacy_override",
    },
    "google/gemini-pro-1.5": {
        "model_name": "Google: Gemini 1.5 Pro",
        "input_cost_per_million": 1.25,
        "output_cost_per_million": 5.0,
        "pricing_source": "openrouter_legacy_override",
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "model_name": "Meta: Llama 3.1 405B Instruct",
        "input_cost_per_million": 4.0,
        "output_cost_per_million": 4.0,
        "pricing_source": "openrouter_legacy_override",
    },
}
_OPENPYXL_DATA_VALIDATION_WARNING = (
    r".*Data Validation extension is not supported and will be removed.*"
)


def _read_review_workbook_sheet(workbook_path: str | Path, *, sheet_name: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_OPENPYXL_DATA_VALIDATION_WARNING,
            category=UserWarning,
        )
        frame = pd.read_excel(workbook_path, sheet_name=sheet_name, keep_default_na=False)
    continuation_columns: dict[str, list[tuple[int, str]]] = {}
    for column in frame.columns:
        match = re.fullmatch(r"(.+)__part_([2-9]|[1-9][0-9]+)", str(column))
        if match and match[1] in frame.columns:
            continuation_columns.setdefault(match[1], []).append((int(match[2]), column))
    for column, parts in continuation_columns.items():
        for _, part_column in sorted(parts):
            frame[column] = frame[column].astype(str) + frame[part_column].astype(str)
            frame = frame.drop(columns=[part_column])
    return frame


def _write_excel_sheet(frame: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str) -> None:
    """Keep complete text in adjacent columns instead of Excel's silent truncation."""
    export_columns: dict[str, pd.Series] = {}
    for column in frame.columns:
        values = frame[column]
        longest = max((len(value) for value in values if isinstance(value, str)), default=0)
        export_columns[column] = values.map(
            lambda value: value[:32767] if isinstance(value, str) else value
        )
        for part_index in range(1, math.ceil(longest / 32767)):
            part_column = f"{column}__part_{part_index + 1}"
            if part_column in frame.columns:
                raise ValueError(f"Reserved Excel continuation column: {part_column}")
            export_columns[part_column] = values.map(
                lambda value, offset=part_index * 32767: (
                    value[offset : offset + 32767] if isinstance(value, str) else ""
                )
            )
    pd.DataFrame(export_columns, index=frame.index).to_excel(
        writer, sheet_name=sheet_name, index=False
    )
    # Model output and reviewer text must remain literal text, including leading '='.
    for row in writer.sheets[sheet_name]:
        for cell in row:
            if cell.data_type == "f":
                cell.data_type = "s"


def questions_to_dataframe(questions: Iterable[str | dict[str, Any]]) -> pd.DataFrame:
    question_list = list(questions)
    if not question_list:
        return pd.DataFrame(
            columns=[
                "question_id",
                "question_text",
                "source_language_code",
                "source_language_name",
                "reference_answer",
                "risk_level",
                "topic",
            ]
        )

    if all(isinstance(question, str) for question in question_list):
        frame = pd.DataFrame(
            {
                "question_id": [f"q{i}" for i in range(1, len(question_list) + 1)],
                "question_text": question_list,
            }
        )
    else:
        frame = pd.DataFrame(question_list)
        required = {"question_text"}
        missing = required - set(frame.columns)
        if missing:
            missing_keys = ", ".join(sorted(missing))
            raise ValueError(f"Question entries are missing required keys: {missing_keys}")
        if "question_id" not in frame.columns:
            frame["question_id"] = [f"q{i}" for i in range(1, len(frame) + 1)]

    frame = frame.copy()
    frame["source_language_code"] = frame.get("source_language_code", "en")
    frame["source_language_name"] = frame.get("source_language_name", "English")
    frame["reference_answer"] = frame.get("reference_answer", "")
    frame["risk_level"] = frame.get("risk_level", "")
    frame["topic"] = frame.get("topic", "")
    return frame


def languages_to_dataframe(languages: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(languages)
    required = {"language_code", "language_name"}
    missing = required - set(frame.columns)
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"Language entries are missing required keys: {missing_keys}")
    return frame.copy()


def review_schema_to_dataframe(schema: Iterable[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(schema)
    required = {"field_key", "field_label", "options"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Review schema is missing required keys: {', '.join(sorted(missing))}")
    return frame.copy()


def build_question_language_matrix(
    questions: pd.DataFrame,
    languages: pd.DataFrame,
) -> pd.DataFrame:
    left = questions.copy()
    right = languages.copy()
    left["join_key"] = 1
    right["join_key"] = 1
    return left.merge(right, on="join_key").drop(columns=["join_key"])


REVIEW_TEXT_COLUMN_WIDTHS = {
    "question_text": 48,
    "translated_question": 58,
    "reference_answer": 58,
    "answer_text": 140,
    "review_notes": 44,
}
DEFAULT_REVIEW_ROW_HEIGHT = 30.0
MAX_REVIEW_ROW_HEIGHT = 409.0


def _auto_fit_columns(
    worksheet,
    *,
    fixed_widths: Mapping[str, float] | None = None,
) -> None:
    fixed_widths = fixed_widths or {}
    header_by_column = {cell.column: str(cell.value or "") for cell in worksheet[1]}
    for column_cells in worksheet.columns:
        column_letter = column_cells[0].column_letter
        header = re.sub(r"__part_\d+$", "", header_by_column.get(column_cells[0].column, ""))
        if header in fixed_widths:
            worksheet.column_dimensions[column_letter].width = fixed_widths[header]
            continue

        lengths = [len(str(cell.value or "")) for cell in column_cells[:25]]
        worksheet.column_dimensions[column_letter].width = min(max(lengths + [12]) + 2, 40)


def _estimate_wrapped_line_count(text: Any, width: float) -> int:
    normalized = str(text or "")
    if not normalized:
        return 1

    usable_width = max(int(width) - 2, 8)
    line_count = 0
    for paragraph in normalized.splitlines() or [""]:
        paragraph_length = max(len(paragraph), 1)
        line_count += max(1, math.ceil(paragraph_length / usable_width))
    return line_count


def _auto_fit_row_heights(
    worksheet,
    *,
    text_columns: Sequence[str] = (),
    default_height: float = DEFAULT_REVIEW_ROW_HEIGHT,
    max_height: float = MAX_REVIEW_ROW_HEIGHT,
) -> None:
    header_map = {str(cell.value or ""): cell.column for cell in worksheet[1]}
    text_column_indexes = [
        header_map[column_name] for column_name in text_columns if column_name in header_map
    ]
    if not text_column_indexes:
        return

    worksheet.row_dimensions[1].height = 24
    for row_index in range(2, worksheet.max_row + 1):
        max_lines = 1
        for column_index in text_column_indexes:
            cell = worksheet.cell(row=row_index, column=column_index)
            column_letter = cell.column_letter
            width = worksheet.column_dimensions[column_letter].width or 12
            max_lines = max(max_lines, _estimate_wrapped_line_count(cell.value, width))
        worksheet.row_dimensions[row_index].height = min(
            max(default_height, max_lines * 15.0 + 10.0),
            max_height,
        )


def _style_and_protect_sheet(
    worksheet,
    editable_columns: set[str],
    *,
    fixed_widths: Mapping[str, float] | None = None,
    row_height_text_columns: Sequence[str] = (),
) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill, Protection

    header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
    editable_fill = PatternFill(fill_type="solid", fgColor="FFF2CC")

    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="top", wrap_text=True)

    header_map = {cell.value: cell.column for cell in worksheet[1]}
    editable_indexes = {
        index
        for name, index in header_map.items()
        if re.sub(r"__part_\d+$", "", str(name)) in editable_columns
    }

    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.protection = Protection(locked=True)
            if cell.column in editable_indexes:
                cell.protection = Protection(locked=False)
                cell.fill = editable_fill

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    _auto_fit_columns(worksheet, fixed_widths=fixed_widths)
    _auto_fit_row_heights(worksheet, text_columns=row_height_text_columns)
    worksheet.protection.sheet = True


def _add_dropdown_validation(
    workbook,
    worksheet_name: str,
    column_name: str,
    options: list[str],
) -> None:
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    if worksheet_name not in workbook.sheetnames or "_lists" not in workbook.sheetnames:
        return

    worksheet = workbook[worksheet_name]
    header_map = {cell.value: cell.column_letter for cell in worksheet[1]}
    target_column = header_map.get(column_name)
    if not target_column:
        return

    lists_sheet = workbook["_lists"]
    list_column = lists_sheet.max_column + 1
    for index, option in enumerate(options, start=1):
        lists_sheet.cell(row=index, column=list_column, value=option)

    column_letter = get_column_letter(list_column)
    formula = f"'_lists'!${column_letter}$1:${column_letter}${len(options)}"

    validation = DataValidation(type="list", formula1=formula, allow_blank=True)
    validation.prompt = f"Select a value for {column_name}"
    validation.error = "Please choose one of the dropdown options."
    worksheet.add_data_validation(validation)
    validation.add(f"{target_column}2:{target_column}1048576")


def _finalize_workbook(
    path: Path,
    worksheet_name: str,
    editable_columns: set[str],
    dropdown_map: dict[str, list[str]],
    *,
    fixed_widths: Mapping[str, float] | None = None,
    row_height_text_columns: Sequence[str] = (),
) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    if "_lists" not in workbook.sheetnames:
        workbook.create_sheet("_lists")
    workbook["_lists"].sheet_state = "hidden"

    _style_and_protect_sheet(
        workbook[worksheet_name],
        editable_columns,
        fixed_widths=fixed_widths,
        row_height_text_columns=row_height_text_columns,
    )
    if "instructions" in workbook.sheetnames:
        _style_and_protect_sheet(workbook["instructions"], set())

    for column_name, options in dropdown_map.items():
        _add_dropdown_validation(workbook, worksheet_name, column_name, options)

    workbook.save(path)


def _load_checkpoint(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return pd.DataFrame()
    return load_table(checkpoint_path)


def load_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


def save_table(frame: pd.DataFrame, path: str | Path) -> None:
    table_path = Path(path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    # A failed write must not destroy the last usable checkpoint.
    with tempfile.NamedTemporaryFile(
        dir=table_path.parent, suffix=table_path.suffix, delete=False
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        if table_path.suffix == ".parquet":
            frame.to_parquet(temporary_path, index=False)
        else:
            frame.to_csv(temporary_path, index=False)
        temporary_path.replace(table_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_checkpoint(frame: pd.DataFrame, path: str | Path | None) -> None:
    if path is None:
        return
    save_table(frame, path)


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, remaining_seconds = divmod(int(seconds), 60)
    return f"{minutes}m {remaining_seconds}s"


def _estimate_remaining_seconds(
    *,
    elapsed_seconds: float,
    completed_in_run: int,
    remaining_in_run: int,
    concurrency: int,
) -> float | None:
    if completed_in_run <= 0 or elapsed_seconds <= 0 or remaining_in_run <= 0:
        return None

    parallelism = max(1, concurrency)
    throughput_seconds = remaining_in_run / (completed_in_run / elapsed_seconds)
    batch_seconds = elapsed_seconds / math.ceil(completed_in_run / parallelism)
    batch_eta = math.ceil(remaining_in_run / parallelism) * batch_seconds

    return min(throughput_seconds, batch_eta)


async def _call_openrouter_async(
    *,
    model: ModelConfig,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    return await asyncio.to_thread(
        _call_openrouter,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
    )


async def _call_openrouter_result_async(
    *,
    model: ModelConfig,
    prompt: str,
    system_prompt: str | None = None,
) -> OpenRouterResult:
    return await asyncio.to_thread(
        _call_openrouter_result,
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
    )


def _openrouter_headers() -> dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _openrouter_error_detail(response: requests.Response | None) -> str:
    if response is None:
        return "No response body."

    try:
        body = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:1000] if text else "No response body."

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message", "") or "").strip()
            code = str(error.get("code", "") or "").strip()
            metadata = error.get("metadata")
            parts = []
            if code:
                parts.append(f"code={code}")
            if message:
                parts.append(message)
            if metadata:
                parts.append(f"metadata={metadata}")
            if parts:
                return " | ".join(parts)[:1000]
        message = str(body.get("message", "") or "").strip()
        if message:
            return message[:1000]

    return str(body)[:1000]


def _collect_text_fragments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_collect_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments: list[str] = []
        preferred_keys = ["text", "content", "output_text"]
        for key in preferred_keys:
            if key in value:
                fragments.extend(_collect_text_fragments(value.get(key)))
        return fragments
    return []


def _extract_openrouter_response_text(body: dict[str, Any]) -> str | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if isinstance(message, dict):
        message_fragments = _collect_text_fragments(message.get("content"))
        if message_fragments:
            return "\n".join(message_fragments).strip()

    choice_fragments = _collect_text_fragments(first_choice.get("text"))
    if choice_fragments:
        return "\n".join(choice_fragments).strip()

    return None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except TypeError, ValueError:
        return None


def _extract_openrouter_result(body: dict[str, Any], response_text: str) -> OpenRouterResult:
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    return OpenRouterResult(
        text=response_text,
        response_id=str(body.get("id", "") or ""),
        response_model=str(body.get("model", "") or ""),
        prompt_tokens=_optional_int(usage.get("prompt_tokens")),
        completion_tokens=_optional_int(usage.get("completion_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
        reasoning_tokens=_optional_int(completion_details.get("reasoning_tokens")),
        cached_tokens=_optional_int(prompt_details.get("cached_tokens")),
        cost_usd=_optional_float(usage.get("cost")),
    )


def _openrouter_result_metadata(prefix: str, result: OpenRouterResult) -> dict[str, Any]:
    return {
        f"{prefix}_openrouter_generation_id": result.response_id,
        f"{prefix}_openrouter_model": result.response_model,
        f"{prefix}_openrouter_prompt_tokens": result.prompt_tokens,
        f"{prefix}_openrouter_completion_tokens": result.completion_tokens,
        f"{prefix}_openrouter_total_tokens": result.total_tokens,
        f"{prefix}_openrouter_reasoning_tokens": result.reasoning_tokens,
        f"{prefix}_openrouter_cached_tokens": result.cached_tokens,
        f"{prefix}_openrouter_cost_usd": result.cost_usd,
    }


def _call_openrouter(
    *,
    model: ModelConfig,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    return _call_openrouter_result(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
    ).text


def _call_openrouter_result(
    *,
    model: ModelConfig,
    prompt: str,
    system_prompt: str | None = None,
) -> OpenRouterResult:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model.model_id,
        "messages": messages,
    }
    if model.temperature is not None:
        payload["temperature"] = model.temperature
    payload.update(model.extra_body)
    payload["provider"] = {
        **payload.get("provider", {}),
        "data_collection": "deny",
        "zdr": not bool(model.zero_retention_exception),
    }
    if model.max_tokens is not None:
        payload["max_tokens"] = model.max_tokens

    last_error: Exception | None = None
    last_error_detail = ""
    for attempt in range(1, model.max_retries + 1):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=_openrouter_headers(),
                json=payload,
                timeout=model.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            response_text = _extract_openrouter_response_text(body)
            if response_text:
                return _extract_openrouter_result(body, response_text)

            choices = body.get("choices")
            first_choice = choices[0] if isinstance(choices, list) and choices else {}
            if not isinstance(first_choice, dict):
                first_choice = {}
            last_error = RuntimeError(
                f"OpenRouter returned no assistant text for {model.name}. "
                f"Finish reason: {first_choice.get('finish_reason')!r}."
            )
            if attempt == model.max_retries:
                break

            wait_seconds = min(2 ** (attempt - 1), 8)
            print(
                f"OpenRouter returned an empty response for {model.name} "
                f"(attempt {attempt}/{model.max_retries}). Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)
        except ReadTimeout as error:
            last_error = error
            if attempt == model.max_retries:
                break
            wait_seconds = min(2 ** (attempt - 1), 8)
            print(
                f"Request timed out for {model.name} "
                f"(attempt {attempt}/{model.max_retries}). Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)
        except HTTPError as error:
            status_code = error.response.status_code if error.response is not None else None
            last_error_detail = _openrouter_error_detail(error.response)
            if status_code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(
                    f"OpenRouter request failed for {model.name} "
                    f"({model.model_id}) with HTTP {status_code}: {last_error_detail}"
                ) from error

            last_error = error
            if attempt == model.max_retries:
                break

            retry_after_header = None
            if error.response is not None:
                retry_after_header = error.response.headers.get("Retry-After")

            if retry_after_header and retry_after_header.isdigit():
                wait_seconds = int(retry_after_header)
            else:
                wait_seconds = min(5 * (2 ** (attempt - 1)), 60)

            print(
                f"Request failed for {model.name} with HTTP {status_code} "
                f"(attempt {attempt}/{model.max_retries}). Retrying in {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    if isinstance(last_error, HTTPError) and last_error.response is not None:
        status_code = last_error.response.status_code
        raise RuntimeError(
            f"OpenRouter request failed for {model.name} ({model.model_id}) "
            f"with HTTP {status_code} after {model.max_retries} attempts: "
            f"{last_error_detail}"
        ) from last_error

    raise RuntimeError(
        f"OpenRouter request failed for {model.name} after {model.max_retries} attempts: "
        f"{last_error}"
    ) from last_error


async def propose_translations(
    matrix: pd.DataFrame,
    *,
    translation_model: ModelConfig,
    translation_system_prompt: str | None = DEFAULT_TRANSLATION_SYSTEM_PROMPT,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    force_restart: bool = False,
) -> pd.DataFrame:
    existing = pd.DataFrame() if force_restart else _load_checkpoint(checkpoint_path)
    signature = translation_model.request_signature(translation_system_prompt)
    completed_rows: list[dict[str, Any]] = []
    completed_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    total = len(matrix)
    if not existing.empty:
        for row in existing.to_dict(orient="records"):
            if row.get("translation_request_signature") != signature:
                continue
            key = (str(row.get("question_id", "")), str(row.get("language_code", "")))
            completed_lookup[key] = row

    for row in matrix.to_dict(orient="records"):
        key = (str(row.get("question_id", "")), str(row.get("language_code", "")))
        existing_row = completed_lookup.get(key)
        if existing_row and existing_row.get("question_text") == row.get("question_text"):
            completed_rows.append(existing_row)

    resumed_count = len(completed_rows)
    remaining_total = total - resumed_count

    if show_progress and resumed_count:
        print(
            f"Resuming translations from checkpoint: {resumed_count} already saved, "
            f"{remaining_total} remaining."
        )
    if show_progress and remaining_total == 0:
        print(f"Translations already complete: {total}/{total}.")
        return pd.DataFrame(completed_rows)

    started = time.monotonic()
    new_completed = 0
    pending_rows = []
    for row in matrix.itertuples(index=False):
        key = (str(row.question_id), str(row.language_code))
        existing_row = completed_lookup.get(key)
        if existing_row and existing_row.get("question_text") == row.question_text:
            continue
        pending_rows.append(row)

    semaphore = asyncio.Semaphore(max(1, translation_model.concurrency))

    async def translate_one(row):
        async with semaphore:
            source_language_name = getattr(row, "source_language_name", None) or "English"
            if row.source_language_code == row.language_code:
                suggestion = row.question_text
            else:
                prompt = (
                    f"Translate the following question from {source_language_name} "
                    f"to {row.language_name}. Make it sound natural and locally idiomatic, "
                    f"not overly formal, while preserving meaning, ambiguity, urgency, and "
                    f"clinical intent.\n\n"
                    f"Question: {row.question_text}"
                )
                result = await _call_openrouter_result_async(
                    model=translation_model,
                    prompt=prompt,
                    system_prompt=translation_system_prompt,
                )
                suggestion = result.text

            row_dict = row._asdict()
            row_dict["translation_suggestion"] = suggestion
            row_dict["translation_model_id"] = translation_model.model_id
            row_dict["translation_request_signature"] = signature
            if row.source_language_code == row.language_code:
                row_dict.update(
                    _openrouter_result_metadata(
                        "translation",
                        OpenRouterResult(text=suggestion),
                    )
                )
            else:
                row_dict.update(_openrouter_result_metadata("translation", result))
            return row_dict

    tasks = [asyncio.create_task(translate_one(row)) for row in pending_rows]
    for task in asyncio.as_completed(tasks):
        row_dict = await task
        completed_rows.append(row_dict)
        _save_checkpoint(pd.DataFrame(completed_rows), checkpoint_path)
        new_completed += 1

        if show_progress:
            completed = resumed_count + new_completed
            elapsed = time.monotonic() - started
            remaining_seconds = _estimate_remaining_seconds(
                elapsed_seconds=elapsed,
                completed_in_run=new_completed,
                remaining_in_run=remaining_total - new_completed,
                concurrency=translation_model.concurrency,
            )
            print(
                f"Translations: {completed}/{total} complete "
                f"(elapsed {elapsed:.0f}s, eta {_format_eta(remaining_seconds)})"
            )

    ordered_rows = []
    for row in matrix.itertuples(index=False):
        key = (str(row.question_id), str(row.language_code))
        match = next(
            item
            for item in completed_rows
            if str(item.get("question_id")) == key[0] and str(item.get("language_code")) == key[1]
        )
        ordered_rows.append(match)
    return pd.DataFrame(ordered_rows)


def _translation_export_frame(translations: pd.DataFrame) -> pd.DataFrame:
    export_frame = translations.copy()
    export_frame["translation_key"] = (
        export_frame["question_id"].astype(str) + "::" + export_frame["language_code"].astype(str)
    )
    export_frame["translation_final"] = export_frame["translation_suggestion"]
    export_frame["translation_review_status"] = ""
    export_frame["translation_reviewer"] = ""
    export_frame["translation_notes"] = ""

    preferred_columns = [
        "translation_key",
        "question_id",
        "language_code",
        "language_name",
        "category",
        "topic",
        "risk_level",
        "question_text",
        "reference_answer",
        "notes",
        "translation_suggestion",
        "translation_final",
        "translation_review_status",
        "translation_reviewer",
        "translation_notes",
    ]
    available_columns = [column for column in preferred_columns if column in export_frame.columns]
    return export_frame[available_columns]


def export_translation_review_workbooks(
    translations: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    output_path = Path(output_dir)
    generated_dir = output_path / "generated"
    completed_dir = output_path / "completed"
    generated_dir.mkdir(parents=True, exist_ok=True)
    completed_dir.mkdir(parents=True, exist_ok=True)

    instructions = pd.DataFrame(
        {
            "step": [
                "Review the suggested translation for one language only.",
                (
                    "Edit only the yellow columns: translation_final, "
                    "translation_review_status, translation_reviewer, "
                    "and translation_notes."
                ),
                "Use translation_review_status values like approved or needs_changes.",
                "Save the completed file into the completed folder for this language group.",
            ]
        }
    )

    export_paths: list[Path] = []
    editable_columns = {
        "translation_final",
        "translation_review_status",
        "translation_reviewer",
        "translation_notes",
    }
    dropdown_map = {
        "translation_review_status": ["approved", "needs_changes"],
    }

    for language_code, language_frame in translations.groupby("language_code", dropna=False):
        export_frame = _translation_export_frame(language_frame)
        file_path = generated_dir / f"translation_review_{language_code}.xlsx"
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            _write_excel_sheet(export_frame, writer, "translations")
            _write_excel_sheet(instructions, writer, "instructions")
        _finalize_workbook(file_path, "translations", editable_columns, dropdown_map)
        export_paths.append(file_path)

    return export_paths


def load_reviewed_translations(
    completed_dir: str | Path,
    *,
    current_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    completed_path = Path(completed_dir)
    if not completed_path.exists():
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for workbook_path in sorted(completed_path.glob("*.xlsx")):
        if workbook_path.name.startswith("~$"):
            continue
        frame = _read_review_workbook_sheet(workbook_path, sheet_name="translations")
        frame["translation_review_file"] = workbook_path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True)
    if current_matrix is not None:
        # Stable IDs alone cannot establish that a review covers the current wording.
        identity = ["question_id", "language_code", "question_text"]
        missing = set(identity) - set(frame.columns)
        if missing:
            raise ValueError(f"Translation reviews are missing context columns: {sorted(missing)}")
        frame = frame.merge(current_matrix[identity].drop_duplicates(), on=identity, how="inner")
    required_columns = {
        "question_id",
        "language_code",
        "translation_final",
        "translation_review_status",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(
            "Completed translation review files are missing required columns: "
            + ", ".join(missing_columns)
        )

    review_status = (
        frame["translation_review_status"].fillna("").astype(str).str.strip().str.casefold()
    )
    approved = frame.loc[review_status == "approved"].copy()
    if approved.empty:
        return approved

    final_text = approved["translation_final"].fillna("").astype(str).str.strip()
    approved = approved.loc[final_text != ""].copy()
    if approved.empty:
        return approved

    approved["translated_question"] = final_text.loc[approved.index]

    key_columns = ["question_id", "language_code"]
    conflicts = (
        approved.groupby(key_columns, dropna=False)["translated_question"]
        .nunique(dropna=False)
        .loc[lambda counts: counts > 1]
    )
    if not conflicts.empty:
        conflict_preview = ", ".join(
            f"{question_id}/{language_code}" for question_id, language_code in conflicts.index[:10]
        )
        raise ValueError(
            "Conflicting approved translations found for the same question/language "
            f"in completed review files: {conflict_preview}. Keep one approved "
            "translation for each pair."
        )

    return approved.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)


async def generate_answers(
    translations: pd.DataFrame,
    *,
    models: Iterable[ModelConfig],
    system_prompt: str | None = None,
    repeats_per_prompt: int = 1,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    force_restart: bool = False,
) -> pd.DataFrame:
    model_list = list(models)
    requests = pd.DataFrame(
        [
            {**row, "model_id": model.model_id, "repeat_index": repeat_index}
            for row in translations.to_dict(orient="records")
            for model in model_list
            for repeat_index in range(1, repeats_per_prompt + 1)
        ]
    )
    return await generate_answer_requests(
        requests,
        models=model_list,
        system_prompt=system_prompt,
        show_progress=show_progress,
        checkpoint_path=checkpoint_path,
        force_restart=force_restart,
    )


async def generate_answer_requests(
    requests: pd.DataFrame,
    *,
    models: Iterable[ModelConfig],
    system_prompt: str | None = None,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    force_restart: bool = False,
) -> pd.DataFrame:
    """Generate exactly the supplied question/language/model/repeat rows, in input order."""
    if requests.empty:
        return pd.DataFrame()
    keys = ["question_id", "language_code", "model_id", "repeat_index"]
    required = {*keys, "translated_question"}
    if missing := required - set(requests.columns):
        raise ValueError(f"Answer requests are missing columns: {sorted(missing)}")
    if requests.duplicated(keys).any():
        raise ValueError("Duplicate answer requests")
    if requests["translated_question"].fillna("").str.strip().eq("").any():
        raise ValueError("Answer requests contain blank translated questions")
    model_lookup = {model.model_id: model for model in models}
    if unknown := set(requests["model_id"]) - set(model_lookup):
        raise ValueError(f"No model configuration for: {sorted(unknown)}")
    signatures = {
        model_id: model.request_signature(system_prompt) for model_id, model in model_lookup.items()
    }
    existing = pd.DataFrame() if force_restart else _load_checkpoint(checkpoint_path)
    completed_lookup = {}
    for row in existing.to_dict(orient="records"):
        if row.get("answer_request_signature") == signatures.get(row.get("model_id")):
            if row.get("answer_request_signature") and str(row.get("answer_text", "")).strip():
                completed_lookup[tuple(row.get(key) for key in keys)] = row

    answers = {}
    pending = []
    for row in requests.to_dict(orient="records"):
        key = tuple(row[column] for column in keys)
        previous = completed_lookup.get(key)
        if previous and previous.get("translated_question") == row["translated_question"]:
            # Refresh source metadata without losing the saved answer and usage metadata.
            answers[key] = {
                **row,
                "response_key": previous["response_key"],
                "candidate_name": model_lookup[row["model_id"]].name,
                **{name: value for name, value in previous.items() if name.startswith("answer_")},
            }
        else:
            pending.append(row)

    total = len(requests)
    resumed_count = len(answers)
    if show_progress and resumed_count:
        print(f"Resuming answers: {resumed_count} saved, {len(pending)} remaining.")
    max_concurrency = (
        max(1, min(model_lookup[row["model_id"]].concurrency for row in pending)) if pending else 1
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    started = time.monotonic()
    failed = asyncio.Event()

    async def answer_one(row):
        model = model_lookup[row["model_id"]]
        async with semaphore:
            if failed.is_set():
                return None
            try:
                result = await _call_openrouter_result_async(
                    model=model,
                    prompt=row["translated_question"],
                    system_prompt=system_prompt,
                )
            except Exception:
                failed.set()
                raise
            return {
                **row,
                "response_key": "::".join(str(row[key]) for key in keys),
                "candidate_name": model.name,
                "answer_request_signature": signatures[model.model_id],
                "answer_text": result.text,
                **_openrouter_result_metadata("answer", result),
            }

    tasks = [asyncio.create_task(answer_one(row)) for row in pending]
    errors = []
    new_completed = 0
    try:
        for task in asyncio.as_completed(tasks):
            try:
                answer_row = await task
            except Exception as error:
                errors.append(error)
                continue
            if answer_row is None:
                continue
            answers[tuple(answer_row[key] for key in keys)] = answer_row
            _save_checkpoint(pd.DataFrame(answers.values()), checkpoint_path)
            new_completed += 1
            if show_progress:
                elapsed = time.monotonic() - started
                remaining_seconds = _estimate_remaining_seconds(
                    elapsed_seconds=elapsed,
                    completed_in_run=new_completed,
                    remaining_in_run=len(pending) - new_completed,
                    concurrency=max_concurrency,
                )
                print(
                    f"Answers: {resumed_count + new_completed}/{total} complete "
                    f"(elapsed {elapsed:.0f}s, eta {_format_eta(remaining_seconds)})"
                )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    if errors:
        raise errors[0]
    return pd.DataFrame(
        [answers[tuple(row[key] for key in keys)] for row in requests.to_dict(orient="records")]
    )


def _response_export_frame(answers: pd.DataFrame, review_schema: pd.DataFrame) -> pd.DataFrame:
    frame = answers.copy()
    for row in review_schema.itertuples(index=False):
        frame[f"review__{row.field_key}"] = ""
    frame["reviewer_name"] = ""
    frame["review_status"] = ""
    frame["review_notes"] = ""

    preferred_columns = [
        "response_key",
        "question_id",
        "language_code",
        "language_name",
        "category",
        "topic",
        "risk_level",
        "candidate_name",
        "model_id",
        "repeat_index",
        "question_text",
        "translated_question",
        "reference_answer",
        "answer_text",
        "reviewer_name",
        "review_status",
    ]
    review_columns = [f"review__{row.field_key}" for row in review_schema.itertuples(index=False)]
    trailing_columns = ["review_notes"]
    ordered = [column for column in preferred_columns if column in frame.columns]
    ordered += [column for column in review_columns if column in frame.columns]
    ordered += [column for column in trailing_columns if column in frame.columns]
    return frame[ordered]


def export_review_workbooks(
    answers: pd.DataFrame,
    *,
    review_schema: pd.DataFrame,
    output_dir: str | Path,
    review_type: str,
) -> list[Path]:
    output_path = Path(output_dir)
    generated_dir = output_path / "generated"
    completed_dir = output_path / "completed"
    generated_dir.mkdir(parents=True, exist_ok=True)
    completed_dir.mkdir(parents=True, exist_ok=True)

    response_frame = _response_export_frame(answers, review_schema)
    instructions = pd.DataFrame(
        {
            "step": [
                f"Review the {review_type} rubric columns in the responses sheet.",
                (
                    "Each row has a stable response_key so multiple reviewers "
                    "can assess the same answer independently."
                ),
                (
                    "Enter your name in reviewer_name and use the dropdown "
                    "fields for the rubric selections."
                ),
                "Leave the original question, translated question, and answer text unchanged.",
                "Long text continues in adjacent __part_2, __part_3, etc. columns; read all parts.",
                "Save completed files into the completed folder so the notebook can import them.",
            ]
        }
    )

    export_paths: list[Path] = []
    editable_columns = {"reviewer_name", "review_status", "review_notes"}
    dropdown_map = {"review_status": ["completed", "partial", "needs_follow_up"]}
    for row in review_schema.itertuples(index=False):
        column_name = f"review__{row.field_key}"
        editable_columns.add(column_name)
        dropdown_map[column_name] = [option.strip() for option in row.options.split("|")]

    master_path = generated_dir / f"{review_type}_all_languages.xlsx"
    with pd.ExcelWriter(master_path, engine="openpyxl") as writer:
        _write_excel_sheet(response_frame, writer, "responses")
        _write_excel_sheet(review_schema, writer, "criteria")
        _write_excel_sheet(instructions, writer, "instructions")
    _finalize_workbook(
        master_path,
        "responses",
        editable_columns,
        dropdown_map,
        fixed_widths=REVIEW_TEXT_COLUMN_WIDTHS,
        row_height_text_columns=["question_text", "translated_question", "answer_text"],
    )
    export_paths.append(master_path)

    for language_code, language_frame in response_frame.groupby("language_code", dropna=False):
        file_path = generated_dir / f"{review_type}_{language_code}.xlsx"
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            _write_excel_sheet(language_frame, writer, "responses")
            _write_excel_sheet(review_schema, writer, "criteria")
            _write_excel_sheet(instructions, writer, "instructions")
        _finalize_workbook(
            file_path,
            "responses",
            editable_columns,
            dropdown_map,
            fixed_widths=REVIEW_TEXT_COLUMN_WIDTHS,
            row_height_text_columns=["question_text", "translated_question", "answer_text"],
        )
        export_paths.append(file_path)

    return export_paths


def load_completed_reviews(completed_dir: str | Path) -> pd.DataFrame:
    completed_path = Path(completed_dir)
    if not completed_path.exists():
        return pd.DataFrame()

    review_frames: list[pd.DataFrame] = []
    for workbook in sorted(completed_path.glob("*.xlsx")):
        if workbook.name.startswith("~$"):
            continue
        response_frame = _read_review_workbook_sheet(workbook, sheet_name="responses")
        response_frame["review_file"] = workbook.name
        review_frames.append(response_frame)

    if not review_frames:
        return pd.DataFrame()

    return pd.concat(review_frames, ignore_index=True)


def summarize_review_answers(
    reviews: pd.DataFrame,
    *,
    review_schema: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reviews.empty:
        return pd.DataFrame(), pd.DataFrame()

    reviews = reviews.copy()
    review_columns = [f"review__{field_key}" for field_key in review_schema["field_key"]]
    for column in review_columns:
        if column not in reviews.columns:
            reviews[column] = ""

    id_vars = [
        "candidate_name",
        "model_id",
        "language_code",
        "language_name",
        "question_id",
        "repeat_index",
        "review_file",
    ]
    available_id_vars = [column for column in id_vars if column in reviews.columns]

    long_reviews = reviews.melt(
        id_vars=available_id_vars,
        value_vars=review_columns,
        var_name="review_field",
        value_name="selected_option",
    )
    long_reviews = long_reviews.dropna(subset=["selected_option"])
    long_reviews["selected_option"] = long_reviews["selected_option"].astype(str).str.strip()
    long_reviews = long_reviews[long_reviews["selected_option"] != ""]
    long_reviews["field_key"] = long_reviews["review_field"].str.removeprefix("review__")
    long_reviews = long_reviews.merge(review_schema, on="field_key", how="left")

    summary = (
        long_reviews.groupby(
            ["candidate_name", "language_code", "field_key", "field_label", "selected_option"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    totals = (
        summary.groupby(["candidate_name", "language_code", "field_key"], dropna=False)["count"]
        .sum()
        .reset_index(name="total_count")
    )
    summary = summary.merge(
        totals,
        on=["candidate_name", "language_code", "field_key"],
        how="left",
    )
    summary["share"] = summary["count"] / summary["total_count"]
    return long_reviews, summary


def model_config_records(models: Iterable[ModelConfig]) -> pd.DataFrame:
    return pd.DataFrame([asdict(model) for model in models])


def estimate_token_count(text: str | None, *, chars_per_token: float = 4.0) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return 0
    safe_chars_per_token = max(chars_per_token, 0.1)
    return max(1, math.ceil(len(normalized) / safe_chars_per_token))


def _first_available_token_average(
    frame: pd.DataFrame | None,
    columns: Sequence[str],
    *,
    chars_per_token: float,
) -> tuple[str | None, float | None]:
    if frame is None or frame.empty:
        return None, None

    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column].fillna("").astype(str).str.strip()
        series = series[series != ""]
        if series.empty:
            continue
        token_average = series.map(
            lambda value: estimate_token_count(value, chars_per_token=chars_per_token)
        ).mean()
        return column, float(token_average)

    return None, None


def _first_available_numeric_average(
    frame: pd.DataFrame | None,
    columns: Sequence[str],
) -> tuple[str | None, float | None]:
    if frame is None or frame.empty:
        return None, None

    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        return column, float(series.mean())

    return None, None


def _numeric_mapping(mapping: Mapping[str, Any] | None) -> dict[str, float]:
    if not mapping:
        return {}

    normalized: dict[str, float] = {}
    for key, value in mapping.items():
        numeric_value = _optional_float(value)
        if numeric_value is not None:
            normalized[str(key)] = numeric_value
    return normalized


def _observed_cost_total(frame: pd.DataFrame | None, column: str) -> float:
    if frame is None or frame.empty or column not in frame.columns:
        return 0.0

    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(series[series > 0].sum())


def _integer_mapping(mapping: Mapping[str, Any] | None) -> dict[str, int]:
    if not mapping:
        return {}

    normalized: dict[str, int] = {}
    for key, value in mapping.items():
        numeric_value = _optional_int(value)
        if numeric_value is not None:
            normalized[str(key)] = numeric_value
    return normalized


def _average_answer_text_tokens_by_model(
    frame: pd.DataFrame | None,
    *,
    chars_per_token: float,
) -> dict[str, float]:
    if (
        frame is None
        or frame.empty
        or "model_id" not in frame.columns
        or "answer_text" not in frame
    ):
        return {}

    token_frame = frame[["model_id", "answer_text"]].copy()
    token_frame["answer_text_tokens"] = token_frame["answer_text"].map(
        lambda value: estimate_token_count(value, chars_per_token=chars_per_token)
    )
    return {
        str(model_id): float(tokens)
        for model_id, tokens in token_frame.groupby("model_id")["answer_text_tokens"].mean().items()
    }


def fetch_openrouter_model_catalog(*, timeout_seconds: int = 30) -> pd.DataFrame:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=timeout_seconds,
    )
    response.raise_for_status()

    records: list[dict[str, Any]] = []
    for item in response.json().get("data", []):
        pricing = item.get("pricing") or {}
        prompt_cost = float(pricing.get("prompt") or 0.0) * 1_000_000
        completion_cost = float(pricing.get("completion") or 0.0) * 1_000_000
        records.append(
            {
                "model_id": str(item.get("id", "")),
                "model_name": str(item.get("name", item.get("id", ""))),
                "created": int(item.get("created", 0) or 0),
                "input_cost_per_million": prompt_cost,
                "output_cost_per_million": completion_cost,
                "pricing_source": "openrouter_catalog",
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["created", "model_id"],
        ascending=[False, True],
    ).reset_index(drop=True)


def resolve_openrouter_model_pricing(
    model_ids: Iterable[str],
    *,
    catalog: pd.DataFrame | None = None,
    price_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    unique_model_ids = list(dict.fromkeys(str(model_id) for model_id in model_ids if str(model_id)))
    if not unique_model_ids:
        return pd.DataFrame(
            columns=[
                "model_id",
                "model_name",
                "created",
                "input_cost_per_million",
                "output_cost_per_million",
                "pricing_source",
            ]
        )

    catalog_frame = (
        fetch_openrouter_model_catalog(timeout_seconds=timeout_seconds)
        if catalog is None
        else catalog
    )
    catalog_lookup: dict[str, dict[str, Any]] = {}
    if not catalog_frame.empty:
        for row in catalog_frame.to_dict(orient="records"):
            model_id = str(row.get("model_id", ""))
            if model_id in unique_model_ids and model_id not in catalog_lookup:
                catalog_lookup[model_id] = row

    merged_overrides: dict[str, Mapping[str, Any]] = dict(DEFAULT_OPENROUTER_LEGACY_PRICE_OVERRIDES)
    if price_overrides:
        merged_overrides.update(
            {str(model_id): override for model_id, override in price_overrides.items()}
        )

    resolved: dict[str, dict[str, Any]] = {}
    for model_id in unique_model_ids:
        if model_id in catalog_lookup:
            resolved[model_id] = dict(catalog_lookup[model_id])

    for model_id, override in merged_overrides.items():
        if model_id not in unique_model_ids:
            continue
        resolved[model_id] = {
            "model_id": model_id,
            "model_name": str(override.get("model_name", model_id)),
            "created": int(override.get("created", 0) or 0),
            "input_cost_per_million": float(override.get("input_cost_per_million", 0.0) or 0.0),
            "output_cost_per_million": float(override.get("output_cost_per_million", 0.0) or 0.0),
            "pricing_source": str(override.get("pricing_source", "override")),
        }

    missing = [model_id for model_id in unique_model_ids if model_id not in resolved]
    if missing:
        missing_models = ", ".join(sorted(missing))
        raise ValueError(f"Missing OpenRouter pricing for model IDs: {missing_models}")

    ordered_rows = [resolved[model_id] for model_id in unique_model_ids]
    return pd.DataFrame(ordered_rows)


def _build_cost_assumptions(
    *,
    questions: pd.DataFrame,
    translation_samples: pd.DataFrame | None,
    answer_samples: pd.DataFrame | None,
    translation_system_prompt: str | None,
    answer_system_prompt: str | None,
    chars_per_token: float,
    fallback_answer_output_tokens: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    _, question_tokens = _first_available_token_average(
        questions,
        ["question_text"],
        chars_per_token=chars_per_token,
    )
    question_tokens = float(question_tokens or 0.0)

    translated_source, translated_tokens = _first_available_token_average(
        translation_samples,
        ["translated_question", "translation_suggestion"],
        chars_per_token=chars_per_token,
    )
    if translated_tokens is None:
        translated_tokens = question_tokens
        translated_source = "question_text_fallback"

    translation_prompt_usage_source, translation_prompt_usage_tokens = (
        _first_available_numeric_average(
            translation_samples,
            ["translation_openrouter_prompt_tokens"],
        )
    )
    translation_completion_usage_source, translation_completion_usage_tokens = (
        _first_available_numeric_average(
            translation_samples,
            ["translation_openrouter_completion_tokens"],
        )
    )

    answer_output_source, answer_output_tokens = _first_available_token_average(
        answer_samples,
        ["answer_text"],
        chars_per_token=chars_per_token,
    )
    if answer_output_tokens is None:
        answer_output_tokens = float(fallback_answer_output_tokens)
        answer_output_source = "fallback_default"

    answer_prompt_usage_source, answer_prompt_usage_tokens = _first_available_numeric_average(
        answer_samples,
        ["answer_openrouter_prompt_tokens"],
    )
    answer_completion_usage_source, answer_completion_usage_tokens = (
        _first_available_numeric_average(
            answer_samples,
            ["answer_openrouter_completion_tokens"],
        )
    )

    translation_system_tokens = float(
        estimate_token_count(translation_system_prompt, chars_per_token=chars_per_token)
    )
    answer_system_tokens = float(
        estimate_token_count(answer_system_prompt, chars_per_token=chars_per_token)
    )

    assumptions = {
        "chars_per_token": float(chars_per_token),
        "question_tokens_per_request": question_tokens,
        "question_tokens_source": "question_text",
        "translated_question_tokens_per_request": float(translated_tokens),
        "translated_question_tokens_source": str(translated_source),
        "translation_prompt_tokens_per_request": float(
            translation_prompt_usage_tokens
            if translation_prompt_usage_tokens is not None
            else question_tokens + translation_system_tokens
        ),
        "translation_prompt_tokens_source": str(
            translation_prompt_usage_source or "question_text + translation_system_prompt"
        ),
        "translation_completion_tokens_per_request": float(
            translation_completion_usage_tokens
            if translation_completion_usage_tokens is not None
            else translated_tokens
        ),
        "translation_completion_tokens_source": str(
            translation_completion_usage_source or translated_source
        ),
        "answer_prompt_tokens_per_request": float(
            answer_prompt_usage_tokens
            if answer_prompt_usage_tokens is not None
            else float(translated_tokens) + answer_system_tokens
        ),
        "answer_prompt_tokens_source": str(
            answer_prompt_usage_source or "translated_question + answer_system_prompt"
        ),
        "answer_completion_tokens_per_request": float(
            answer_completion_usage_tokens
            if answer_completion_usage_tokens is not None
            else answer_output_tokens
        ),
        "answer_completion_tokens_source": str(
            answer_completion_usage_source or answer_output_source
        ),
        "translation_system_prompt_tokens": translation_system_tokens,
        "answer_system_prompt_tokens": answer_system_tokens,
    }

    assumption_rows = [
        {
            "metric": "chars_per_token",
            "estimated_tokens": assumptions["chars_per_token"],
            "source": "heuristic",
        },
        {
            "metric": "question_tokens_per_request",
            "estimated_tokens": assumptions["question_tokens_per_request"],
            "source": assumptions["question_tokens_source"],
        },
        {
            "metric": "translated_question_tokens_per_request",
            "estimated_tokens": assumptions["translated_question_tokens_per_request"],
            "source": assumptions["translated_question_tokens_source"],
        },
        {
            "metric": "translation_prompt_tokens_per_request",
            "estimated_tokens": assumptions["translation_prompt_tokens_per_request"],
            "source": assumptions["translation_prompt_tokens_source"],
        },
        {
            "metric": "translation_completion_tokens_per_request",
            "estimated_tokens": assumptions["translation_completion_tokens_per_request"],
            "source": assumptions["translation_completion_tokens_source"],
        },
        {
            "metric": "answer_prompt_tokens_per_request",
            "estimated_tokens": assumptions["answer_prompt_tokens_per_request"],
            "source": assumptions["answer_prompt_tokens_source"],
        },
        {
            "metric": "answer_completion_tokens_per_request",
            "estimated_tokens": assumptions["answer_completion_tokens_per_request"],
            "source": assumptions["answer_completion_tokens_source"],
        },
    ]
    assumptions_frame = pd.DataFrame(assumption_rows)
    assumptions_frame["estimated_tokens"] = assumptions_frame["estimated_tokens"].round(1)
    return assumptions, assumptions_frame


def estimate_openrouter_cost_scenarios(
    *,
    questions: pd.DataFrame,
    languages: pd.DataFrame,
    scenarios: Mapping[str, Sequence[str]],
    repeats_per_prompt: int = 1,
    translation_model_id: str | None = None,
    question_count: int | None = None,
    language_count: int | None = None,
    translation_samples: pd.DataFrame | None = None,
    answer_samples: pd.DataFrame | None = None,
    translation_system_prompt: str | None = DEFAULT_TRANSLATION_SYSTEM_PROMPT,
    answer_system_prompt: str | None = None,
    price_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    observed_answer_spend_by_model: Mapping[str, Any] | None = None,
    observed_answer_request_count_by_model: Mapping[str, Any] | None = None,
    chars_per_token: float = 4.0,
    fallback_answer_output_tokens: float = DEFAULT_FALLBACK_ANSWER_OUTPUT_TOKENS,
    timeout_seconds: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized_scenarios = {
        str(name): [str(model_id) for model_id in model_ids if str(model_id)]
        for name, model_ids in scenarios.items()
    }

    all_model_ids: list[str] = []
    for model_ids in normalized_scenarios.values():
        all_model_ids.extend(model_ids)
    if translation_model_id:
        all_model_ids.append(str(translation_model_id))

    pricing_frame = resolve_openrouter_model_pricing(
        all_model_ids,
        price_overrides=price_overrides,
        timeout_seconds=timeout_seconds,
    )
    pricing_lookup = {str(row["model_id"]): row for row in pricing_frame.to_dict(orient="records")}

    assumptions, assumptions_frame = _build_cost_assumptions(
        questions=questions,
        translation_samples=translation_samples,
        answer_samples=answer_samples,
        translation_system_prompt=translation_system_prompt,
        answer_system_prompt=answer_system_prompt,
        chars_per_token=chars_per_token,
        fallback_answer_output_tokens=fallback_answer_output_tokens,
    )

    scenario_question_count = int(question_count if question_count is not None else len(questions))
    scenario_language_count = int(language_count if language_count is not None else len(languages))
    requests_per_model = (
        scenario_question_count * scenario_language_count * max(1, repeats_per_prompt)
    )
    translation_request_count = (
        scenario_question_count * scenario_language_count if translation_model_id else 0
    )

    observed_answer_cost_by_model: dict[str, float] = {}
    observed_answer_spend_totals_by_model: dict[str, float] = {}
    observed_answer_counts_by_model = _integer_mapping(observed_answer_request_count_by_model)
    if (
        answer_samples is not None
        and not answer_samples.empty
        and {"model_id", "answer_openrouter_cost_usd"}.issubset(answer_samples.columns)
    ):
        usage_frame = answer_samples[["model_id", "answer_openrouter_cost_usd"]].copy()
        usage_frame["answer_openrouter_cost_usd"] = pd.to_numeric(
            usage_frame["answer_openrouter_cost_usd"],
            errors="coerce",
        )
        usage_frame = usage_frame.dropna(subset=["answer_openrouter_cost_usd"])
        usage_frame = usage_frame[usage_frame["answer_openrouter_cost_usd"] > 0]
        if not usage_frame.empty:
            observed_answer_cost_by_model = {
                str(model_id): float(cost)
                for model_id, cost in usage_frame.groupby("model_id")["answer_openrouter_cost_usd"]
                .mean()
                .items()
            }
            observed_answer_spend_totals_by_model = {
                str(model_id): float(cost)
                for model_id, cost in usage_frame.groupby("model_id")["answer_openrouter_cost_usd"]
                .sum()
                .items()
            }
            observed_answer_counts_by_model.update(
                {
                    str(model_id): int(count)
                    for model_id, count in usage_frame.groupby("model_id").size().items()
                }
            )

    explicit_observed_spend_by_model = _numeric_mapping(observed_answer_spend_by_model)
    if explicit_observed_spend_by_model:
        observed_answer_counts_by_model.update(
            _integer_mapping(observed_answer_request_count_by_model)
        )
        observed_answer_spend_totals_by_model.update(explicit_observed_spend_by_model)
        if answer_samples is not None and not answer_samples.empty and "model_id" in answer_samples:
            sample_counts = {
                str(model_id): int(count)
                for model_id, count in answer_samples.groupby("model_id").size().items()
            }
        else:
            sample_counts = {}

        for model_id, total_spend in explicit_observed_spend_by_model.items():
            observed_count = observed_answer_counts_by_model.get(model_id) or sample_counts.get(
                model_id
            )
            if observed_count:
                observed_answer_counts_by_model[model_id] = observed_count
                observed_answer_cost_by_model[model_id] = total_spend / observed_count

    visible_answer_tokens_by_model = _average_answer_text_tokens_by_model(
        answer_samples,
        chars_per_token=chars_per_token,
    )

    translation_cost = 0.0
    translation_price_source = ""
    translation_cost_estimation_source = ""
    if translation_model_id:
        observed_translation_average_cost = None
        if (
            translation_samples is not None
            and not translation_samples.empty
            and "translation_openrouter_cost_usd" in translation_samples.columns
        ):
            observed_translation_costs = pd.to_numeric(
                translation_samples["translation_openrouter_cost_usd"],
                errors="coerce",
            ).dropna()
            observed_translation_costs = observed_translation_costs[observed_translation_costs > 0]
            if not observed_translation_costs.empty:
                observed_translation_average_cost = float(observed_translation_costs.mean())

        translation_price = pricing_lookup[str(translation_model_id)]
        translation_input_cost = (
            translation_request_count
            * assumptions["translation_prompt_tokens_per_request"]
            * float(translation_price["input_cost_per_million"])
            / 1_000_000
        )
        translation_output_cost = (
            translation_request_count
            * assumptions["translation_completion_tokens_per_request"]
            * float(translation_price["output_cost_per_million"])
            / 1_000_000
        )
        if observed_translation_average_cost is not None:
            translation_cost = translation_request_count * observed_translation_average_cost
            translation_cost_estimation_source = "observed_openrouter_usage_cost"
        else:
            translation_cost = translation_input_cost + translation_output_cost
            translation_cost_estimation_source = "catalog_price_x_estimated_tokens"
        translation_price_source = str(translation_price["pricing_source"])

    summary_rows: list[dict[str, Any]] = []
    per_model_rows: list[dict[str, Any]] = []
    for scenario_name, model_ids in normalized_scenarios.items():
        answer_cost_total = 0.0
        for model_id in model_ids:
            price = pricing_lookup[model_id]
            estimated_input_cost_per_request = (
                assumptions["answer_prompt_tokens_per_request"]
                * float(price["input_cost_per_million"])
                / 1_000_000
            )
            estimated_input_cost = (
                requests_per_model
                * assumptions["answer_prompt_tokens_per_request"]
                * float(price["input_cost_per_million"])
                / 1_000_000
            )
            estimated_output_cost = (
                requests_per_model
                * assumptions["answer_completion_tokens_per_request"]
                * float(price["output_cost_per_million"])
                / 1_000_000
            )
            observed_average_cost = observed_answer_cost_by_model.get(model_id)
            if observed_average_cost is not None:
                estimated_total_cost = requests_per_model * observed_average_cost
                cost_estimation_source = "observed_openrouter_usage_cost"
            else:
                estimated_total_cost = estimated_input_cost + estimated_output_cost
                cost_estimation_source = "catalog_price_x_estimated_tokens"
            observed_total_spend = observed_answer_spend_totals_by_model.get(model_id)
            observed_request_count = observed_answer_counts_by_model.get(model_id)
            visible_completion_tokens = visible_answer_tokens_by_model.get(
                model_id,
                float(assumptions["answer_completion_tokens_per_request"]),
            )
            implied_billed_output_tokens = None
            implied_hidden_output_tokens = None
            if observed_average_cost is not None and float(price["output_cost_per_million"]) > 0:
                output_cost_per_million = float(price["output_cost_per_million"])
                output_cost_after_input = max(
                    observed_average_cost - estimated_input_cost_per_request,
                    0.0,
                )
                implied_billed_output_tokens = (
                    output_cost_after_input * 1_000_000 / output_cost_per_million
                )
                implied_hidden_output_tokens = max(
                    implied_billed_output_tokens - visible_completion_tokens,
                    0.0,
                )
            answer_cost_total += estimated_total_cost
            per_model_rows.append(
                {
                    "scenario_name": scenario_name,
                    "model_id": model_id,
                    "model_name": price["model_name"],
                    "pricing_source": price["pricing_source"],
                    "request_count": requests_per_model,
                    "input_cost_per_million": float(price["input_cost_per_million"]),
                    "output_cost_per_million": float(price["output_cost_per_million"]),
                    "estimated_input_cost_usd": estimated_input_cost,
                    "estimated_output_cost_usd": estimated_output_cost,
                    "catalog_token_estimated_total_cost_usd": (
                        estimated_input_cost + estimated_output_cost
                    ),
                    "estimated_total_cost_usd": estimated_total_cost,
                    "observed_total_spend_usd": observed_total_spend,
                    "observed_request_count": observed_request_count,
                    "observed_average_cost_per_request_usd": observed_average_cost,
                    "visible_completion_tokens_per_request": visible_completion_tokens,
                    "implied_billed_output_tokens_per_request": implied_billed_output_tokens,
                    "implied_hidden_or_reasoning_tokens_per_request": implied_hidden_output_tokens,
                    "cost_estimation_source": cost_estimation_source,
                }
            )

        summary_rows.append(
            {
                "scenario_name": scenario_name,
                "question_count": scenario_question_count,
                "language_count": scenario_language_count,
                "model_count": len(model_ids),
                "repeats_per_prompt": max(1, repeats_per_prompt),
                "translation_model_id": str(translation_model_id or ""),
                "translation_pricing_source": translation_price_source,
                "translation_request_count": translation_request_count,
                "answer_request_count": requests_per_model * len(model_ids),
                "translation_estimated_cost_usd": translation_cost,
                "translation_cost_estimation_source": translation_cost_estimation_source,
                "answer_estimated_cost_usd": answer_cost_total,
                "total_estimated_cost_usd": translation_cost + answer_cost_total,
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    per_model_frame = pd.DataFrame(per_model_rows)

    for frame in [summary_frame, per_model_frame]:
        for column in frame.columns:
            if column.endswith("_usd") or column.endswith("_per_million"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce").round(4)
            elif column.endswith("_tokens_per_request"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce").round(1)

    return assumptions_frame, summary_frame, per_model_frame


def build_budget_recommendations(
    cost_summary: pd.DataFrame,
    *,
    extra_answer_iterations: Sequence[int] = (0, 1, 2, 3),
    round_up_to_usd: float = 5.0,
    recommended_extra_iterations: int = 2,
    safety_multiplier: float = 1.5,
) -> pd.DataFrame:
    if cost_summary.empty:
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "budget_tier",
                "extra_answer_iterations",
                "target_repeats_per_prompt",
                "estimated_total_cost_usd",
                "safety_multiplier",
                "safety_adjusted_total_cost_usd",
                "communicated_budget_usd",
                "recommended_to_communicate",
            ]
        )

    safe_round_increment = max(float(round_up_to_usd), 0.01)
    rows: list[dict[str, Any]] = []
    for row in cost_summary.to_dict(orient="records"):
        base_repeats = max(1, int(row.get("repeats_per_prompt", 1) or 1))
        translation_cost = float(row.get("translation_estimated_cost_usd", 0.0) or 0.0)
        answer_cost = float(row.get("answer_estimated_cost_usd", 0.0) or 0.0)

        for extra_iterations in extra_answer_iterations:
            extra_iteration_count = max(0, int(extra_iterations))
            target_repeats = base_repeats + extra_iteration_count
            scaled_answer_cost = answer_cost * (target_repeats / base_repeats)
            estimated_total_cost = translation_cost + scaled_answer_cost
            safe_total_cost = estimated_total_cost * max(float(safety_multiplier), 1.0)
            communicated_budget = (
                math.ceil(safe_total_cost / safe_round_increment) * safe_round_increment
            )

            if extra_iteration_count == 0:
                budget_tier = "Base plan"
            elif extra_iteration_count == 1:
                budget_tier = "+1 extra iteration"
            elif extra_iteration_count == recommended_extra_iterations:
                budget_tier = "Safe budget"
            else:
                budget_tier = f"Conservative (+{extra_iteration_count} extra iterations)"

            rows.append(
                {
                    "scenario_name": str(row.get("scenario_name", "")),
                    "budget_tier": budget_tier,
                    "extra_answer_iterations": extra_iteration_count,
                    "target_repeats_per_prompt": target_repeats,
                    "estimated_total_cost_usd": estimated_total_cost,
                    "safety_multiplier": max(float(safety_multiplier), 1.0),
                    "safety_adjusted_total_cost_usd": safe_total_cost,
                    "communicated_budget_usd": communicated_budget,
                    "recommended_to_communicate": (
                        extra_iteration_count == recommended_extra_iterations
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame["estimated_total_cost_usd"] = frame["estimated_total_cost_usd"].round(4)
    frame["safety_adjusted_total_cost_usd"] = frame["safety_adjusted_total_cost_usd"].round(4)
    frame["communicated_budget_usd"] = frame["communicated_budget_usd"].round(2)
    return frame


def summarize_openrouter_activity(
    activity: pd.DataFrame | None,
    *,
    model_aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    columns = [
        "model_id",
        "observed_spend_usd",
        "observed_requests",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "observed_spend_per_request_usd",
        "completion_tokens_per_request",
        "reasoning_tokens_per_request",
        "reasoning_share_of_completion_tokens",
    ]
    if activity is None or activity.empty:
        return pd.DataFrame(columns=columns)

    required_columns = {
        "Slug",
        "Usage",
        "Requests",
        "Prompt Tokens",
        "Completion Tokens",
        "Reasoning Tokens",
    }
    missing_columns = sorted(required_columns - set(activity.columns))
    if missing_columns:
        raise ValueError(
            "OpenRouter activity export is missing required columns: " + ", ".join(missing_columns)
        )

    frame = activity.copy()
    if model_aliases:
        normalized_aliases = {str(source): str(target) for source, target in model_aliases.items()}
        frame["Slug"] = frame["Slug"].astype(str).replace(normalized_aliases)

    numeric_columns = [
        "Usage",
        "Requests",
        "Prompt Tokens",
        "Completion Tokens",
        "Reasoning Tokens",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    summary = (
        frame.groupby("Slug", as_index=False)
        .agg(
            observed_spend_usd=("Usage", "sum"),
            observed_requests=("Requests", "sum"),
            prompt_tokens=("Prompt Tokens", "sum"),
            completion_tokens=("Completion Tokens", "sum"),
            reasoning_tokens=("Reasoning Tokens", "sum"),
        )
        .rename(columns={"Slug": "model_id"})
    )

    safe_requests = summary["observed_requests"].replace(0, pd.NA)
    safe_completion_tokens = summary["completion_tokens"].replace(0, pd.NA)
    summary["observed_spend_per_request_usd"] = summary["observed_spend_usd"] / safe_requests
    summary["completion_tokens_per_request"] = summary["completion_tokens"] / safe_requests
    summary["reasoning_tokens_per_request"] = summary["reasoning_tokens"] / safe_requests
    summary["reasoning_share_of_completion_tokens"] = (
        summary["reasoning_tokens"] / safe_completion_tokens
    )

    for column in [
        "observed_spend_usd",
        "observed_spend_per_request_usd",
        "completion_tokens_per_request",
        "reasoning_tokens_per_request",
        "reasoning_share_of_completion_tokens",
    ]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").round(4)
    for column in [
        "observed_requests",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
    ]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce").fillna(0).astype(int)

    return summary.sort_values("observed_spend_usd", ascending=False).reset_index(drop=True)


def _ceil_to_increment(value: float, increment: float) -> float:
    safe_increment = max(float(increment), 0.01)
    return math.ceil(float(value) / safe_increment) * safe_increment


def _budget_rows_from_summary(
    *,
    summary_frame: pd.DataFrame,
    scenario_models: Mapping[str, Sequence[str]],
    observed_spend_by_model: Mapping[str, float],
    safety_multiplier: float,
    round_up_to_usd: float,
    note: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary_frame.to_dict(orient="records"):
        scenario_name = str(row.get("scenario_name", ""))
        model_ids = [str(model_id) for model_id in scenario_models.get(scenario_name, [])]
        observed_spend = sum(
            float(observed_spend_by_model.get(model_id, 0.0)) for model_id in model_ids
        )
        projected_total = float(row.get("total_estimated_cost_usd", 0.0) or 0.0)
        safety_adjusted_total = projected_total * max(float(safety_multiplier), 1.0)
        rows.append(
            {
                "scope": scenario_name,
                "safety_multiplier": max(float(safety_multiplier), 1.0),
                "observed_spend_applied_usd": observed_spend,
                "projected_total_answer_requests": int(row.get("answer_request_count", 0) or 0),
                "projected_total_spend_usd": projected_total,
                "safety_adjusted_total_spend_usd": safety_adjusted_total,
                "projected_remaining_spend_usd": max(projected_total - observed_spend, 0.0),
                "communicated_budget_usd": _ceil_to_increment(
                    safety_adjusted_total,
                    round_up_to_usd,
                ),
                "model_ids": ", ".join(model_ids),
                "note": note,
            }
        )
    return rows


def _focused_budget_table(
    calibrated_budgets: pd.DataFrame,
    budget_recommendations: pd.DataFrame,
) -> pd.DataFrame:
    if not calibrated_budgets.empty:
        columns = [
            "scope",
            "observed_spend_applied_usd",
            "projected_total_spend_usd",
            "safety_multiplier",
            "safety_adjusted_total_spend_usd",
            "projected_remaining_spend_usd",
            "communicated_budget_usd",
        ]
        return calibrated_budgets[[column for column in columns if column in calibrated_budgets]]

    if budget_recommendations.empty:
        return pd.DataFrame(
            columns=[
                "scope",
                "target_repeats_per_prompt",
                "projected_total_spend_usd",
                "safety_multiplier",
                "safety_adjusted_total_spend_usd",
                "communicated_budget_usd",
            ]
        )

    focused = budget_recommendations.loc[
        budget_recommendations["recommended_to_communicate"]
    ].copy()
    focused = focused.rename(
        columns={
            "scenario_name": "scope",
            "estimated_total_cost_usd": "projected_total_spend_usd",
            "safety_adjusted_total_cost_usd": "safety_adjusted_total_spend_usd",
        }
    )
    return focused[
        [
            "scope",
            "target_repeats_per_prompt",
            "projected_total_spend_usd",
            "safety_multiplier",
            "safety_adjusted_total_spend_usd",
            "communicated_budget_usd",
        ]
    ]


def _build_budget_overview(
    *,
    questions: pd.DataFrame,
    languages: pd.DataFrame,
    models: Sequence[ModelConfig],
    repeats_per_prompt: int,
    approved_translations: pd.DataFrame | None,
    answer_samples: pd.DataFrame | None,
    activity_summary: pd.DataFrame,
    recommended_budgets: pd.DataFrame,
    observed_pipeline_spend: float = 0.0,
) -> pd.DataFrame:
    configured_answer_requests = (
        len(questions) * len(languages) * len(models) * max(1, repeats_per_prompt)
    )
    approved_translation_count = (
        0
        if approved_translations is None or approved_translations.empty
        else len(approved_translations)
    )
    approved_answer_requests = approved_translation_count * len(models) * max(1, repeats_per_prompt)
    saved_answer_count = (
        0 if answer_samples is None or answer_samples.empty else len(answer_samples)
    )
    observed_spend = (
        0.0 if activity_summary.empty else float(activity_summary["observed_spend_usd"].sum())
    )
    observed_requests = (
        0 if activity_summary.empty else int(activity_summary["observed_requests"].sum())
    )
    recommended_budget = None
    if not recommended_budgets.empty and "communicated_budget_usd" in recommended_budgets:
        recommended_budget = float(recommended_budgets["communicated_budget_usd"].max())

    rows = [
        {"metric": "questions", "value": len(questions)},
        {"metric": "languages", "value": len(languages)},
        {"metric": "answer_models", "value": len(models)},
        {"metric": "repeats_per_prompt", "value": max(1, repeats_per_prompt)},
        {"metric": "full_configured_answer_requests", "value": configured_answer_requests},
        {"metric": "approved_translations", "value": approved_translation_count},
        {"metric": "approved_answer_requests", "value": approved_answer_requests},
        {"metric": "saved_answer_rows", "value": saved_answer_count},
        {"metric": "observed_openrouter_requests", "value": observed_requests},
        {"metric": "observed_openrouter_spend_usd", "value": round(observed_spend, 4)},
        {
            "metric": "observed_pipeline_metadata_spend_usd",
            "value": round(float(observed_pipeline_spend), 4),
        },
    ]
    if recommended_budget is not None:
        rows.append(
            {
                "metric": "max_communicated_budget_usd",
                "value": round(recommended_budget, 2),
            }
        )
    return pd.DataFrame(rows)


def build_openrouter_budget_report(
    *,
    questions: pd.DataFrame,
    languages: pd.DataFrame,
    models: Iterable[ModelConfig],
    translation_model: ModelConfig,
    repeats_per_prompt: int = 1,
    translation_samples: pd.DataFrame | None = None,
    answer_samples: pd.DataFrame | None = None,
    approved_translations: pd.DataFrame | None = None,
    openrouter_activity: pd.DataFrame | None = None,
    openrouter_activity_model_aliases: Mapping[str, str] | None = None,
    translation_system_prompt: str | None = DEFAULT_TRANSLATION_SYSTEM_PROMPT,
    answer_system_prompt: str | None = None,
    excluded_model_scenarios: Mapping[str, Sequence[str]] | None = None,
    extra_answer_iterations: Sequence[int] = (0,),
    recommended_extra_iterations: int = 0,
    budget_round_up_to_usd: float = 5.0,
    calibrated_safety_multiplier: float = 1.5,
    calibrated_round_up_to_usd: float = 10.0,
    timeout_seconds: int = 30,
) -> BudgetReport:
    model_list = list(models)
    configured_model_ids = [model.model_id for model in model_list]
    scenario_models: dict[str, list[str]] = {
        "Configured study setup": configured_model_ids,
    }
    for scenario_name, excluded_model_ids in (excluded_model_scenarios or {}).items():
        excluded = {str(model_id) for model_id in excluded_model_ids}
        scenario_models[str(scenario_name)] = [
            model_id for model_id in configured_model_ids if model_id not in excluded
        ]

    activity_summary = summarize_openrouter_activity(
        openrouter_activity,
        model_aliases=openrouter_activity_model_aliases,
    )
    observed_spend_by_model = (
        {}
        if activity_summary.empty
        else dict(zip(activity_summary["model_id"], activity_summary["observed_spend_usd"]))
    )
    observed_requests_by_model = (
        {}
        if activity_summary.empty
        else dict(zip(activity_summary["model_id"], activity_summary["observed_requests"]))
    )

    assumptions, scenario_summary, per_model_estimates = estimate_openrouter_cost_scenarios(
        questions=questions,
        languages=languages,
        scenarios=scenario_models,
        repeats_per_prompt=repeats_per_prompt,
        translation_model_id=translation_model.model_id,
        translation_system_prompt=translation_system_prompt,
        translation_samples=translation_samples,
        answer_samples=answer_samples,
        answer_system_prompt=answer_system_prompt,
        observed_answer_spend_by_model=observed_spend_by_model,
        observed_answer_request_count_by_model=observed_requests_by_model,
        timeout_seconds=timeout_seconds,
    )
    budget_iterations = build_budget_recommendations(
        scenario_summary,
        extra_answer_iterations=extra_answer_iterations,
        round_up_to_usd=budget_round_up_to_usd,
        recommended_extra_iterations=recommended_extra_iterations,
        safety_multiplier=calibrated_safety_multiplier,
    )

    approved_count = (
        0
        if approved_translations is None or approved_translations.empty
        else len(approved_translations)
    )
    _, current_summary, current_per_model = estimate_openrouter_cost_scenarios(
        questions=questions,
        languages=languages,
        scenarios={"current approved translations only": configured_model_ids},
        repeats_per_prompt=repeats_per_prompt,
        translation_model_id=translation_model.model_id,
        question_count=approved_count,
        language_count=1,
        translation_system_prompt=translation_system_prompt,
        translation_samples=translation_samples,
        answer_samples=answer_samples,
        answer_system_prompt=answer_system_prompt,
        observed_answer_spend_by_model=observed_spend_by_model,
        observed_answer_request_count_by_model=observed_requests_by_model,
        timeout_seconds=timeout_seconds,
    )

    calibrated_budget_rows = []
    if observed_spend_by_model:
        calibrated_budget_rows.extend(
            _budget_rows_from_summary(
                summary_frame=current_summary,
                scenario_models={"current approved translations only": configured_model_ids},
                observed_spend_by_model=observed_spend_by_model,
                safety_multiplier=calibrated_safety_multiplier,
                round_up_to_usd=calibrated_round_up_to_usd,
                note="OpenRouter activity export",
            )
        )
        calibrated_budget_rows.extend(
            _budget_rows_from_summary(
                summary_frame=scenario_summary,
                scenario_models=scenario_models,
                observed_spend_by_model=observed_spend_by_model,
                safety_multiplier=calibrated_safety_multiplier,
                round_up_to_usd=calibrated_round_up_to_usd,
                note="OpenRouter activity export",
            )
        )

    calibrated_budgets = pd.DataFrame(calibrated_budget_rows)
    for column in [
        "observed_spend_applied_usd",
        "projected_total_spend_usd",
        "safety_adjusted_total_spend_usd",
        "projected_remaining_spend_usd",
        "communicated_budget_usd",
    ]:
        if column in calibrated_budgets.columns:
            calibrated_budgets[column] = pd.to_numeric(
                calibrated_budgets[column],
                errors="coerce",
            ).round(4)

    recommended_budgets = _focused_budget_table(calibrated_budgets, budget_iterations)

    configured_per_model = per_model_estimates.loc[
        per_model_estimates["scenario_name"] == "Configured study setup"
    ].copy()
    model_cost_drivers = configured_per_model.merge(
        activity_summary,
        on="model_id",
        how="left",
        suffixes=("", "_activity"),
    )
    driver_columns = [
        "model_id",
        "model_name",
        "observed_spend_usd",
        "observed_requests",
        "observed_spend_per_request_usd",
        "observed_total_spend_usd",
        "observed_request_count",
        "observed_average_cost_per_request_usd",
        "completion_tokens_per_request",
        "reasoning_tokens_per_request",
        "estimated_total_cost_usd",
        "implied_hidden_or_reasoning_tokens_per_request",
        "cost_estimation_source",
    ]
    model_cost_drivers = model_cost_drivers[
        [column for column in driver_columns if column in model_cost_drivers]
    ]
    if (
        "observed_spend_usd" in model_cost_drivers
        and model_cost_drivers["observed_spend_usd"].notna().any()
    ):
        model_cost_drivers = model_cost_drivers.sort_values(
            "observed_spend_usd",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)
    elif "observed_total_spend_usd" in model_cost_drivers:
        model_cost_drivers = model_cost_drivers.sort_values(
            "observed_total_spend_usd",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    overview = _build_budget_overview(
        questions=questions,
        languages=languages,
        models=model_list,
        repeats_per_prompt=repeats_per_prompt,
        approved_translations=approved_translations,
        answer_samples=answer_samples,
        activity_summary=activity_summary,
        recommended_budgets=recommended_budgets,
        observed_pipeline_spend=(
            _observed_cost_total(translation_samples, "translation_openrouter_cost_usd")
            + _observed_cost_total(answer_samples, "answer_openrouter_cost_usd")
        ),
    )

    return BudgetReport(
        overview=overview,
        recommended_budgets=recommended_budgets,
        model_cost_drivers=model_cost_drivers,
        activity_summary=activity_summary,
        assumptions=assumptions,
        scenario_summary=scenario_summary,
        budget_iterations=budget_iterations,
        per_model_estimates=per_model_estimates,
        current_approved_summary=current_summary,
        current_approved_per_model=current_per_model,
    )
