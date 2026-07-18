"""Regression tests for PR #4 review fixes."""

from __future__ import annotations

import importlib.util
import re
from datetime import date
from pathlib import Path
from types import ModuleType

from neptune_ais._dashboard_template import _JS_FILTERS, _RIGHT_PANEL


def _load_crossings_gif_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "generate_crossings_gif.py"
    spec = importlib.util.spec_from_file_location("generate_crossings_gif", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_date_label_is_portable_for_single_day() -> None:
    module = _load_crossings_gif_script()

    assert module._format_date_label([date(2024, 6, 15)]) == "15 Jun 2024"


def test_date_label_is_portable_for_multiple_days() -> None:
    module = _load_crossings_gif_script()

    days = [date(2024, 6, 15), date(2024, 6, 16)]
    assert module._format_date_label(days) == "15–16 Jun 2024"


def test_layer_toggle_aria_state_matches_initial_active_state() -> None:
    layer_section = _RIGHT_PANEL.template.split('id="layer-toggles"', 1)[1].split(
        "</div>", 1
    )[0]
    buttons = re.findall(
        r'<button class="([^"]*)" data-layer="([^"]+)" aria-pressed="(true|false)">',
        layer_section,
    )

    assert len(buttons) == 6
    for classes, _layer, aria_pressed in buttons:
        assert aria_pressed == str("active" in classes.split()).lower()


def test_layer_toggle_click_handler_synchronizes_aria_state() -> None:
    assert "const isActive = btn.classList.contains('active');" in _JS_FILTERS
    assert "state.layers[btn.dataset.layer] = isActive;" in _JS_FILTERS
    assert "btn.setAttribute('aria-pressed', String(isActive));" in _JS_FILTERS
