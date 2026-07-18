"""Regression tests for PR #4 review fixes."""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from datetime import date

from neptune_ais._date_label import format_date_label
from neptune_ais._dashboard_template import _JS_FILTERS, _RIGHT_PANEL


def test_date_label_is_portable_for_single_day() -> None:
    assert format_date_label([date(2024, 6, 15)]) == "15 Jun 2024"


def test_date_label_is_portable_for_multiple_days() -> None:
    days = [date(2024, 6, 15), date(2024, 6, 16)]
    assert format_date_label(days) == "15–16 Jun 2024"


def test_date_label_includes_month_for_cross_month_range() -> None:
    days = [date(2024, 6, 30), date(2024, 7, 1)]
    assert format_date_label(days) == "30 Jun–1 Jul 2024"


def test_date_label_includes_year_for_cross_year_range() -> None:
    days = [date(2024, 12, 31), date(2025, 1, 1)]
    assert format_date_label(days) == "31 Dec 2024–1 Jan 2025"


def test_date_label_helper_import_does_not_require_numpy() -> None:
    code = textwrap.dedent(
        """
        import sys
        from datetime import date

        class BlockNumpyImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numpy" or fullname.startswith("numpy."):
                    raise ModuleNotFoundError("numpy blocked for clean-install test")
                return None

        sys.meta_path.insert(0, BlockNumpyImports())

        from neptune_ais._date_label import format_date_label

        assert format_date_label([date(2024, 6, 15)]) == "15 Jun 2024"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
