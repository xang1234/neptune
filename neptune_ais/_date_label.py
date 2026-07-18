"""Date-label formatting shared by visualization scripts and their tests."""

from __future__ import annotations

from datetime import date


def format_date_label(days: list[date]) -> str:
    """Format a single date or compact date range with unambiguous boundaries."""
    if len(days) > 1:
        start, end = days[0], days[-1]
        if start.year != end.year:
            return (
                f"{start.day} {start.strftime('%b %Y')}–"
                f"{end.day} {end.strftime('%b %Y')}"
            )
        if start.month != end.month:
            return f"{start.day} {start.strftime('%b')}–{end.day} {end.strftime('%b %Y')}"
        return f"{start.day}–{end.day} {end.strftime('%b %Y')}"
    return f"{days[0].day} {days[0].strftime('%b %Y')}"
