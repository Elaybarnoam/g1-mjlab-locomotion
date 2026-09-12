"""Static research report rendering."""

from .policy_spec import generate_walking_policy_spec
from .render import render_report

__all__ = ["generate_walking_policy_spec", "render_report"]
