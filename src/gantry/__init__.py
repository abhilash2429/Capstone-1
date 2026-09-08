"""Gantry - a runtime harness for tool-using agents.

The library is organised around the idea that an agent is a *loop*, and that
everything which makes a loop trustworthy lives outside the prompt: what the
loop is allowed to do (:mod:`gantry.tools`), what it is allowed to spend
(:mod:`gantry.contract`), what it is allowed to touch (:mod:`gantry.sandbox`),
and what it actually did (:mod:`gantry.telemetry`).
"""

__version__ = "0.1.0"
