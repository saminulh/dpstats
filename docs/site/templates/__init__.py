"""Page templates: one declarative spec, and the renderers that fill it.

Every task-space page is an instance of exactly one template declared in
:mod:`templates.spec`. A renderer walks that template's ordered sections and
fills each one from the single source the spec names, so a page cannot gain a
section the spec does not declare and cannot lose one silently.

This directory is also the mkdocstrings custom-template root; its Jinja files
sit under ``python/material`` and are unrelated to these modules.
"""
