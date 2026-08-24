# specter.cli

!!! note
    `mkdocs-click` doesn't render this CLI's options: `specter`'s CLI is
    built on `rich_click`, whose custom help formatter silently drops
    options when introspected the way `mkdocs-click` expects. Until
    that's resolved, this page documents the underlying Python
    functions directly — for the actual flags, run `specter --help` /
    `specter simulate particles --help`.

::: specter.cli._cli
    options:
      show_root_heading: true

::: specter.cli.simulate
    options:
      show_root_heading: true
