"""How platform values become portable JSON and durable files.

Transport owns byte determinism, document I/O, persisted-value spelling,
configured-value portability, and registry-document algebra. It imports no
domain concern, and the user interface never imports transport: users speak
the verbs their concerns expose. Workspace and bundle path layout lives in
``dpstats.workspace``, not here. Each name has one home in its
owning module; this package re-exports nothing.
"""
