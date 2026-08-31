# Skill adapter smoke

The workspace `vllm-ascend-main2main` thin adapter invoked the analyzer from the
implementation worktree with `scenario=main2main`, the explicit cache parent,
and exact repository SHAs. The command completed successfully and produced
`main2main-report.md`.

The adapter captures engine stdout/stderr while rendering its concise report,
so its outer redirected logs are intentionally empty. Raw analyzer stdout and
stderr are retained in every historical and performance sample directory.
