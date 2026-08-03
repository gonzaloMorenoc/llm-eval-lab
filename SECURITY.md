# Security Policy

## Supported versions

This is a learning-oriented QA framework, not production infrastructure.
Security fixes land on `main` and ship in the next release; older versions are
not patched.

| Version | Supported |
|---------|-----------|
| 0.3.x   | ✅        |
| < 0.3   | ❌        |

## Reporting a vulnerability

Please **do not open a public issue** for a security problem.

Use GitHub's [private vulnerability reporting](https://github.com/gonzaloMorenoc/llm-eval-lab/security/advisories/new)
instead. Include the affected component, reproduction steps, and impact.
Expect an initial response within 7 days.

## What this project does with your secrets

Worth understanding before you point it at a real provider:

- **API keys are read from environment variables only** (`config/.env`, which
  is gitignored). They are never written to `config.yaml` or to any report.
- **Errors are redacted before persistence.** Provider SDKs routinely echo the
  failing key back in exception messages, so everything written to
  `results/<run_id>/report.json` passes through `src/redaction.py`, which
  masks OpenAI, Groq, Google, and bearer-token shapes.
- **Reports contain model outputs.** A `report.json` holds whatever the model
  under test said, including any prompt content it echoed back. Treat the
  `results/` directory as sensitive — it is gitignored by default.

## Scope

Evaluating a chatbot means deliberately sending it adversarial prompts. The
`datasets/safety.jsonl` file contains prompt injections and jailbreak attempts
**as test inputs** — that is the point of the tool. Reports about the contents
of those datasets are not vulnerabilities.

Genuine issues include: secrets leaking into persisted output or logs, code
execution via a crafted dataset or config file, and injection through the
dashboard's HTML rendering.
