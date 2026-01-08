---
name: Bug Report
about: Report a bug to help us improve NovoMD
title: '[BUG] '
labels: bug
assignees: ''
---

## Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Send request to '...'
2. With payload '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- **NovoMD Version**: [e.g., 1.1.0]
- **Python Version**: [e.g., 3.11]
- **OS**: [e.g., Ubuntu 22.04, macOS 14, Windows 11]
- **Deployment**: [e.g., Docker, local, Kubernetes]

## Request/Response (if applicable)
```bash
# Example request
curl -X POST http://localhost:8010/smiles-to-omd \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"smiles": "CCO"}'
```

```json
// Response or error message
```

## Logs
```
Paste relevant logs here
```

## Additional Context
Add any other context about the problem here.
