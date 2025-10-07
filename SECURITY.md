# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in NovoMD, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report security issues via one of these methods:

1. **GitHub Security Advisories** (preferred)
   - Go to the [Security tab](https://github.com/quantnexusai/NovoMD/security)
   - Click "Report a vulnerability"
   - Provide detailed information about the vulnerability

2. **Email**
   - Send details to: security@quantnexusai.com
   - Use GPG encryption if possible (key available on request)

### What to Include

Please provide as much information as possible:

- Type of vulnerability
- Full path or URL of the affected resource
- Step-by-step instructions to reproduce the issue
- Proof of concept or exploit code (if available)
- Potential impact of the vulnerability
- Suggested mitigation or fix (if available)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 7 days
  - High: 30 days
  - Medium: 90 days
  - Low: Best effort

### What to Expect

1. **Acknowledgment**: We'll confirm receipt of your report
2. **Investigation**: We'll investigate and validate the issue
3. **Fix Development**: We'll develop and test a fix
4. **Disclosure**: We'll coordinate disclosure timing with you
5. **Credit**: We'll credit you in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

### For Users

1. **API Key Security**
   - Never commit API keys to version control
   - Use strong, randomly generated API keys (32+ characters)
   - Rotate API keys regularly
   - Store keys securely (environment variables, secrets managers)

2. **Network Security**
   - Deploy behind HTTPS/TLS in production
   - Use firewall rules to restrict access
   - Implement rate limiting
   - Use reverse proxy (nginx, Traefik, AWS ALB)

3. **Container Security**
   - Keep Docker images updated
   - Scan images for vulnerabilities
   - Run containers as non-root user (already configured)
   - Limit container resources

4. **Dependency Management**
   - Regularly update dependencies
   - Monitor for security advisories
   - Use tools like `pip-audit` or `safety`

### For Developers

1. **Code Review**
   - All code changes require review
   - Look for common vulnerabilities (injection, XSS, etc.)
   - Use static analysis tools

2. **Input Validation**
   - Validate all user inputs
   - Sanitize data before processing
   - Use Pydantic models for validation

3. **Error Handling**
   - Don't expose sensitive information in error messages
   - Log security events appropriately
   - Implement proper exception handling

4. **Authentication**
   - Use secure random key generation
   - Implement rate limiting for auth endpoints
   - Log authentication failures

## Known Limitations

1. **Mock Data**: Current implementation uses mock/simulated data for demonstrations. Production deployments should integrate with real molecular dynamics engines.

2. **Rate Limiting**: Not implemented by default. Users should implement at the proxy level.

3. **HTTPS**: Not configured by default. Users must deploy behind a reverse proxy with SSL/TLS.

## Security Updates

Subscribe to security advisories:
- Watch the repository and enable security alerts
- Check [Security Advisories](https://github.com/quantnexusai/NovoMD/security/advisories)

## Disclosure Policy

- We follow responsible disclosure practices
- We'll work with reporters to understand and fix issues
- We'll publicly disclose issues after fixes are released
- We'll credit security researchers (unless they prefer anonymity)

## Contact

For security-related questions: security@quantnexusai.com

---

Thank you for helping keep NovoMD secure!
