# Contributing Guidelines

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a welcoming environment for all contributors.

## How to Contribute

### Reporting Issues

1. **Check existing issues** first to avoid duplicates
2. **Use issue templates** when creating new issues
3. **Provide detailed information**:
   - Clear description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Error messages or logs

### Suggesting Features

1. **Create a feature request** using the template
2. **Explain the motivation** behind the feature
3. **Provide examples** of how it would work
4. **Consider alternatives** you've thought about

### Contributing Code

#### Development Workflow

1. **Fork the repository**
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our coding standards
4. **Write/update tests** for your changes
5. **Run tests locally**:
   ```bash
   make test
   ```
6. **Run linters**:
   ```bash
   make lint
   ```
7. **Commit your changes** using conventional commits:
   ```bash
   git commit -m "feat: add new feature"
   ```
8. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
9. **Create a Pull Request**

#### Pull Request Guidelines

- **Reference the issue** your PR addresses
- **Provide a clear description** of changes
- **Include test results**
- **Update documentation** if needed
- **Ensure all CI checks pass**
- **Request review** when ready

#### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or modifications
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

Examples:
```bash
feat(auth): add OAuth2 authentication
fix(api): handle null response correctly
docs: update installation instructions
```

### Code Style

#### General Guidelines

- Write clean, readable code
- Follow the principle of least surprise
- Keep functions small and focused
- Use descriptive variable and function names
- Add comments for complex logic
- Avoid code duplication

#### Language-Specific Standards

**Python:**
- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use Black for formatting
- Use isort for imports

**JavaScript/TypeScript:**
- Use ESLint configuration
- Prefer const over let
- Use arrow functions appropriately
- Use async/await over promises

**Shell Scripts:**
- Use ShellCheck
- Prefer `[[` over `[`
- Quote variables
- Use `set -euo pipefail`

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Test edge cases
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

### Documentation

- Update README if needed
- Add docstrings to functions
- Update API documentation
- Include examples where helpful
- Keep documentation in sync with code

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/project.git
   cd project
   ```

2. **Install dependencies**:
   ```bash
   # Python projects
   pip install -r requirements-dev.txt
   
   # Node.js projects
   npm install
   ```

3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Run tests**:
   ```bash
   make test
   ```

## Review Process

1. **Automated checks** run on all PRs
2. **Code review** by maintainers
3. **Address feedback** promptly
4. **Approval required** before merge
5. **Squash and merge** to maintain clean history

## Release Process

We use semantic versioning and automated releases:

1. Commits to `main` trigger release workflow
2. Version bumps based on commit types
3. Changelog generated automatically
4. GitHub releases created with artifacts

## Getting Help

- Check the documentation
- Look for similar issues
- Ask in discussions
- Contact maintainers

## Recognition

Contributors are recognized in:
- Release notes
- Contributors list
- Project documentation

Thank you for contributing! 🎉