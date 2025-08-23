# Contributing to Low-Latency RAG System

We welcome contributions to the Low-Latency RAG System! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/low-latency-rag.git
   cd low-latency-rag
   ```
3. **Install development dependencies**:
   ```bash
   pip install -r requirements.deploy.txt
   pip install pytest pytest-asyncio black flake8 mypy
   ```

## Development Workflow

### Setting up the Development Environment

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.deploy.txt
   ```

3. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-api-key"  # Optional, for OpenAI features
   export RAG_AUTH_TOKEN="test-token"    # For API testing
   ```

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards:
   - Follow PEP 8 style guidelines
   - Add type hints where appropriate
   - Include docstrings for new functions and classes
   - Write tests for new functionality

3. **Test your changes**:
   ```bash
   # Run the demo
   python demo.py
   
   # Test API startup (with environment variables set)
   uvicorn app:app --reload
   
   # Run any existing tests
   pytest test.py -v
   ```

### Code Quality

Before submitting your changes:

1. **Format your code**:
   ```bash
   black *.py
   ```

2. **Lint your code**:
   ```bash
   flake8 *.py
   ```

3. **Type check** (optional but recommended):
   ```bash
   mypy *.py
   ```

## Submitting Changes

1. **Commit your changes** with a clear message:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference any related issues
   - Include screenshots for UI changes
   - List any breaking changes

## Types of Contributions

### Bug Reports
- Use the GitHub issue template
- Include steps to reproduce
- Provide system information
- Include error messages and logs

### Feature Requests
- Describe the use case
- Explain the expected behavior
- Consider backwards compatibility
- Discuss potential implementation approaches

### Code Contributions
- **Bug fixes**: Focus on the specific issue
- **New features**: Ensure they align with project goals
- **Performance improvements**: Include benchmarks
- **Documentation**: Always welcome!

## Development Guidelines

### Code Style
- Follow PEP 8 conventions
- Use meaningful variable and function names
- Keep functions focused and small
- Add comments for complex logic

### Testing
- Write tests for new functionality
- Ensure existing tests still pass
- Test both success and failure cases
- Include integration tests for API endpoints

### Documentation
- Update README.md for user-facing changes
- Add inline documentation for complex code
- Update API documentation for endpoint changes
- Include usage examples

## Architecture Guidelines

### Adding New Document Formats
1. Extend the `_extract_text_from_file` method in `RAGOpenAI`
2. Add format detection logic
3. Include appropriate dependencies in requirements
4. Test with sample files

### Adding New LLM Backends
1. Create a new RAG class following the existing pattern
2. Implement the required interface methods
3. Add configuration options
4. Update the API to support the new backend

### Performance Considerations
- Profile changes that might affect performance
- Consider memory usage for large documents
- Optimize vector operations where possible
- Test with realistic document sizes

## Release Process

1. **Version Bumping**: Update version in `pyproject.toml`
2. **Changelog**: Update with new features and fixes
3. **Testing**: Comprehensive testing across environments
4. **Documentation**: Ensure all docs are up to date
5. **Tagging**: Create appropriate git tags

## Community

- Be respectful and inclusive
- Help others learn and contribute
- Share knowledge and best practices
- Provide constructive feedback

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Request Comments**: For code-specific questions

Thank you for contributing to the Low-Latency RAG System! 🚀