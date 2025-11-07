# deepagents cli

This is the CLI for deepagents

## Skills

Skills are reusable agent capabilities that can be loaded into the CLI. The CLI looks for skills in `~/.deepagents/skills/` by default.

### Example Skills

Example skills are provided in the `examples/skills/` directory:

- **web-research** - Structured web research workflow with planning, parallel delegation, and synthesis
- **langgraph-docs** - LangGraph documentation lookup and guidance

To use an example skill, copy it to your skills directory:

```bash
mkdir -p ~/.deepagents/skills
cp -r examples/skills/web-research ~/.deepagents/skills/
```

### Managing Skills

```bash
# List available skills
deepagents skills list

# Create a new skill from template
deepagents skills create my-skill

# View detailed information about a skill
deepagents skills info web-research
```
## Development

### Running Tests

To run the test suite:

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_tools.py

# Run with coverage
pytest --cov=deepagents_cli --cov-report=html
```

### Test Dependencies

The test suite uses:
- `pytest` - Test framework
- `responses` - Mock HTTP requests for testing
- `pytest-cov` - Coverage reporting
- `pytest-asyncio` - Async test support
