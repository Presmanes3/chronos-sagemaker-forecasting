# GitHub Copilot Instructions

## Code Style Guidelines

### Logging
- Use minimalist log messages
- No emojis in logs
- Keep log messages concise and informative
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Use english

### Documentation
- Use minimalist docstrings
- Include only essential information: brief description, parameters, and return values
- Avoid verbose or redundant documentation
- Use english

```python
def calculate_forecast(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Generate forecast for given horizon.
    
    Args:
        data: Input time series data.
        horizon: Number of periods to forecast.
    
    Returns:
        DataFrame with forecast results.
    """
```

### Code Architecture
- Follow SOLID principles:
  - Single Responsibility Principle: Each class/function should have one responsibility
  - Open/Closed Principle: Open for extension, closed for modification
  - Liskov Substitution Principle: Subtypes must be substitutable for base types
  - Interface Segregation Principle: Prefer small, specific interfaces
  - Dependency Inversion Principle: Depend on abstractions, not concretions
- Use composition over inheritance for better maintainability
- Separate code blocks into subclasses when appropriate
- Keep functions and methods small and focused

### Code Organization
- Group related functionality into modules
- Use clear and descriptive naming conventions
- Prefer dependency injection for testability

### Testing
- All unit tests must be placed in `test/unit/`
- All end-to-end tests must be placed in `test/e2e/`
- Follow the naming convention: `test_<module_name>.py`
- Keep test files organized by the module they test

```
test/
├── unit/
│   ├── test_config.py
│   ├── test_predictor.py
│   └── ...
└── e2e/
    ├── test_inference.py
    └── ...
```

### General Best Practices
- Write clean, readable code
- Avoid code duplication (DRY principle)
- Handle errors appropriately
- Use type hints for function signatures
- Keep dependencies minimal and explicit
