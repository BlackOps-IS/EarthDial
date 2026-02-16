# Contributing to EarthDial

Thank you for your interest in contributing to EarthDial — the AI Decision Intelligence Layer for Planetary Systems.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/EarthDial.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Set up your NVIDIA API key: `cp .env.example .env` and add your key
5. Run locally: `python -m streamlit run app.py`

## What We're Looking For

### High Priority
- **NVIDIA cuGraph integration** — Replace NetworkX with GPU-accelerated graph algorithms
- **Earth-2 forecast ingestion** — Parse NetCDF4/GRIB2 outputs from FourCastNet/CorrDiff
- **Multi-hazard support** — Extend risk engine for floods, earthquakes, industrial incidents
- **Performance optimization** — cuDF/cuPy acceleration for risk computation

### Welcome Contributions
- Bug fixes and error handling improvements
- Documentation and code comments
- Unit tests for risk engine and optimizer
- Visualization layer enhancements
- Mobile responsiveness improvements
- Accessibility improvements

## Development Guidelines

- Follow PEP 8 style conventions
- Add docstrings to all public functions
- Include type hints where practical
- Test your changes locally before submitting
- Keep commits focused and well-described

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes
3. Test locally with `python -m streamlit run app.py`
4. Submit a pull request with a clear description
5. Reference any related issues

## Reporting Issues

Use GitHub Issues with these labels:
- `bug` — Something isn't working
- `feature` — New functionality request
- `research` — Exploration of new capabilities
- `nvidia` — NVIDIA technology integration

## Code of Conduct

Be respectful, constructive, and collaborative. We're building technology to protect lives.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
