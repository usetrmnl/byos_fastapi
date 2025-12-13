# TRMNL Server

This project is an e-ink device server that allows a client to connect to it and generates images and data for display on TRMNL e-ink devices.

## Coding Style Guidelines

When contributing to this project, please adhere to the following coding style guidelines:

* DO NOT CREATE NEW CODE FILES unless explicitly told to.
* SPACES, not TABS, for indentation. Use 4 spaces per indentation level.
* Code in a functional style, with concise functions that do one thing only.
* NEVER duplicate code. Always re-use existing code or create new helper functions. If they are reusable, add them to `utils.py` or `models.py` as appropriate.
* When importing, prefer explicit imports (`from sys import stderr`) rather than just importing the module. A critical example is doing `from os.path import join, dirname, abspath` instead of `import os` and then using `os.path.join()`, etc. Never mind how many imports this creates; explicit imports are preferred for clarity.
* Inside a package, prefer package-relative imports (`from .utils import helper_function`) rather than absolute imports (`from trmnl_server.utils import helper_function`).
* When creating new functions, include type hints for all parameters and return values.
* Do not create one-liner wrappers around existing/internal module functions unless absolutely necessary. Use the public ones instead.
* When considering creating utility functions, try not to create one or two-liners. Inline the logic instead if they are that simple.
* Add utility functions to `utils.py` and constants to `config.py`, making sure to import them where needed and that any major configuration parameters are handled in a consistent way.
* NEVER add import statements inside functions or methods. Add any and all imports at the top of the file.
* Only perform database operations in `models.py`. Create or re-use new helpers there as needed.
* Before writing new helpers or adding inline logic for things that might be reusable, check if there are existing ones that can be re-used or adapted in `utils.py` or `models.py`.

