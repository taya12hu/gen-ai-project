"""
Centralized environment loading.

Every module that needs a value from backend/.env used to load it itself
(each with its own `BACKEND_DIR = Path(__file__).resolve().parent.parent`
+ `load_dotenv(...)` boilerplate, copy-pasted across ~6 files). Importing
`app` (any submodule) triggers this exactly once, so nothing else in the
package needs its own load_dotenv() call - modules just read os.environ.
"""

from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"

load_dotenv(BACKEND_DIR / ".env")
