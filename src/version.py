# MIT License
#
# Copyright (c) 2026 Ganesh Kambli
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
src/version.py
----------------
Single source of truth for the application's version string.

Previously this string was duplicated as a hardcoded literal in several
places (the FastAPI `version=` kwarg in src/api/app.py, the /health,
/api/v1/status, and /api/v1/version endpoint fallbacks in
src/api/routers/admin.py, and src/utils/version_check.py's own
APP_VERSION constant), so bumping the version meant remembering to edit
every one of those spots -- and they could silently drift out of sync
with each other and with the actual released version.

Bump this constant in lock-step with CHANGELOG.md when cutting a new
release. Everything else in the codebase that needs the running app's
version should import APP_VERSION from here rather than hardcoding its
own copy.
"""

from __future__ import annotations

APP_VERSION: str = "1.0.0"
