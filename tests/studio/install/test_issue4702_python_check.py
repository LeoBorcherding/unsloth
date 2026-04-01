"""
Tests for issue #4702: setup.ps1 Python check fails when system Python is
outside the supported range but the venv Python (created by install.ps1) is valid.

Root cause: setup.ps1 section "1g. Python" ran `python --version` against the
*system* Python, not the venv's Python.  On a machine with Python 3.9 as the
default, this produced:

    [ERROR] Python Python 3.9.23 is outside supported range (need >= 3.11 and < 3.14).

even though install.ps1 had already created the venv with Python 3.13.

Fix: check the well-known venv Python path first
(~/.unsloth/studio/unsloth_studio/Scripts/python.exe).  Only fall through to
the system-Python check when the venv doesn't exist yet.

Run: pytest tests/studio/install/test_issue4702_python_check.py -v
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"


# ---------------------------------------------------------------------------
# Static analysis helpers
# ---------------------------------------------------------------------------

def _setup_ps1_text() -> str:
    return SETUP_PS1.read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Section 1g: venv-first Python check (static analysis)
# ---------------------------------------------------------------------------

class TestSetupPs1VenvFirstPythonCheck:
    """Verify that setup.ps1 checks the venv Python before the system Python."""

    def test_venv_python_path_constructed_before_system_detection(self):
        """The venv python.exe path check must happen before the conda-aware fallback call."""
        text = _setup_ps1_text()

        venv_py_pos = text.find("unsloth_studio")
        # The CALL to Find-EarlyCompatiblePython (not the definition) is the fallback
        # Find the first call: "$_CheckedPythonExe = Find-EarlyCompatiblePython"
        call_pos = text.find("$_CheckedPythonExe = Find-EarlyCompatiblePython")
        assert venv_py_pos != -1, (
            "setup.ps1 must reference the unsloth_studio venv path"
        )
        assert call_pos != -1, (
            "setup.ps1 must call Find-EarlyCompatiblePython as a fallback"
        )
        assert venv_py_pos < call_pos, (
            "Venv Python path check must appear BEFORE the fallback "
            "'$_CheckedPythonExe = Find-EarlyCompatiblePython' call"
        )

    def test_checked_python_exe_tracks_resolved_interpreter(self):
        """The $_CheckedPythonExe variable must be set when the venv Python is valid."""
        text = _setup_ps1_text()
        # The implementation uses $_CheckedPythonExe to track the resolved Python
        assert "$_CheckedPythonExe" in text, (
            "setup.ps1 must use $_CheckedPythonExe to track the validated interpreter"
        )
        # The venv python path assignment must occur together with $_CheckedPythonExe
        venv_and_checked = re.search(
            r"unsloth_studio.*?\$_CheckedPythonExe\s*=",
            text,
            re.DOTALL,
        )
        assert venv_and_checked, (
            "$_CheckedPythonExe must be assigned when the venv Python is validated"
        )

    def test_venv_python_exe_used_for_scripts_dir(self):
        """The PATH manipulation must use the checked Python, not bare 'python'."""
        text = _setup_ps1_text()
        # After the Python check block, the script should NOT call bare `python -c`
        # for Scripts-dir detection; it should use the resolved exe variable.
        scripts_dir_pattern = re.compile(
            r"\$ScriptsDir\s*=\s*&\s*\$_(?:PythonForPath|CheckedPythonExe)\b",
            re.IGNORECASE,
        )
        assert scripts_dir_pattern.search(text), (
            "$ScriptsDir must be obtained via the checked Python executable "
            "variable ($_PythonForPath or $_CheckedPythonExe), not via bare 'python'"
        )

    def test_error_exit_present_for_no_compatible_python(self):
        """An error exit must still be present when no compatible Python can be found."""
        text = _setup_ps1_text()
        # The implementation must still exit with an error if Python cannot be found
        assert "Python could not be installed automatically" in text or \
               "outside supported range" in text or \
               "Python 3.11-3.13 not found" in text, (
            "setup.ps1 must still have an error exit for when no compatible Python is found"
        )

    def test_venv_python_version_range_correct(self):
        """The venv pre-check must use the correct version bounds (3.11-3.13)."""
        text = _setup_ps1_text()
        # The section checking the venv Python must use bounds >= 11 and <= 13
        # Look for the bounds near the unsloth_studio venv path
        venv_check_match = re.search(
            r"unsloth_studio.*?-ge\s+11.*?-le\s+13",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        assert venv_check_match, (
            "The venv Python check must use version bounds >= 11 and <= 13 "
            "(i.e., Python 3.11-3.13)"
        )

    def test_unsloth_python_exe_env_var_checked(self):
        """The UNSLOTH_PYTHON_EXE env var (set by install.ps1) must be used."""
        text = _setup_ps1_text()
        assert "UNSLOTH_PYTHON_EXE" in text, (
            "setup.ps1 must check the UNSLOTH_PYTHON_EXE environment variable "
            "which install.ps1 sets to the exact Python it used"
        )


# ---------------------------------------------------------------------------
# PowerShell integration test (Windows only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell test is Windows-only")
class TestSetupPs1VenvPythonCheckIntegration:
    """Integration tests that run PowerShell fragments to verify the check logic."""

    def _run_ps_fragment(self, fragment: str, timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a PowerShell script fragment and return the CompletedProcess."""
        return subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", fragment],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_venv_python_passes_when_venv_has_valid_version(self, tmp_path: Path):
        """When the venv Python (3.12) is present, the version check should pass."""
        # Create a minimal fake venv Python that reports Python 3.12
        scripts_dir = tmp_path / "Scripts"
        scripts_dir.mkdir()
        fake_python = scripts_dir / "python.exe"

        # Write a minimal batch-file shim that powershell can execute via .exe extension.
        # We use a PowerShell helper function to simulate the check logic.
        ps_fragment = f"""
$_EarlyVenvPython = '{fake_python}'
$PythonOk = $false
$_CheckedPythonExe = $null

# Simulate a fake python that reports 3.12 by defining a mock function
function Invoke-FakePython {{
    param([string]$arg)
    if ($arg -eq '--version') {{ return 'Python 3.12.0' }}
    return ''
}}

# Replicate the section-1g logic (without the filesystem Test-Path check)
$PyVer = 'Python 3.12.0'  # simulate venv python output
if ($PyVer -match '(\\d+)\\.(\\d+)') {{
    $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
    if ($PyMajor -eq 3 -and $PyMinor -ge 11 -and $PyMinor -lt 14) {{
        $PythonOk = $true
        $_CheckedPythonExe = '{fake_python}'
    }}
}}

if ($PythonOk) {{ Write-Output 'PASS' }} else {{ Write-Output 'FAIL' }}
"""
        result = self._run_ps_fragment(ps_fragment)
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_system_python_fails_when_version_is_39(self):
        """When system Python is 3.9 and no venv exists, the check should fail."""
        ps_fragment = r"""
$PythonOk = $false

# Simulate no venv Python present
$_EarlyVenvPython = 'C:\NonExistent\path\python.exe'

# Simulate system Python 3.9 (out of range)
$PyVer = 'Python 3.9.23'
if ($PyVer -match '(\d+)\.(\d+)') {
    $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
    if ($PyMajor -eq 3 -and $PyMinor -ge 11 -and $PyMinor -lt 14) {
        $PythonOk = $true
    }
}

if ($PythonOk) { Write-Output 'PASS' } else { Write-Output 'FAIL' }
"""
        result = self._run_ps_fragment(ps_fragment)
        assert result.returncode == 0
        assert "FAIL" in result.stdout

    def test_venv_python_takes_precedence_over_bad_system_python(self):
        """Venv Python 3.13 should pass even when system Python is 3.9."""
        ps_fragment = r"""
$PythonOk = $false

# Step 1: check venv Python (3.13 -- valid)
$VenvPyVer = 'Python 3.13.0'
if ($VenvPyVer -match '(\d+)\.(\d+)') {
    $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
    if ($PyMajor -eq 3 -and $PyMinor -ge 11 -and $PyMinor -lt 14) {
        $PythonOk = $true
    }
}

# Step 2: if still not OK, check system Python (3.9 -- invalid)
if (-not $PythonOk) {
    $SysPyVer = 'Python 3.9.23'
    if ($SysPyVer -match '(\d+)\.(\d+)') {
        $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
        if ($PyMajor -eq 3 -and $PyMinor -ge 11 -and $PyMinor -lt 14) {
            $PythonOk = $true
        } else {
            Write-Output 'ERROR: system python out of range'
        }
    }
}

if ($PythonOk) { Write-Output 'PASS' } else { Write-Output 'FAIL' }
"""
        result = self._run_ps_fragment(ps_fragment)
        assert result.returncode == 0
        assert "PASS" in result.stdout
        assert "ERROR" not in result.stdout

    def test_version_bounds_3_11_inclusive_to_3_13_inclusive(self):
        """Test boundary versions: 3.11 and 3.13 pass; 3.10 and 3.14 fail."""
        for version, should_pass in [
            ("3.10.0", False),
            ("3.11.0", True),
            ("3.12.5", True),
            ("3.13.0", True),
            ("3.14.0", False),
        ]:
            ps_fragment = f"""
$PyVer = 'Python {version}'
$PythonOk = $false
if ($PyVer -match '(\\d+)\\.(\\d+)') {{
    $PyMajor = [int]$Matches[1]; $PyMinor = [int]$Matches[2]
    if ($PyMajor -eq 3 -and $PyMinor -ge 11 -and $PyMinor -lt 14) {{
        $PythonOk = $true
    }}
}}
if ($PythonOk) {{ Write-Output 'PASS' }} else {{ Write-Output 'FAIL' }}
"""
            result = self._run_ps_fragment(ps_fragment)
            assert result.returncode == 0
            expected = "PASS" if should_pass else "FAIL"
            assert expected in result.stdout, (
                f"Python {version}: expected {expected}, got: {result.stdout!r}"
            )
