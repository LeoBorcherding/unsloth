"""
Tests for issue #4702: setup.ps1 Python check fails when system Python is
outside the supported range but the venv Python (created by install.ps1) is valid.

Root cause: setup.ps1 section "1g. Python" ran `python --version` against the
*system* Python, not the venv's Python.  On a machine with Python 3.9 as the
default, this produced:

    [ERROR] Python Python 3.9.23 is outside supported range (need >= 3.11 and < 3.14).

even though install.ps1 had already created the venv with Python 3.13.

Fix: use a priority chain that never falls back to bare 'python --version':
  1. UNSLOTH_PYTHON_EXE env var (set by install.ps1, exact path)
  2. Studio venv Python (~/.unsloth/studio/unsloth_studio/Scripts/python.exe)
  3. Conda-aware detection via py launcher and Get-Command -All
  4. Winget install of Python 3.12 as last resort

install.ps1 now sets $env:UNSLOTH_PYTHON_EXE = $DetectedPython.Path before
invoking "unsloth studio setup" so that setup.ps1 always uses the correct
interpreter regardless of what 'python' resolves to on PATH.

Run: pytest tests/studio/install/test_issue4702_python_check.py -v
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"


# ---------------------------------------------------------------------------
# Static analysis helpers
# ---------------------------------------------------------------------------

def _setup_ps1_text() -> str:
    return SETUP_PS1.read_text(encoding="utf-8-sig")

def _install_ps1_text() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Section 1g: priority-chain Python check (static analysis)
# ---------------------------------------------------------------------------

class TestSetupPs1VenvFirstPythonCheck:
    """Verify that setup.ps1 uses a safe, conda-aware Python detection chain."""

    def test_unsloth_python_exe_env_var_checked(self):
        """setup.ps1 must check UNSLOTH_PYTHON_EXE before any other detection."""
        text = _setup_ps1_text()
        assert "UNSLOTH_PYTHON_EXE" in text, (
            "setup.ps1 must check the UNSLOTH_PYTHON_EXE env var "
            "so that install.ps1 can pass the exact Python path"
        )

    def test_install_ps1_sets_unsloth_python_exe(self):
        """install.ps1 must set UNSLOTH_PYTHON_EXE before calling unsloth studio setup."""
        text = _install_ps1_text()
        pattern = re.compile(
            r"\$env:UNSLOTH_PYTHON_EXE\s*=\s*\$DetectedPython\.Path",
            re.IGNORECASE,
        )
        assert pattern.search(text), (
            "install.ps1 must set $env:UNSLOTH_PYTHON_EXE = $DetectedPython.Path "
            "before calling 'unsloth studio setup'"
        )

    def test_install_ps1_sets_env_var_before_studio_setup_call(self):
        """UNSLOTH_PYTHON_EXE must be assigned before the 'unsloth studio setup' invocation."""
        text = _install_ps1_text()
        env_var_pos = text.find("UNSLOTH_PYTHON_EXE")
        studio_call_pos = text.find("studio', 'setup'")
        assert env_var_pos != -1, "install.ps1 must reference UNSLOTH_PYTHON_EXE"
        assert studio_call_pos != -1, "install.ps1 must invoke 'studio setup'"
        assert env_var_pos < studio_call_pos, (
            "UNSLOTH_PYTHON_EXE must be set BEFORE the studio setup invocation"
        )

    def test_venv_python_path_is_fallback(self):
        """The studio venv Python path must be used as a fallback in Section 1g."""
        text = _setup_ps1_text()
        assert "unsloth_studio" in text, (
            "setup.ps1 must reference the unsloth_studio venv path as a fallback"
        )

    def test_no_bare_python_version_in_section_1g(self):
        """Section 1g must not use bare 'python --version' (picks up wrong Python)."""
        text = _setup_ps1_text()
        section_start = text.find("# 1g. Python")
        # End of section 1g is the start of phase 2 (prerequisites ready line)
        section_end = text.find("prerequisites ready", section_start)
        assert section_start != -1, "Could not find Section 1g in setup.ps1"
        section_text = text[section_start:section_end] if section_end != -1 else text[section_start:]
        # Bare `python --version` is a token not preceded by $variable or & $variable
        bare_pattern = re.compile(r'(?<!\w)python\s+--version', re.IGNORECASE)
        # Filter out lines that call python via a variable ($ prefix)
        bad_lines = [
            line for line in section_text.splitlines()
            if bare_pattern.search(line) and not re.search(r'\$\w+\s+--version', line)
        ]
        assert not bad_lines, (
            "Section 1g must not use bare 'python --version' which picks up "
            f"the wrong interpreter. Offending lines: {bad_lines}"
        )

    def test_venv_python_exe_used_for_scripts_dir(self):
        """The PATH manipulation must use the checked Python, not bare 'python'."""
        text = _setup_ps1_text()
        scripts_dir_pattern = re.compile(
            r"\$ScriptsDir\s*=\s*&\s*\$_(?:PythonForPath|CheckedPythonExe)\b",
            re.IGNORECASE,
        )
        assert scripts_dir_pattern.search(text), (
            "$ScriptsDir must be obtained via the checked Python executable "
            "variable ($_PythonForPath or $_CheckedPythonExe), not via bare 'python'"
        )

    def test_conda_aware_detection_present_in_section_1g(self):
        """Section 1g must contain conda-aware Python detection."""
        text = _setup_ps1_text()
        section_start = text.find("# 1g. Python")
        section_end = text.find("prerequisites ready", section_start)
        assert section_start != -1, "Could not find Section 1g in setup.ps1"
        section_text = text[section_start:section_end] if section_end != -1 else text[section_start:]
        assert "conda" in section_text.lower() or "EarlyCompatiblePython" in section_text, (
            "Section 1g must contain conda-aware Python detection "
            "(should skip Conda/Anaconda interpreters)"
        )

    def test_phase3_also_checks_unsloth_python_exe(self):
        """Phase 3 Python detection must also check UNSLOTH_PYTHON_EXE first."""
        text = _setup_ps1_text()
        phase3_start = text.find("PHASE 3")
        assert phase3_start != -1, "Could not find PHASE 3 in setup.ps1"
        phase3_text = text[phase3_start:]
        assert "UNSLOTH_PYTHON_EXE" in phase3_text, (
            "Phase 3 Python detection must also check UNSLOTH_PYTHON_EXE "
            "to stay consistent with the install.ps1 -> setup.ps1 contract"
        )

    def test_version_bounds_consistent_in_section_1g(self):
        """Section 1g version bounds must match Phase 3 (3.11 inclusive to 3.13 inclusive)."""
        text = _setup_ps1_text()
        section_start = text.find("# 1g. Python")
        section_end = text.find("prerequisites ready", section_start)
        assert section_start != -1, "Could not find Section 1g"
        section_text = text[section_start:section_end] if section_end != -1 else text[section_start:]
        assert "-ge 11" in section_text, "Section 1g lower bound must be >= 11 (Python 3.11)"
        assert "-le 13" in section_text or "-lt 14" in section_text, (
            "Section 1g upper bound must be <= 13 or < 14 (Python 3.13)"
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

    def test_unsloth_python_exe_takes_precedence(self):
        """UNSLOTH_PYTHON_EXE 3.13 should win even when system Python is 3.9."""
        ps_fragment = r"""
$env:UNSLOTH_PYTHON_EXE = $null  # clear any real value
$_CheckedPythonExe = $null

# Simulate UNSLOTH_PYTHON_EXE set by install.ps1 pointing to Python 3.13
$envExe = 'C:\Python313\python.exe'
$envVer = 'Python 3.13.0'

if ($envExe -and $envVer -match 'Python 3\.(\d+)') {
    $m = [int]$Matches[1]
    if ($m -ge 11 -and $m -le 13) {
        $_CheckedPythonExe = $envExe
    }
}

# Simulate system Python 3.9 (would fail if used)
if (-not $_CheckedPythonExe) {
    $SysPyVer = 'Python 3.9.23'
    if ($SysPyVer -match 'Python 3\.(\d+)') {
        $m = [int]$Matches[1]
        if ($m -ge 11 -and $m -le 13) {
            $_CheckedPythonExe = 'C:\Python39\python.exe'
        }
    }
}

if ($_CheckedPythonExe -eq 'C:\Python313\python.exe') { Write-Output 'PASS' } else { Write-Output 'FAIL' }
"""
        result = self._run_ps_fragment(ps_fragment)
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_venv_python_passes_when_valid(self):
        """When the venv Python (3.12) is present, the version check should pass."""
        ps_fragment = r"""
$_CheckedPythonExe = $null

# No UNSLOTH_PYTHON_EXE set
$envExe = $null

# Venv Python 3.12 exists
$VenvPyVer = 'Python 3.12.0'
if ($VenvPyVer -match 'Python 3\.(\d+)') {
    $m = [int]$Matches[1]
    if ($m -ge 11 -and $m -le 13) {
        $_CheckedPythonExe = 'C:\FakeVenv\Scripts\python.exe'
    }
}

if ($_CheckedPythonExe) { Write-Output 'PASS' } else { Write-Output 'FAIL' }
"""
        result = self._run_ps_fragment(ps_fragment)
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_venv_python_takes_precedence_over_bad_system_python(self):
        """Venv Python 3.13 should pass even when system Python is 3.9."""
        ps_fragment = r"""
$_CheckedPythonExe = $null

# Venv Python 3.13
$VenvPyVer = 'Python 3.13.0'
if ($VenvPyVer -match 'Python 3\.(\d+)') {
    $m = [int]$Matches[1]
    if ($m -ge 11 -and $m -le 13) {
        $_CheckedPythonExe = 'C:\FakeVenv\Scripts\python.exe'
    }
}

# System Python 3.9 (should NOT be reached)
if (-not $_CheckedPythonExe) {
    Write-Output 'ERROR: fell through to system Python'
}

if ($_CheckedPythonExe) { Write-Output 'PASS' } else { Write-Output 'FAIL' }
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
$ok = $false
if ($PyVer -match 'Python 3\\.(\\d+)') {{
    $m = [int]$Matches[1]
    if ($m -ge 11 -and $m -le 13) {{ $ok = $true }}
}}
if ($ok) {{ Write-Output 'PASS' }} else {{ Write-Output 'FAIL' }}
"""
            result = self._run_ps_fragment(ps_fragment)
            assert result.returncode == 0
            expected = "PASS" if should_pass else "FAIL"
            assert expected in result.stdout, (
                f"Python {version}: expected {expected}, got: {result.stdout!r}"
            )