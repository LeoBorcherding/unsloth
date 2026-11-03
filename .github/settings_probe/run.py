import json, os, subprocess, sys, xml.etree.ElementTree as ET
from pathlib import Path

root = Path(__file__).resolve().parent
results = {}
for arm in ("before", "after"):
    env = {**os.environ, "PYTHONPATH": str(root / arm)}
    probe = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "from studio_regress.journeys import settings; print(settings.__file__)",
        ],
        env = env,
        text = True,
    ).strip()
    assert Path(probe).resolve() == root / arm / "studio_regress/journeys/settings.py", probe
    xml = root / (arm + ".xml")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(root / "test_settings_current.py"),
            "-q",
            "-o",
            "addopts=",
            "--junitxml",
            str(xml),
        ],
        env = env,
    )
    cases = list(ET.parse(xml).iter("testcase"))
    failures = [
        case.attrib["name"]
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    ]
    assert len(cases) == 4 and not any(case.find("skipped") is not None for case in cases)
    if arm == "before":
        assert process.returncode == 1 and failures == [
            "test_settings_persistence_uses_current_and_legacy_accessible_names[Context window usage]"
        ], failures
    else:
        assert process.returncode == 0 and failures == [], failures
    results[arm] = {"cases": len(cases), "failures": failures, "source": probe}
(root / "result.json").write_text(json.dumps(results, indent = 2))
print(json.dumps(results, indent = 2))
