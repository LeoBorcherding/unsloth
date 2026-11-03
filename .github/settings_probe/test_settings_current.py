import asyncio, re
from types import SimpleNamespace
import pytest
from studio_regress.contract import StepUnreachable
from studio_regress.journeys import settings


class _NamedSwitches:
    def __init__(self, names):
        self.names = names

    def get_by_role(self, role, *, name):
        assert role == "switch"
        return _NamedSwitches(
            [
                value
                for value in self.names
                if (name.search(value) if isinstance(name, re.Pattern) else name in value)
            ]
        )

    @property
    def first(self):
        return self

    async def wait_for(self, **_):
        if not self.names:
            raise TimeoutError("no visible switch")

    async def count(self):
        return len(self.names)


def _settings_ctx(names):
    switches = _NamedSwitches(names)
    return SimpleNamespace(page = SimpleNamespace(locator = lambda _: switches))


@pytest.mark.parametrize("label", ["Context window usage", "Show context window usage"])
def test_settings_persistence_uses_current_and_legacy_accessible_names(label):
    result = asyncio.run(settings._switch(_settings_ctx([label, "Plain text composer"])))
    assert result.names == [label]


@pytest.mark.parametrize("label", ["Plain text composer", "Context window usage threshold"])
def test_settings_persistence_still_rejects_an_absent_control(label):
    with pytest.raises(StepUnreachable, match = "not present"):
        asyncio.run(settings._switch(_settings_ctx([label])))
