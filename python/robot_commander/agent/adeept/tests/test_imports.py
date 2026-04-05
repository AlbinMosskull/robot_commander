import pytest
import platform

@pytest.mark.skipif(platform.machine() not in ('armv7l', 'aarch64'), 
                    reason="Only meant to be run on the Raspberry Pi")
def test_adeept_agent_imports():
    import robot_commander.agent.adeept.adeept_agent
