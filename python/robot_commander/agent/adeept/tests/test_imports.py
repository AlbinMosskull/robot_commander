import pytest
import platform

@pytest.mark.skipif(platform.machine() not in ('armv7l', 'aarch64'), 
                    reason="Not running on ARM architecture")
def test_adeept_agent_imports():
    import robot_commander.agent.adeept.adeept_agent
