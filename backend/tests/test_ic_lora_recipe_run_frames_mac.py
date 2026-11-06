"""Frame count of a queued Home recipe on the Mac layout-to-render cell, priced by the real budget.

The cell has a priced frame cap. Past it, a two-stage job is an ``unmeasured`` bet, and below it
a short memory pool makes the last priced count ``over``. A clip just above the cap snaps down to
that ``over`` count and up to the first ``unmeasured`` count.
"""

from __future__ import annotations

import pytest

from _routes._errors import HTTPError
from api_types import IcLoraSettings
from tests.lora_catalog_helpers import add_ic_lora
from tests.test_ic_lora_recipe_run_frames import _GIB, _ID, Gate, Runner

# 540p stage 1 at the last priced count needs 17 GB. A 16 GB pool is short of it.
_SHORT_POOL_GIB = 16


@pytest.fixture
def mac_runner(test_state, fake_services, create_fake_model_files) -> Runner:
    create_fake_model_files()
    test_state.config.system = "Darwin"
    test_state.config.total_ram_bytes = int(48 * _GIB)
    test_state.config.local_generations_mode = "streaming_models_loading"
    fake_services.gpu_info.mps_pool_bytes = int(_SHORT_POOL_GIB * _GIB)
    gate = Gate(
        IcLoraSettings(skip_stage_2=False, stage_2_ic_lora=True, resolution_factor=1.0),
        "540p", "over", {},
    )
    add_ic_lora(fake_services, id=_ID, default_settings=gate.settings)
    return Runner(test_state, gate)


def _first_unmeasured_count(runner: Runner, test_state) -> int:
    test_state.state.app_settings.budget_mode = "reckless"
    for clip in range(9, 2000, 8):
        _, risk = runner.run(clip)
        if risk is not None and risk.tier == "unmeasured":
            return clip
    raise AssertionError("the Mac cell never reached an unmeasured count")


@pytest.mark.parametrize("above_cap", [1, 5])
@pytest.mark.parametrize("live_mode", ["standard", "reckless"])
def test_an_admitted_unmeasured_job_still_runs_when_its_snapped_down_count_is_refused(
    mac_runner: Runner, test_state, above_cap: int, live_mode: str
) -> None:
    """The bug is the down probe raising: create stored only the unmeasured bet, which does not
    cover the over count, so execute failed a job that create admitted."""
    snapped_up = _first_unmeasured_count(mac_runner, test_state)
    snapped_down = snapped_up - 8
    clip = snapped_down + above_cap
    _, down_risk = mac_runner.run(snapped_down)
    assert down_risk is not None and down_risk.tier == "over", "the pool must make the cap over"

    created_count, created_risk = mac_runner.run(clip)
    assert created_count == snapped_up
    assert created_risk is not None and created_risk.tier == "unmeasured"

    test_state.state.app_settings.budget_mode = live_mode
    count, risk = mac_runner.run(clip, risk_allowance=False, unmeasured_bet=True)
    assert count == snapped_up
    assert risk is not None and risk.tier == "unmeasured"


def test_a_refused_snapped_down_count_does_not_hide_the_snapped_up_refusal(
    mac_runner: Runner, test_state
) -> None:
    """The bug is the double refusal carrying the message of the count the job did not ask for."""
    snapped_up = _first_unmeasured_count(mac_runner, test_state)
    snapped_down = snapped_up - 8
    test_state.state.app_settings.budget_mode = "standard"

    def refusal(clip_frames: int) -> str:
        with pytest.raises(HTTPError) as exc_info:
            mac_runner.run(clip_frames)
        return exc_info.value.detail

    up_only, down_only = refusal(snapped_up), refusal(snapped_down)
    assert up_only != down_only, "the two counts must refuse for different reasons"
    assert refusal(snapped_down + 1) == up_only
