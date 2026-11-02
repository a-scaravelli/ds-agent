"""FileLeadInput (cases 14 to 19): missing, malformed, partial, consumed and edited files; notices; never edited."""

import os

import pytest
from orchestrator_helpers import (
    GUIDE_CONTINUE,
    PLAN_APPROVE,
    REVIEW_ACCEPT,
    World,
    drive_to_review,
)
from registry_helpers import make_cycle

from kaggle_agent.orchestrator import FileLeadInput, LeadFileError
from kaggle_agent.orchestrator.lead_file import read_lead
from kaggle_agent.registry import SqliteRegistry
from kaggle_agent.review import Accept, Approve, LeadContext, LeadResponse, Revise

PLAN = LeadContext("C1", "AWAITING_PLAN_APPROVAL")
REVIEW = LeadContext("C1", "AWAITING_REVIEW")


def guidance_ctx(position):
    return LeadContext("C1", "AWAITING_GUIDANCE", round_position=position)


@pytest.fixture
def lead(tmp_path):
    registry = SqliteRegistry()
    registry.add_cycle(make_cycle("C1"))
    path = tmp_path / "lead.toml"
    return FileLeadInput(path, tmp_path, registry, "C1"), path


# 14
def test_a_missing_file_an_empty_file_or_another_table_returns_none(lead):
    reader, path = lead
    assert reader.plan_decision(PLAN) is None and reader.guidance(guidance_ctx(1)) is None
    assert reader.review_decision(REVIEW) is None
    path.write_text("", encoding="utf-8")
    assert reader.plan_decision(PLAN) is None and reader.review_decision(REVIEW) is None
    path.write_text('[review]\ndecision = "accept"\n', encoding="utf-8")
    assert reader.plan_decision(PLAN) is None and reader.guidance(guidance_ctx(1)) is None
    assert reader.review_decision(REVIEW) == Accept()
    path.write_text("[plan]\n[guidance]\n", encoding="utf-8")  # empty tables are not answers
    assert reader.plan_decision(PLAN) is None and reader.guidance(guidance_ctx(1)) is None


def test_valid_decisions_are_returned(lead):
    reader, path = lead
    path.write_text('[plan]\ndecision = "approve"\n[guidance]\nround = 2\naction = "guidance"\ntext = "try x"\n',
                    encoding="utf-8")
    assert reader.plan_decision(PLAN) == Approve()
    assert reader.guidance(guidance_ctx(2)) == LeadResponse("guidance", "try x")
    path.write_text('[plan]\ndecision = "revise"\nplan = 1\ntext = "use trees"\n', encoding="utf-8")
    assert reader.plan_decision(PLAN) == Revise("use trees")


BAD = [
    ("[plan\nx", "not valid TOML"),
    ('[plan]\ndecision = "approve"\nextra = 1\n', "plan.extra"),
    ('submit = true\n', "submit"),
    ('[plan]\ndecision = "maybe"\n', "plan.decision"),
    ('[plan]\ntext = "x"\n', "plan.decision"),
    ('[plan]\ndecision = "revise"\nplan = 1\n', "plan.text"),
    ('[plan]\ndecision = "revise"\ntext = "x"\n', "plan.plan"),
    ('[guidance]\naction = "continue"\n', "guidance.round"),
    ('[guidance]\nround = 1\n', "guidance.action"),
    ('[guidance]\nround = 1\naction = "protocol_change"\n', "guidance.action"),
    ('[guidance]\nround = 1\naction = "guidance"\n', "guidance.text"),
    ('[guidance]\nround = 0\naction = "continue"\n', "guidance.round"),
    ('[review]\ndecision = "reject"\n', "review.decision"),
    ('plan = 3\n', "plan"),
]


@pytest.mark.parametrize("text, key", BAD)
# 15
def test_a_malformed_or_invalid_file_raises_with_the_key_path(lead, text, key):
    reader, path = lead
    path.write_text(text, encoding="utf-8")
    for call in (lambda: reader.plan_decision(PLAN), lambda: reader.guidance(guidance_ctx(1)),
                 lambda: reader.review_decision(REVIEW)):
        with pytest.raises(LeadFileError) as error:
            call()
        assert key in str(error.value)
    with pytest.raises(LeadFileError):
        read_lead(path)


def test_an_invalid_lead_file_ends_the_command_with_the_lead_file_code_and_changes_nothing(tmp_path):
    world = World(tmp_path)
    assert world.run("start")[0] == 10
    calls = len(world.claude.prompts)
    world.lead('[plan]\ndecision = "approve"\nsubmit = true\n')
    code, text = world.run("resume")
    assert code == 12 and "plan.submit" in text and "did not change" in text
    assert world.state() == "AWAITING_PLAN_APPROVAL" and len(world.claude.prompts) == calls
    world.lead('[plan]\ndecision = "maybe"\n')
    assert world.run("resume")[0] == 12
    world.lead(PLAN_APPROVE)  # fixing the file is enough
    assert world.run("resume")[0] == 10 and world.state() == "AWAITING_GUIDANCE"


# 16
def test_a_partial_file_leaves_the_plan_waiting_and_status_says_which_keys_to_fill(tmp_path):
    world = World(tmp_path)
    assert world.run("start")[0] == 10
    world.lead(GUIDE_CONTINUE.format(n=1))
    code, text = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_PLAN_APPROVAL"
    code, status = world.run("status")
    assert code == 0 and "[plan]" in status and 'decision = "approve"' in status and 'decision = "revise"' in status
    assert str(world.lead_file) in status


# 17
def test_consumed_decisions_are_ignored(lead):
    reader, path = lead
    path.write_text('[guidance]\nround = 1\naction = "guidance"\ntext = "t"\n', encoding="utf-8")
    assert reader.guidance(guidance_ctx(1)) == LeadResponse("guidance", "t")
    assert reader.guidance(guidance_ctx(2)) is None  # round 2 does not reuse round 1's entry
    assert any("stale guidance for round 1 ignored; round 2 is waiting" in n for n in reader.emitted)
    path.write_text('[guidance]\nround = 3\naction = "continue"\n', encoding="utf-8")
    assert reader.guidance(guidance_ctx(2)) is None  # an entry for a later round is not used early


def test_applied_guidance_and_revision_are_not_applied_again(tmp_path):
    world = World(tmp_path)
    world.run("start")
    world.lead('[plan]\ndecision = "revise"\nplan = 1\ntext = "prefer simple models"\n')
    code, _ = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_PLAN_APPROVAL"
    coordinator = [p for p, k in zip(world.claude.prompts, world.claude.kinds) if k == "coordinator"]
    assert len(coordinator) == 2 and "prefer simple models" in coordinator[1] and "prefer simple models" not in coordinator[0]
    # the stale revision stays in the file: plan 2 is still waiting
    code, _ = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_PLAN_APPROVAL"
    assert len([k for k in world.claude.kinds if k == "coordinator"]) == 2
    # approve plan 2 and answer round 1 in one edit: the old revision entry stays in the file, ignored
    world.lead('[plan]\ndecision = "approve"\n[guidance]\nround = 1\naction = "guidance"\ntext = "try scaling"\n')
    code, _ = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_GUIDANCE"  # round 2 waits: round 1's entry is stale
    code, text = world.run("resume")
    assert code == 10 and "round 2" in text
    registry = world.registry()
    assert [g.text for g in registry.list_guidance("toy-1")] == ["try scaling"]
    registry.close()


# 18
def test_a_file_edited_while_waiting_is_read_again_at_every_call(tmp_path):
    world = World(tmp_path)
    world.run("start")
    world.lead('[plan]\ndecision = "maybe"\n')
    assert world.run("resume")[0] == 12
    world.lead("")
    assert world.run("resume")[0] == 10 and world.state() == "AWAITING_PLAN_APPROVAL"
    world.lead(PLAN_APPROVE)
    assert world.run("resume")[0] == 10 and world.state() == "AWAITING_GUIDANCE"
    world.lead(PLAN_APPROVE + '[guidance]\nround = 1\naction = "continue"\n')
    assert world.run("resume")[0] == 10 and world.state() == "AWAITING_GUIDANCE"  # round 2 now


# 19
def test_the_lead_file_is_never_edited_and_the_file_flow_writes_the_notices(tmp_path):
    world = World(tmp_path)
    world.run("start")
    text = PLAN_APPROVE + GUIDE_CONTINUE.format(n=1)
    world.lead(text)
    os.utime(world.lead_file, (1_700_000_000, 1_700_000_000))
    before = (world.lead_file.read_bytes(), world.lead_file.stat().st_mtime_ns)
    code, out = world.run("resume")
    assert code == 10 and world.state() == "AWAITING_GUIDANCE"
    assert (world.lead_file.read_bytes(), world.lead_file.stat().st_mtime_ns) == before
    notices = (world.project / "NOTICES.txt").read_text(encoding="utf-8")
    assert "waiting for plan approval" in notices and "stale guidance for round 1 ignored" in notices
    assert "notice: stale guidance for round 1 ignored; round 2 is waiting" in out
    assert "notice: waiting for guidance after round 2" in out
    code, again = world.run("resume")  # nothing new to say: each notice is written once
    assert "notice:" not in again
    assert (world.lead_file.read_bytes(), world.lead_file.stat().st_mtime_ns) == before


def test_the_loop_notice_goes_to_the_notices_file(lead):
    reader, path = lead
    reader.notice("a protocol change is not guidance")
    reader.notice("a protocol change is not guidance")
    lines = (path.parent / "NOTICES.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and lines[0].endswith("a protocol change is not guidance")


def test_the_review_is_waiting_until_the_file_accepts(tmp_path):
    world = World(tmp_path)
    drive_to_review(world)
    code, text = world.run("status")
    assert "[review]" in text and 'decision = "accept"' in text
    assert REVIEW_ACCEPT.strip().splitlines()[0] == "[review]"
