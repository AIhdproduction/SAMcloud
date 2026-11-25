"""AI assistant domains: registry choice (indoor / outdoor / none), unchanged indoor tools and prompts, tool-tier plumbing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from assistant_synthetic import LEVEL_ID, make_plan, make_project
from samcloud.assistant import agent as agent_module
from samcloud.assistant import credentials as ai_creds
from samcloud.assistant import settings as ai_settings
from samcloud.assistant.domains import registry
from samcloud.assistant.domains.base import TIER_ELEMENTS, TIER_FULL, ToolDef, UnavailableDomain, normalize_tier
from samcloud.assistant.domains.indoor import IndoorDomain
from samcloud.assistant.domains.outdoor import OutdoorDomain
from samcloud.assistant.domains.registry import OutdoorPendingDomain, domain_for, resolve_tier, unsupported_reason
from samcloud.assistant.prompts import system_prompt
from samcloud.assistant.providers.base import ToolCall
from samcloud.assistant.service import AssistantService
from samcloud.assistant.tools import ToolContext, ToolResult, dispatch, tool_names, tool_specs
from test_assistant_agent import FakeProvider, call, say, settings_for

DUMMY_KEY = "sk-test-DUMMY-0000"
SITE = "site"

# indoor tool list of phase 1 (names, and a hash of names + JSON schemas): must not change through the domain layer
INDOOR_TOOL_NAMES = ["list_elements", "element_details", "points_in_box", "section", "fit_surface", "move_wall",
                     "set_wall_thickness", "delete_wall", "set_opening", "delete_opening", "add_furniture", "move_furniture",
                     "rotate_furniture", "delete_furniture", "set_furniture_class", "rename_room"]
INDOOR_SCHEMA_SHA256 = "9beeb674667ad0f024afafcc7b0edd49384a4f9717f04816b040b91dc66ab546"
# SHA-256 of the phase 1 system prompts (elements tier, proposal wording)
PROMPT_SHA256 = {"de": "4d4139f41878e6e2df0c6b48c62c9d7291a12e5d5a06f0d2a00e82e7edfe54d3",
                 "en": "f47721d698c9bcddc9b56af97c66abaa39d4595b15413be8d95adbe522c8c2e7"}


@pytest.fixture(autouse=True)
def _private_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real user's AI settings / credentials."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


@pytest.fixture
def clean_registry() -> Any:
    """Registry back to the built-ins after a test registered its own domain."""
    registry._reset_for_tests()
    yield registry
    registry._reset_for_tests()


def _enable(provider: str = "openai", **overrides: Any) -> None:
    values: dict[str, Any] = {"enabled": True, "provider": provider, "model": "fake-model", "cost_limit_request_usd": 0.0,
                              "cost_limit_project_day_usd": 0.0, "privacy_ack_cloud": True, **overrides}
    settings = ai_settings.AssistantSettings(**values)
    ai_settings.save_settings(settings)
    ai_creds.save_ai_credentials({"openai_api_key": DUMMY_KEY, "ollama_url": "http://127.0.0.1:9"})


def _project_file(project: Path, *, class_set: str, drone: bool = False) -> None:
    project.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {"project": {"name": "synthetic"}, "settings": {"classification": {"class_set": class_set}},
                                "workflow": {}}
    if drone:
        document["acquisition"] = {"type": "drone_gps"}
    (project / "project.samcloud.yml").write_text(yaml.safe_dump(document), encoding="utf-8")


def _drone_project(root: Path, *, state_only: bool = False) -> Path:
    project = root / "drone"
    if state_only:
        _project_file(project, class_set="outdoor")
        (project / "products" / "environment").mkdir(parents=True)
        (project / "products" / "environment" / "state.json").write_text('{"steps": {}}', encoding="utf-8")
    else:
        _project_file(project, class_set="outdoor", drone=True)
    return project


def _schema_hash(specs: list[Any]) -> str:
    return hashlib.sha256(json.dumps([[s.name, s.parameters] for s in specs], sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------ registry

def test_registry_indoor_project_with_plan_model(tmp_path: Path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    domain = domain_for(project, LEVEL_ID)
    assert isinstance(domain, IndoorDomain) and domain.kind == "indoor" and domain.level_id == LEVEL_ID
    assert domain.availability() == (True, None) and unsupported_reason(project, LEVEL_ID) is None
    assert domain.export_actions("accept") == ["export", "export_ifc"] == domain.export_actions("undo")
    assert domain_for(project, LEVEL_ID, mode="indoor").kind == "indoor"


@pytest.mark.parametrize("state_only", [False, True])
def test_registry_drone_project_gets_the_outdoor_domain(tmp_path: Path, state_only: bool) -> None:
    project = _drone_project(tmp_path, state_only=state_only)
    domain = domain_for(project, SITE)
    assert isinstance(domain, OutdoorDomain) and not isinstance(domain, OutdoorPendingDomain) and domain.kind == "outdoor"
    # no scene.pkl yet: the outdoor domain says why (the environment chain has not built the site model)
    assert domain.availability() == (False, "assistant.agent.outdoor_no_scene")
    assert "assistant.agent.outdoor_no_scene" in registry.DOMAIN_REASONS
    assert unsupported_reason(project, SITE) == "assistant.agent.outdoor_no_scene"
    assert unsupported_reason(project, SITE, mode="outdoor") == "assistant.agent.outdoor_no_scene"
    assert domain.supported_tiers == (TIER_ELEMENTS, TIER_FULL)
    elements_tools = [tool.name for tool in domain.tools(TIER_ELEMENTS)]
    full_tools = [tool.name for tool in domain.tools(TIER_FULL)]
    assert {"add_tree", "set_tree", "add_manhole", "offset_fence", "set_site_wall", "set_curb"} <= set(elements_tools)
    assert "adjust_terrain" not in elements_tools and {"adjust_terrain", "apply_to_category", "set_building"} <= set(full_tools)
    assert domain.export_actions("accept") == ["env:scene,export,qa:cad_only"] == domain.export_actions("undo")
    # a mode override to indoor is not an environment-chain project (same rule as the DXF page)
    assert domain_for(project, SITE, mode="indoor") is None
    assert unsupported_reason(project, SITE, mode="indoor") == "assistant.agent.no_plan"


def test_registry_ortho_outdoor_without_environment_chain(tmp_path: Path) -> None:
    project = tmp_path / "ortho_outdoor"
    _project_file(project, class_set="outdoor")
    assert domain_for(project, SITE) is None
    assert unsupported_reason(project, SITE) == "assistant.agent.needs_environment"
    assert unsupported_reason(project, "level_x", mode="outdoor") == "assistant.agent.needs_environment"
    # an outdoor ortho site with its own plan_model keeps working as before (indoor domain, DXF export only)
    from samcloud.bim.plan_model import save_plan_model
    from samcloud.ortho.plan import clean_plan_cache_dir

    save_plan_model(make_plan(), clean_plan_cache_dir(project, None))
    domain = domain_for(project, SITE)
    assert isinstance(domain, IndoorDomain) and domain.export_actions() == ["export"]


def test_registry_nothing_means_no_plan(tmp_path: Path) -> None:
    assert domain_for(tmp_path / "nothing", LEVEL_ID) is None
    assert unsupported_reason(tmp_path / "nothing", LEVEL_ID) == "assistant.agent.no_plan"
    indoor_without_plan = tmp_path / "indoor"
    _project_file(indoor_without_plan, class_set="indoor")
    assert unsupported_reason(indoor_without_plan, LEVEL_ID, mode="indoor") == "assistant.agent.no_plan"


class _FakeOutdoor(UnavailableDomain):
    kind = "outdoor"
    supported_tiers = (TIER_ELEMENTS, TIER_FULL)

    def availability(self) -> tuple[bool, str | None]:
        return True, None


def test_register_domain_replaces_the_outdoor_domain(tmp_path: Path, clean_registry: Any) -> None:
    project = _drone_project(tmp_path)
    assert isinstance(domain_for(project, SITE), OutdoorDomain)  # the outdoor module replaced the placeholder on load
    clean_registry.register_domain("outdoor", lambda p, level, mode: _FakeOutdoor(p, level, mode=mode),
                                   clean_registry.uses_environment_chain)
    domain = domain_for(project, SITE)
    assert isinstance(domain, _FakeOutdoor) and unsupported_reason(project, SITE) is None
    assert clean_registry.registered_kinds() == ["outdoor", "indoor"]  # the replacement keeps the outdoor priority
    indoor = make_project(tmp_path, with_cloud=False)
    assert isinstance(domain_for(indoor, LEVEL_ID), IndoorDomain)


# ------------------------------------------------------------------------------------------------------- indoor tools

def test_indoor_tool_list_identical_to_phase_one(tmp_path: Path) -> None:
    domain = IndoorDomain(make_project(tmp_path, with_cloud=False), LEVEL_ID)
    legacy = tool_specs()
    for images in (False, True):
        specs = tool_specs(domain, TIER_ELEMENTS, images_allowed=images)
        assert [s.name for s in specs] == INDOOR_TOOL_NAMES
        assert _schema_hash(specs) == INDOOR_SCHEMA_SHA256
        plain = tool_specs(images_allowed=images)
        assert [(s.name, s.description, s.parameters) for s in specs] == [(s.name, s.description, s.parameters) for s in plain]
        # full control (package K): the phase 1 tools unchanged first, then the full-tier operations
        full = tool_specs(domain, TIER_FULL, images_allowed=images)
        assert _schema_hash(full[:len(INDOOR_TOOL_NAMES)]) == INDOOR_SCHEMA_SHA256
        assert len(full) > len(INDOOR_TOOL_NAMES)
    assert [s.name for s in legacy] == INDOOR_TOOL_NAMES == tool_names()
    assert [d.name for d in domain.read_tools()] == INDOOR_TOOL_NAMES[:5]
    assert all(d.is_edit for d in domain.edit_tools()) and not any(d.is_edit for d in domain.read_tools())


def test_indoor_domain_wraps_the_existing_helpers(tmp_path: Path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    domain = IndoorDomain(project, LEVEL_ID)
    plan = domain.current_model()
    assert {row["id"] for row in domain.element_list({"kind": "wall"})} == {"W001", "W002", "W003", "W004", "W005"}
    assert domain.element_details("O001")["found"] is True
    door = domain.selection_from_global_id("", [1.0, 2.0])
    assert door == {"type": "point", "x": 1.0, "y": 2.0}
    sel = {"type": "element", "kind": "opening", "id": "O001"}
    assert len(domain.outline(sel)) == 1 and len(domain.outline(sel)[0]) == 4
    assert domain.z_range(sel) is not None and domain.global_ids_for([sel])
    assert domain.refresh_selection(sel)["label"].endswith("O001")
    assert domain.refresh_selection({"type": "element", "kind": "wall", "id": "W999"}) is None
    assert domain.view_frame() == {"units_per_metre": float(plan.units_per_metre or 1.0),
                                   "floor_z": float(plan.floor_z) / float(plan.units_per_metre or 1.0)}
    backend = domain.history_backend()
    assert backend.kind == "indoor" and backend.base_filename == "plan_model.base.json"
    assert backend.semantic_hash(backend.load(backend.model_path(project, LEVEL_ID))) == backend.semantic_hash(plan)
    proposal = domain.new_proposal(plan)
    assert proposal.summary() == [] and domain.preview(proposal) == {"added": [], "removed": [], "moved": []}


# ----------------------------------------------------------------------------------------------------- tier settings

@pytest.mark.parametrize(("provider", "expected"), [("anthropic", True), ("openai", True), ("openrouter", True), ("ollama", False)])
def test_effective_full_control_defaults_per_provider(provider: str, expected: bool) -> None:
    settings = ai_settings.AssistantSettings(provider=provider)
    assert settings.full_control is None
    assert ai_settings.effective_full_control(settings) is expected
    assert ai_settings.effective_full_control(ai_settings.AssistantSettings(provider=provider, full_control=True)) is True
    assert ai_settings.effective_full_control(ai_settings.AssistantSettings(provider=provider, full_control=False)) is False
    assert registry.requested_tier(settings) == (TIER_FULL if expected else TIER_ELEMENTS)


def test_full_control_default_null_and_explicit_choice_survives_provider_change(tmp_path: Path) -> None:
    defaults = yaml.safe_load(ai_settings.DEFAULTS_PATH.read_text(encoding="utf-8"))
    assert "full_control" in defaults and defaults["full_control"] is None
    assert ai_settings.load_settings().full_control is None
    settings = ai_settings.load_settings()
    settings.provider, settings.full_control = "ollama", True
    ai_settings.save_settings(settings)
    loaded = ai_settings.load_settings()
    assert loaded.full_control is True and ai_settings.effective_full_control(loaded) is True
    loaded.provider = "anthropic"
    loaded.full_control = False
    ai_settings.save_settings(loaded)
    again = ai_settings.load_settings()
    again.provider = "ollama"  # switching provider never overwrites the user's choice
    ai_settings.save_settings(again)
    assert ai_settings.load_settings().full_control is False
    path = ai_settings.user_settings_path()
    for text, expected in (("null", None), ("default", None), ("yes", True), ("off", False)):
        path.write_text(f"provider: openai\nfull_control: {text}\n", encoding="utf-8")
        assert ai_settings.load_settings().full_control is expected


def test_resolve_tier_needs_domain_support(tmp_path: Path) -> None:
    cloud, local = settings_for("openai"), settings_for("ollama")
    indoor = IndoorDomain(make_project(tmp_path, with_cloud=False), LEVEL_ID)
    assert resolve_tier(indoor, cloud) == TIER_FULL  # indoor full control (package K)
    assert resolve_tier(indoor, local) == TIER_ELEMENTS
    full = _FakeOutdoor(tmp_path, SITE)
    assert resolve_tier(full, cloud) == TIER_FULL and resolve_tier(full, local) == TIER_ELEMENTS
    assert resolve_tier(full, settings_for("ollama", full_control=True)) == TIER_FULL
    assert resolve_tier(None, cloud) == TIER_ELEMENTS
    assert normalize_tier("FULL") == TIER_FULL and normalize_tier(None) == TIER_ELEMENTS and normalize_tier("x") == TIER_ELEMENTS


# ------------------------------------------------------------------------------------------------- tier in the turn

def _echo(name: str) -> Any:
    def handler(ctx: ToolContext, call_: ToolCall, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(json.dumps({"tool": name, "tier": ctx.tier}))
    return handler


class _TierDomain(IndoorDomain):
    """Indoor plan with one extra read tool in the full tier (stands in for a later package)."""

    supported_tiers = (TIER_ELEMENTS, TIER_FULL)

    def read_tools(self, tier: str = TIER_ELEMENTS) -> list[ToolDef]:
        tools = super().read_tools(tier)
        if tier == TIER_FULL:
            tools.append(ToolDef("project_overview", "Overview of the whole project.", {"type": "object", "properties": {}},
                                 _echo("project_overview")))
        return tools


def test_tier_reaches_tool_specs_and_dispatch(tmp_path: Path) -> None:
    domain = _TierDomain(make_project(tmp_path, with_cloud=False), LEVEL_ID)
    assert "project_overview" not in [s.name for s in tool_specs(domain, TIER_ELEMENTS)]
    assert "rename_room" in [s.name for s in tool_specs(domain, TIER_FULL)]
    assert "project_overview" in [s.name for s in tool_specs(domain, TIER_FULL)]
    base = domain.load_base()
    full = domain.tool_context(base, tier=TIER_FULL)
    assert json.loads(dispatch(full, ToolCall("1", "project_overview", {})).content) == {"tool": "project_overview", "tier": "full"}
    elements = domain.tool_context(base, tier=TIER_ELEMENTS)
    result = dispatch(elements, ToolCall("2", "project_overview", {}))
    assert result.error and result.error_key == "assistant.agent.unknown_tool"
    assert json.loads(dispatch(elements, ToolCall("3", "list_elements", {"kind": "wall"})).content)["total"] == 5


def test_system_prompt_unchanged_for_indoor_and_names_the_tier(tmp_path: Path) -> None:
    domain = IndoorDomain(make_project(tmp_path, with_cloud=False), LEVEL_ID)
    for lang in ("de", "en"):
        legacy = system_prompt(lang)
        assert hashlib.sha256(legacy.encode()).hexdigest() == PROMPT_SHA256[lang]
        assert system_prompt(lang, vocabulary=domain.vocabulary(lang), tier=TIER_ELEMENTS) == legacy
    full_de = system_prompt("de", vocabulary=domain.vocabulary("de"), tier=TIER_FULL)
    full_en = system_prompt("en", vocabulary=domain.vocabulary("en"), tier=TIER_FULL)
    assert "Vollkontrolle aktiv" in full_de and "direkt angewendet" in full_de and "nur ein Vorschlag" not in full_de
    assert "Full control active" in full_en and "applied directly" in full_en and "only a proposal" not in full_en
    assert "Innenprojekt" in full_de and "indoor project" in full_en


def test_run_turn_uses_the_domain_tools_and_tier_prompt(tmp_path: Path) -> None:
    project = make_project(tmp_path, with_cloud=False)
    domain = _TierDomain(project, LEVEL_ID)
    provider = FakeProvider([call("project_overview"), say("Nothing to change.")])
    result = agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt="overview", selection=[],
                                   settings=settings_for("openai"), creds={}, provider=provider, domain=domain)
    assert result.ok and result.proposal is None
    assert "project_overview" in provider.tools_seen[0]
    assert "Full control active" in provider.requests[0][0].content or "Vollkontrolle aktiv" in provider.requests[0][0].content
    assert result.tool_log[0]["tool"] == "project_overview" and result.tool_log[0]["ok"]
    # the same domain with a local model (full control off by default) gets the elements tier
    local = FakeProvider([say("ok")])
    agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt="overview", selection=[], settings=settings_for("ollama"),
                          creds={}, provider=local, domain=domain)
    assert "project_overview" not in local.tools_seen[0]
    assert "Full control" not in local.requests[0][0].content and "Vollkontrolle" not in local.requests[0][0].content
    # without a domain and without full control the turn is the phase 1 indoor turn
    plain = FakeProvider([say("ok")])
    agent_module.run_turn(project=project, level_id=LEVEL_ID, prompt="x", selection=[],
                          settings=settings_for("openai", full_control=False), creds={}, provider=plain)
    assert plain.tools_seen[0] == INDOOR_TOOL_NAMES
    assert hashlib.sha256(plain.requests[0][0].content.encode()).hexdigest() in PROMPT_SHA256.values()


# ------------------------------------------------------------------------------------------------------------ service

def test_service_reports_domain_and_tier(tmp_path: Path) -> None:
    _enable("openai")
    indoor = AssistantService(make_project(tmp_path, with_cloud=False), LEVEL_ID)
    assert indoor.is_available() == (True, None)
    assert (indoor.domain_kind, indoor.tool_tier, indoor.unsupported_reason) == ("indoor", TIER_FULL, None)  # cloud default
    _enable("openai", full_control=False)
    assert indoor.tool_tier == TIER_ELEMENTS
    _enable("openai")
    assert indoor.export_actions() == ["export", "export_ifc"]
    assert indoor.view_frame()["units_per_metre"] == pytest.approx(float(indoor.plan().units_per_metre or 1.0))
    drone = AssistantService(_drone_project(tmp_path), SITE)
    assert drone.is_available() == (False, "assistant.agent.outdoor_no_scene")
    assert (drone.domain_kind, drone.unsupported_reason) == ("outdoor", "assistant.agent.outdoor_no_scene")
    result = drone.ask("move the tree", [])
    assert result.error_kind == "unavailable" and result.error
    # with a (synthetic) site model the outdoor domain is available, full control is its default tier with a cloud provider
    from assistant_outdoor_synthetic import make_outdoor_project

    site = AssistantService(make_outdoor_project(tmp_path / "site_model"), SITE)
    assert site.is_available() == (True, None)
    assert (site.domain_kind, site.unsupported_reason, site.tool_tier) == ("outdoor", None, TIER_FULL)
    assert site.export_actions() == ["env:scene,export,qa:cad_only"]
    ortho = tmp_path / "ortho_outdoor"
    _project_file(ortho, class_set="outdoor")
    service = AssistantService(ortho, SITE, mode="outdoor")
    assert service.is_available() == (False, "assistant.agent.needs_environment") and service.domain_kind is None
    # settings reasons still come first
    _enable("openai", model="")
    assert drone.is_available() == (False, "assistant.agent.no_model")


def test_service_with_full_tier_domain(tmp_path: Path, clean_registry: Any) -> None:
    _enable("openai")
    project = make_project(tmp_path, with_cloud=False)
    clean_registry.register_domain("indoor", lambda p, level, mode: _TierDomain(p, level, mode=mode),
                                   lambda p, level, mode: True, priority=registry.PRIORITY_INDOOR)
    service = AssistantService(project, LEVEL_ID)
    assert service.tool_tier == TIER_FULL
    _enable("openai", full_control=False)
    assert service.tool_tier == TIER_ELEMENTS


# --------------------------------------------------------------------------------------------------------- hint line

def test_unavailable_reason_for_the_page(tmp_path: Path) -> None:
    from samcloud.app.components.assistant_panel import assistant_unavailable_reason

    drone = _drone_project(tmp_path)
    assert assistant_unavailable_reason(drone, SITE, mode="outdoor") is None  # assistant off: nothing
    _enable("openai")
    assert assistant_unavailable_reason(drone, SITE, mode="outdoor") == "assistant.agent.outdoor_no_scene"
    assert assistant_unavailable_reason(make_project(tmp_path, with_cloud=False), LEVEL_ID, mode="indoor") is None
    ai_creds.delete_ai_key("openai")  # a settings problem is not a project hint
    assert assistant_unavailable_reason(drone, SITE, mode="outdoor") is None
