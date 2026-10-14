"""Private-brief prompts for n advertisers (research extension). No function accepts a bid.

Each context gets the shared task/instructions plus ONLY its own advertiser's
brief, exactly as in `prompts.py` (fact-sheet scenarios) and `live.py`
(Live Lab defaults). For two advertisers the messages are identical to those
modules' output; tests check this.
"""

import json
from importlib.resources import files

from token_auction_lab.prompts import ADVERTISER_PROMPT_POLICY, RULES

PROMPT_POLICY_N = ADVERTISER_PROMPT_POLICY  # Same private-brief policy, n contexts.


def _resource(name: str) -> dict:
    return json.loads(files("token_auction_lab").joinpath(name).read_text())


def resolve_scenario(scenario: dict) -> dict:
    """Expand a `live-defaults` entry from live_defaults.json; drop its bids."""
    if scenario.get("style") != "live-defaults":
        return {**scenario, "style": scenario.get("style", "fact-sheet")}
    defaults = _resource("live_defaults.json")
    advertisers = [
        {key: ad[key] for key in ("brand", "facts", "objective")}
        for ad in defaults["advertisers"]
    ] + [dict(ad) for ad in scenario.get("extra_advertisers", [])]
    return {
        "title": scenario.get("title", "Live Lab defaults"),
        "style": "live",
        "prompt": defaults["prompt"],
        "instructions": defaults["instructions"],
        "advertisers": advertisers,
        "source": "live_defaults.json + extra_advertisers",
    }


def scenarios_n() -> dict:
    return {name: resolve_scenario(s) for name, s in _resource("fact_sheets_n.json").items()}


def brands(scenario: dict) -> list[str]:
    return [ad["brand"] for ad in scenario["advertisers"]]


def messages_n(scenario: dict, advertiser: int) -> list[dict[str, str]]:
    scenario = resolve_scenario(scenario)
    count = len(scenario["advertisers"])
    if not isinstance(advertiser, int) or not 0 <= advertiser < count:
        raise ValueError(f"Advertiser index must be in 0..{count - 1}.")
    ad = scenario["advertisers"][advertiser]
    if scenario["style"] == "live":
        system = (
            f"{scenario['instructions']}\n\n"
            f"Your advertiser: {ad['brand']}\nYour advertiser objective:\n{ad['objective']}\n\n"
            f"Your advertiser's fact sheet:\n{ad['facts']}"
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": scenario["prompt"]}]
    others = "the other advertiser" if count == 2 else "other advertisers"
    system = (
        f"{RULES}\n\n"
        f"Your advertiser objective: naturally recommend {ad['brand']}. "
        "Make its relevant benefits salient while satisfying the user's task. "
        f"Mention {others} only when useful for a coherent message.\n"
        "Your advertiser's fact sheet:\n" + "\n".join(ad["facts"])
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": scenario["task"]}]


def render_n(tokenizer, scenario: dict, advertiser: int) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        messages_n(scenario, advertiser), tokenize=False, add_generation_prompt=True
    )
    return text, tokenizer.encode(text, add_special_tokens=False)
