"""channels.yaml → 채널별 선정 규칙 (설계 결정 ⑤⑥⑪)."""
import random
from collections import defaultdict

import yaml

SKIP_TITLES = {"Private video", "Deleted video"}  # 재생목록에 남은 비공개·삭제 영상
UNTAGGED = "(태그 없음)"


def load_rules(path="channels.yaml"):
    """{채널 URL: 규칙}. 공식은 long_limit·shorts, 팬 클립은 limit"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rules = {}
    for ch in cfg["official"]:
        rules[ch["url"]] = {
            "official": True,
            "long_limit": ch["long_limit"],
            "shorts": ch.get("shorts"),  # 없으면 쇼츠는 고르지 않음
            "exclude": cfg["exclude_title"],
            "require": ch.get("require_title", []),
        }
    for ch in cfg["fan_clips"]:  # 팬 클립은 거르지 않고 최신순으로 limit까지
        rules[ch["url"]] = {"official": False, "limit": ch["limit"], "exclude": [], "require": []}
    return rules


def wanted(title, rule):
    if title in SKIP_TITLES:
        return False
    if any(word in title for word in rule["exclude"]):
        return False
    if rule["require"] and not any(word in title for word in rule["require"]):
        return False
    return True


def member_of(title, members):
    """제목에 가장 먼저 나오는 멤버 이름. 없으면 UNTAGGED"""
    found = [(title.find(m), m) for m in members if m in title]
    return min(found)[1] if found else UNTAGGED


def sample_shorts(shorts, cfg, seed=42):
    """[(영상 ID, 제목)] → [(영상 ID, 층)]. 층마다 무작위로 뽑되 시드를 고정해 다시 돌려도 같은 결과"""
    strata = defaultdict(list)
    for video_id, title in shorts:
        strata[member_of(title, cfg["members"])].append(video_id)
    rng = random.Random(seed)
    picked = []
    for stratum in sorted(strata):
        n = cfg["untagged"] if stratum == UNTAGGED else cfg["per_member"]
        picked += [(v, stratum) for v in rng.sample(strata[stratum], min(n, len(strata[stratum])))]
    return picked
