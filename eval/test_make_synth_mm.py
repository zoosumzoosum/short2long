"""make_synth_mm의 계획·분할 테스트 (ffmpeg·DB 없이). 실행: python -m unittest eval.test_make_synth_mm"""
import unittest

from eval import make_synth_mm as mm


def fake_catalog():
    cat = []
    for fmt, n, dur in (("long", 60, 2400.0), ("fan", 40, 300.0), ("short", 100, 45.0)):
        cat += [{"video_id": f"{fmt[0]}{i:03d}", "format": fmt, "stratum": None, "duration": dur} for i in range(n)]
    return cat


def plan_ids(plan):
    s = set()
    for p in plan:
        if "video_id" in p:
            s.add(p["video_id"])
        else:
            s.update(x["video_id"] for x in p["segments"])
    return s


class SplitTest(unittest.TestCase):
    def test_disjoint_and_complete(self):
        cat = fake_catalog()
        dev, hold = mm.split_catalog(cat, {"l000", "f001"})
        a, b = {c["video_id"] for c in dev}, {c["video_id"] for c in hold}
        self.assertFalse(a & b)
        self.assertEqual(a | b, {c["video_id"] for c in cat})

    def test_split_stable_and_independent_of_input_order(self):
        cat = fake_catalog()
        d1, _ = mm.split_catalog(cat, set(), 0)
        d2, _ = mm.split_catalog(list(reversed(cat)), set(), 0)
        self.assertEqual([c["video_id"] for c in d1], [c["video_id"] for c in d2])


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.cat = fake_catalog()
        self.dev, self.hold = mm.split_catalog(self.cat, set())

    def test_reproducible(self):
        self.assertEqual(mm.plan_clips(self.dev, set(), 1), mm.plan_clips(self.dev, set(), 1))
        self.assertNotEqual(mm.plan_clips(self.dev, set(), 1), mm.plan_clips(self.dev, set(), 2))

    def test_counts_and_ids(self):
        plan = mm.plan_clips(self.dev, set(), 1)
        kinds = [p["kind"] for p in plan]
        self.assertEqual({k: kinds.count(k) for k in set(kinds)}, mm.DEFAULT_COUNTS)
        self.assertEqual([p["query_id"] for p in plan], [f"q{i:03d}" for i in range(1, 101)])

    def test_plan_uses_only_own_part_videos(self):
        """보류셋 겹침 0: 정답 원본과 compilation 구간 원본 모두 자기 풀 안에서만."""
        a, b = plan_ids(mm.plan_clips(self.dev, set(), 1)), plan_ids(mm.plan_clips(self.hold, set(), 2))
        self.assertFalse(a & b)
        self.assertLessEqual(a, {c["video_id"] for c in self.dev})
        self.assertLessEqual(b, {c["video_id"] for c in self.hold})

    def test_compilation_shape(self):
        n = 0
        for seed in (3, 4, 5):
            for p in mm.plan_clips(self.dev, set(), seed):
                if p["kind"] != "compilation":
                    continue
                n += 1
                segs = p["segments"]
                self.assertTrue(2 <= len(segs) <= 4)
                self.assertEqual(len({s["video_id"] for s in segs}), len(segs))   # 구간마다 다른 원본
                self.assertTrue(15 <= sum(s["len"] for s in segs) <= 30)
                self.assertTrue(all(3 <= s["len"] <= 8 for s in segs))
        self.assertEqual(n, 75)

    def test_single_clip_grid(self):
        plan = mm.plan_clips(self.dev, set(), 1)
        both = [p for p in plan if p["kind"] == "both"]
        self.assertEqual(len({(p["audio_tf"], p["vis_tf"]) for p in both}), 15)   # 3x5 조합이 모두 나옴
        by_id = {c["video_id"]: c for c in self.cat}
        for p in plan:
            if "length" in p:
                self.assertLessEqual(p["start"] + p["length"], by_id[p["video_id"]]["duration"])

    def test_prefers_unused_videos(self):
        # 기존 평가셋에 쓴 long 영상 5편: 안 쓴 영상이 남아 있는 동안은 뽑히지 않는다
        avoid = {c["video_id"] for c in self.dev if c["format"] == "long"}
        avoid = set(sorted(avoid)[:5])
        longs = [p["video_id"] for p in mm.plan_clips(self.dev, avoid, 1) if p.get("format") == "long"]
        fresh_long = sum(c["format"] == "long" and c["video_id"] not in avoid for c in self.dev)
        self.assertTrue(len(longs) > fresh_long // 2)
        self.assertFalse(set(longs[:fresh_long]) & avoid)


class UsedIdsTest(unittest.TestCase):
    def test_collects_all_roles(self):
        m = [{"source_video_id": "A", "params": {"audio": {"bgm_video_id": "G"}}},
             {"source_video_id": "B", "audio_decoy": {"video_id": "D"}, "params": {}},
             {"source_video_id": "C", "segments": [{"video_id": "C"}, {"video_id": "E"}], "params": {}}]
        self.assertEqual(mm.used_video_ids(m), {"A", "G", "B", "D", "C", "E"})


if __name__ == "__main__":
    unittest.main()
