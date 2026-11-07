"""app/fusion.py 단위 테스트 (DB·GPU 없이, 가짜 후보로).
    python -m unittest tests.test_fusion -v
"""
import unittest

from app import fusion
from app.fusion import FusionParams


def c(video, off, score):
    return {"video_id": video, "off": off, "score": score}


class Windows(unittest.TestCase):
    def test_last_window_reaches_end(self):
        w = fusion.make_windows(20.0, 6.0, 3.0)
        self.assertEqual(w[0], (0.0, 6.0))
        self.assertEqual(w[-1][1], 20.0)
        self.assertEqual(w[-1], (14.0, 20.0))  # 끝에 닿게 당겨진 마지막 창

    def test_exact_fit_has_no_duplicate_window(self):
        self.assertEqual(fusion.make_windows(12.0, 6.0, 3.0), [(0.0, 6.0), (3.0, 9.0), (6.0, 12.0)])

    def test_short_clip_is_one_window(self):
        self.assertEqual(fusion.make_windows(4.0, 6.0, 3.0), [(0.0, 4.0)])

    def test_owned_ranges_tile_the_clip(self):
        ws = fusion.make_windows(20.0, 6.0, 3.0)
        own = fusion.owned_ranges(ws)
        self.assertEqual(own[0][0], 0.0)
        self.assertEqual(own[-1][1], 20.0)
        for (_, hi), (lo, _) in zip(own, own[1:]):
            self.assertAlmostEqual(hi, lo)  # 빈틈도 겹침도 없음


class Rrf(unittest.TestCase):
    def test_agreement_adds_scores(self):
        p = FusionParams(k=10, w_vis=1.0)
        out = fusion.rrf_window([c("A", 5.0, 90)], [c("A", 6.0, 3)], p)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["score"], 2 / 11)
        self.assertEqual((out[0]["audio_rank"], out[0]["vis_rank"]), (1, 1))
        self.assertEqual(out[0]["off"], 5.0)  # 소리 오프셋 우선

    def test_far_offsets_stay_separate(self):
        out = fusion.rrf_window([c("A", 5.0, 90)], [c("A", 12.0, 3)], FusionParams(group_tol=3.0))
        self.assertEqual(len(out), 2)

    def test_uses_rank_not_score(self):
        # 소리 점수가 화면보다 수십 배 커도 순위만 본다
        p = FusionParams(k=10, w_vis=1.0)
        out = fusion.rrf_window([c("A", 0, 1000), c("B", 0, 999)], [c("B", 0, 0.5), c("A", 0, 0.4)], p)
        self.assertAlmostEqual(out[0]["score"], 1 / 11 + 1 / 12)
        self.assertAlmostEqual(out[1]["score"], 1 / 12 + 1 / 11)  # A와 B가 정확히 동점

    def test_weight_breaks_swap_tie(self):
        a, v = [c("DECOY", 0, 500)], [c("TRUE", 0, 4)]
        self.assertEqual(fusion.rrf_window(a, v, FusionParams(w_vis=1.0))[0]["video_id"], "DECOY")  # 동점이면 소리 먼저
        self.assertEqual(fusion.rrf_window(a, v, FusionParams(w_vis=1.2))[0]["video_id"], "TRUE")

    def test_one_method_missing(self):
        out = fusion.rrf_window([], [c("A", 3, 2), c("B", 9, 1)], FusionParams(k=10))
        self.assertEqual([o["video_id"] for o in out], ["A", "B"])
        self.assertIsNone(out[0]["audio_rank"])

    def test_both_empty(self):
        self.assertEqual(fusion.rrf_window([], [], FusionParams()), [])


class OffsetGuard(unittest.TestCase):
    P = FusionParams(k=1.0, group_tol=1e6, guard_tol=3.0)

    def test_audio_rank1_offset_wins_even_if_far_from_vis(self):
        out = fusion.rrf_window([c("A", 84.6, 500)], [c("A", 219.0, 3)], self.P)
        self.assertEqual(out[0]["off"], 84.6)  # both: 화면 오프셋이 틀려도 소리 1등이 바로잡는다

    def test_audio_low_rank_far_from_vis_uses_vis_offset(self):
        # swap: 진짜 영상 T가 소리 목록 3등에 우연히 걸림 (오프셋 엉터리) -> 화면 오프셋을 쓴다
        audio = [c("DECOY", 7.0, 500), c("X", 1.0, 20), c("T", 900.0, 10)]
        out = fusion.rrf_window(audio, [c("T", 20.0, 4)], self.P)
        t = next(o for o in out if o["video_id"] == "T")
        self.assertEqual(t["off"], 20.0)

    def test_audio_low_rank_close_to_vis_uses_audio_offset(self):
        audio = [c("DECOY", 7.0, 500), c("T", 21.0, 10)]
        t = next(o for o in fusion.rrf_window(audio, [c("T", 20.0, 4)], self.P) if o["video_id"] == "T")
        self.assertEqual(t["off"], 21.0)  # 3초 이내면 소리(정밀)

    def test_audio_only_and_vis_only(self):
        self.assertEqual(fusion.rrf_window([c("A", 5.0, 9)], [], self.P)[0]["off"], 5.0)
        self.assertEqual(fusion.rrf_window([], [c("A", 6.0, 9)], self.P)[0]["off"], 6.0)


class Merge(unittest.TestCase):
    P = FusionParams(merge_tol=3.0)

    def win(self, i, best, atop=None):
        return {"own": (i * 3.0, i * 3.0 + 3.0), "best": best, "audio_top": atop}

    def best(self, v, off, s=0.1):
        return {"video_id": v, "off": off, "score": s}

    def test_consecutive_same_offset_merge(self):
        ws = [self.win(i, self.best("A", 10.0 + 0.5 * i)) for i in range(4)]
        segs = fusion.merge_windows(ws, self.P)
        self.assertEqual(len(segs), 1)
        self.assertEqual((segs[0]["q_start"], segs[0]["q_end"]), (0.0, 12.0))

    def test_offset_jump_splits(self):
        ws = [self.win(0, self.best("A", 10)), self.win(1, self.best("A", 10)),
              self.win(2, self.best("A", 40)), self.win(3, self.best("A", 40))]
        self.assertEqual([(s["q_start"], s["q_end"]) for s in fusion.merge_windows(ws, self.P)], [(0.0, 6.0), (6.0, 12.0)])

    def test_video_change_splits_compilation(self):
        ws = [self.win(0, self.best("A", 10)), self.win(1, self.best("B", 10)), self.win(2, self.best("B", 10))]
        segs = fusion.merge_windows(ws, self.P)
        self.assertEqual([s["video_id"] for s in segs], ["A", "B"])
        self.assertEqual(segs[1]["q_start"], 3.0)

    def test_none_breaks(self):
        ws = [self.win(0, self.best("A", 10)), self.win(1, None), self.win(2, self.best("A", 10))]
        self.assertEqual(len(fusion.merge_windows(ws, FusionParams(bridge=False))), 2)

    def test_bridge_absorbs_single_dissent(self):
        ws = [self.win(0, self.best("A", 10)), self.win(1, self.best("X", 3)), self.win(2, self.best("A", 10))]
        segs = fusion.merge_windows(ws, FusionParams(bridge=True))
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["q_end"], 9.0)
        self.assertEqual(len(fusion.merge_windows(ws, FusionParams(bridge=False))), 3)

    def test_bridge_does_not_touch_two_in_a_row(self):
        ws = [self.win(0, self.best("A", 10)), self.win(1, self.best("X", 3)), self.win(2, self.best("X", 3)),
              self.win(3, self.best("A", 10))]
        self.assertEqual(len(fusion.merge_windows(ws, FusionParams(bridge=True))), 3)

    def test_min_windows(self):
        ws = [self.win(0, self.best("A", 10)), self.win(1, None), self.win(2, self.best("B", 5)), self.win(3, self.best("B", 5))]
        segs = fusion.merge_windows(ws, FusionParams(bridge=False, min_windows=2))
        self.assertEqual([s["video_id"] for s in segs], ["B"])

    def test_start_is_offset_plus_q_start(self):
        ws = [self.win(1, self.best("A", 10.0)), self.win(2, self.best("A", 10.0))]
        seg = fusion.to_segment(fusion.merge_windows(ws, self.P)[0])
        self.assertEqual(seg["q_start"], 3.0)
        self.assertEqual(seg["start"], 13.0)  # 질의 3초 지점이 원본 13초 = 질의 0초가 원본 10초


class Path(unittest.TestCase):
    P = FusionParams(k=1.0, switch=1.0, merge_tol=3.0)  # 갈아타는 벌점 = 1/(1+1) = 0.5

    def cands(self, *specs):
        return [[{"video_id": v, "off": o, "score": s} for v, o, s in lst] for lst in specs]

    def test_single_outlier_window_follows_neighbours(self):
        cs = self.cands([("A", 10, 0.5)], [("B", 40, 0.5), ("A", 10, 0.3)], [("A", 10, 0.5)])
        path = fusion.pick_path(cs, self.P)
        self.assertEqual([c["video_id"] for c in path], ["A", "A", "A"])  # B(0.5) 대신 A(0.3): 갈아타는 벌점 2번(1.0) > 이득 0.2

    def test_real_change_is_followed(self):
        cs = self.cands(*([[("A", 10, 0.5)]] * 3 + [[("B", 5, 0.5), ("A", 10, 0.1)]] * 3))
        path = fusion.pick_path(cs, self.P)
        self.assertEqual([c["video_id"] for c in path], ["A"] * 3 + ["B"] * 3)  # 3창 이어 바뀌면 이득 1.2 > 벌점 0.5

    def test_zero_switch_is_independent_argmax(self):
        cs = self.cands([("A", 10, 0.5)], [("B", 40, 0.5), ("A", 10, 0.4)], [("A", 10, 0.5)])
        path = fusion.pick_path(cs, FusionParams(k=1.0, switch=1e-9, merge_tol=3.0))
        self.assertEqual([c["video_id"] for c in path], ["A", "B", "A"])

    def test_empty_window_gives_none(self):
        path = fusion.pick_path(self.cands([("A", 10, 0.5)], [], [("A", 10, 0.5)]), self.P)
        self.assertIsNone(path[1])
        self.assertEqual(path[0]["video_id"], "A")

    def test_build_result_with_switch_keeps_one_segment(self):
        ws = fusion.make_windows(12.0, 6.0, 3.0)
        raws = [{"a": a, "b": b, "audio": [c("A", 10, 100)], "vis": [c("A", 10, 3)]} for a, b in ws]
        raws[1]["audio"] = [c("Z", 70, 100)]  # 가운데 창만 소리가 엉뚱한 곳
        segs = fusion.build_result(raws, FusionParams(k=1.0, switch=2.0, bridge=False))["segments"]
        self.assertEqual([s["video_id"] for s in segs], ["A"])


class EndToEnd(unittest.TestCase):
    def raw(self, a, b, audio, vis):
        return {"a": a, "b": b, "audio": audio, "vis": vis}

    def test_swap_decoy_follows_weight_and_reports_audio_source(self):
        raws = [self.raw(i * 3.0, i * 3.0 + 6.0, [c("DECOY", 7.0, 300)], [c("TRUE", 20.0, 4)]) for i in range(3)]
        seg = fusion.build_result(raws, FusionParams(w_vis=1.2))["segments"]
        self.assertEqual(len(seg), 1)
        self.assertEqual(seg[0]["video_id"], "TRUE")
        self.assertEqual(seg[0]["audio_source"], "DECOY")
        tied = fusion.build_result(raws, FusionParams(w_vis=1.0))["segments"]
        self.assertEqual(tied[0]["video_id"], "DECOY")  # 가중이 같으면 동점 -> 소리 (순위만으로는 못 가른다)

    def test_nothing_found_gives_no_segment(self):
        res = fusion.build_result([self.raw(0.0, 6.0, [], [])], FusionParams())
        self.assertEqual(res, {"segments": [], "candidates": []})

    def test_compilation_boundary_is_not_smeared(self):
        # 질의 0~9초는 A, 9~18초는 B. 창 6초/간격 3초: 6~12초 창은 섞여 있어 어느 쪽이든 1등이 하나 나옴
        wins = fusion.make_windows(18.0, 6.0, 3.0)
        raws = []
        for a, b in wins:
            v = "A" if (a + b) / 2 < 9 else "B"
            raws.append(self.raw(a, b, [c(v, 5.0, 100)], [c(v, 5.0, 3)]))
        segs = fusion.build_result(raws, FusionParams())["segments"]
        self.assertEqual([s["video_id"] for s in segs], ["A", "B"])
        self.assertEqual(segs[0]["q_end"], segs[1]["q_start"])
        self.assertEqual((segs[0]["q_start"], segs[1]["q_end"]), (0.0, 18.0))

    def test_candidates_sum_over_windows(self):
        raws = [self.raw(i * 3.0, i * 3.0 + 6.0, [c("A", 5.0, 100)], [c("A", 5.5, 3)]) for i in range(3)]
        cands = fusion.build_result(raws, FusionParams(k=10, w_vis=1.0))["candidates"]
        self.assertEqual(cands[0]["video_id"], "A")
        self.assertAlmostEqual(cands[0]["score"], 3 * 2 / 11, places=3)
        self.assertEqual(len(cands), 1)  # 오프셋이 비슷하니 한 후보
        self.assertAlmostEqual(cands[0]["start_sec"], 5.0, delta=0.6)


if __name__ == "__main__":
    unittest.main()
