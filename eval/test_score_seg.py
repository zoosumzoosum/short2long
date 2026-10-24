"""구간 채점(SegR)·속은 비율 손계산 테스트. 실행: python -m unittest eval.test_score_seg"""
import unittest

from eval.score import group_by, score_queries, single_rows, summarize, summarize_seg


def comp():
    """24초 compilation: 0~8 A(원본 100초부터), 8~16 B(50초부터), 16~24 C(10초부터)"""
    return {"query_id": "c1", "kind": "compilation", "transform": "compilation", "format": "mixed",
            "source_video_id": "A", "start_sec": 100.0, "end_sec": 108.0, "query_dur_sec": 24.0,
            "segments": [{"q_start": 0, "q_end": 8, "video_id": "A", "start": 100.0},
                         {"q_start": 8, "q_end": 16, "video_id": "B", "start": 50.0},
                         {"q_start": 16, "q_end": 24, "video_id": "C", "start": 10.0}]}


def seg(qs, qe, vid, start, score=1.0):
    return {"q_start": qs, "q_end": qe, "video_id": vid, "start": start, "score": score}


def one(m, segments=None, candidates=None, tol=3.0):
    res = {"query_id": m["query_id"]}
    if segments is not None:
        res["segments"] = segments
    else:
        res["candidates"] = candidates
    rows, _ = score_queries([m], [res], tol)
    return rows[0]


class SegmentRuleTest(unittest.TestCase):
    def test_all_correct(self):
        r = one(comp(), [seg(0, 8, "A", 100), seg(8, 16, "B", 50), seg(16, 24, "C", 10)])
        self.assertEqual((r["seg_total"], r["seg_correct"], r["seg_wrong"], r["seg_missing"], r["seg_extra"]),
                         (3, 3, 0, 0, 0))

    def test_overlap_boundary_50_percent(self):
        # 정답 구간 0~8: 예측이 0~4(딱 50%)면 인정, 0~3.99면 겹침 부족 = 예측 없음
        self.assertEqual(one(comp(), [seg(0, 4, "A", 100)])["seg_correct"], 1)
        r = one(comp(), [seg(0, 3.99, "A", 100)])
        self.assertEqual((r["seg_correct"], r["seg_missing"]), (0, 3))

    def test_tolerance_boundary(self):
        self.assertEqual(one(comp(), [seg(0, 8, "A", 103.0)])["seg_correct"], 1)
        r = one(comp(), [seg(0, 8, "A", 103.01)])
        self.assertEqual((r["seg_correct"], r["seg_wrong"]), (0, 1))

    def test_wrong_video_is_wrong_not_missing(self):
        r = one(comp(), [seg(0, 8, "X", 100)])
        self.assertEqual((r["seg_wrong"], r["seg_missing"]), (1, 2))

    def test_top_score_decides_among_overlapping(self):
        # 같은 시간대의 예측 둘: 점수 높은 쪽(틀린 X)이 이긴다
        r = one(comp(), [seg(0, 8, "A", 100, 0.5), seg(0, 8, "X", 5, 0.9)])
        self.assertEqual((r["seg_correct"], r["seg_wrong"]), (0, 1))
        r = one(comp(), [seg(0, 8, "A", 100, 0.9), seg(0, 8, "X", 5, 0.5)])
        self.assertEqual(r["seg_correct"], 1)

    def test_exact_prediction_not_hijacked_by_wider_higher_score(self):
        # 정답 A[0,8]을 정확히 맞힌 예측이 있어도, 옆 구간까지 덮는 [3,24]가 점수가 더 높다고 가로채면 안 된다
        # (겹침: 정확한 예측 8초 > 긴 예측 5초)
        r = one(comp(), [seg(0, 8, "A", 100, 0.5), seg(3, 24, "X", 0, 0.9)])
        self.assertEqual((r["seg_correct"], r["seg_wrong"]), (1, 2))  # A는 맞고, 긴 예측이 B·C 구간을 이겨 틀림 2개
        # 축소판: 정답 [0,4] 하나, 예측 [0,4] 점수 0.5 vs [1.5,20] 점수 0.9
        m = {"query_id": "t", "kind": "compilation", "transform": "compilation", "format": "mixed",
             "source_video_id": "A", "start_sec": 10.0, "end_sec": 14.0, "query_dur_sec": 4.0,
             "segments": [{"q_start": 0, "q_end": 4, "video_id": "A", "start": 10.0}]}
        r = one(m, [seg(0, 4, "A", 10, 0.5), seg(1.5, 20, "X", 0, 0.9)])
        self.assertEqual((r["seg_correct"], r["seg_wrong"]), (1, 0))

    def test_equal_overlap_decided_by_score(self):
        # 둘 다 정답 구간 100%를 덮으면 점수로 가른다 (겹침 차이가 0.1초 미만이어도 같은 것으로 본다)
        m = {"query_id": "t", "kind": "compilation", "transform": "compilation", "format": "mixed",
             "source_video_id": "A", "start_sec": 10.0, "end_sec": 14.0, "query_dur_sec": 4.0,
             "segments": [{"q_start": 0, "q_end": 4, "video_id": "A", "start": 10.0}]}
        self.assertEqual(one(m, [seg(0, 4, "A", 10, 0.9), seg(-1, 5, "X", 0, 0.5)])["seg_correct"], 1)
        self.assertEqual(one(m, [seg(0, 4, "A", 10, 0.5), seg(-1, 5, "X", 0, 0.9)])["seg_wrong"], 1)
        self.assertEqual(one(m, [seg(0, 4, "A", 10, 0.5), seg(0, 3.95, "X", 0, 0.9)])["seg_wrong"], 1)  # 0.05초 차 = 같음

    def test_start_shift_when_pred_segment_begins_later(self):
        # 예측이 질의 1초부터 시작(원본 101초) = 정답 구간 시작(0초)으로 환산하면 100초 -> 정답
        self.assertEqual(one(comp(), [seg(1, 8, "A", 101)])["seg_correct"], 1)

    def test_extra_segment_outside_truth(self):
        # 질의가 24초인데 30~40초에 예측: 정답이 없는 곳 -> extra
        self.assertEqual(one(comp(), [seg(30, 40, "X", 0)])["seg_extra"], 1)
        self.assertEqual(one(comp(), [seg(0, 8, "A", 100)])["seg_extra"], 0)

    def test_empty_prediction(self):
        r = one(comp(), [])
        self.assertEqual((r["seg_correct"], r["seg_missing"]), (0, 3))


class LegacyResultTest(unittest.TestCase):
    def test_single_answer_covers_whole_query(self):
        # 기존 방식(후보 목록) 결과: 1위가 A면 compilation의 A 구간만 맞는다 -> 1/3
        r = one(comp(), candidates=[{"video_id": "A", "start_sec": 100.0, "score": 1.0},
                                    {"video_id": "B", "start_sec": 50.0, "score": 0.9}])
        self.assertEqual((r["seg_total"], r["seg_correct"], r["seg_wrong"]), (3, 1, 2))

    def test_legacy_manifest_segr_equals_r1(self):
        m = {"query_id": "q1", "source_video_id": "A", "start_sec": 10.0, "end_sec": 22.0,
             "query_dur_sec": 12.0, "transform": "none", "format": "long"}
        for cand_start, hit in ((12.0, 1), (14.0, 0)):
            r = one(m, candidates=[{"video_id": "A", "start_sec": cand_start, "score": 1.0}])
            self.assertEqual(r["seg_correct"], hit)
            self.assertEqual(r["rank"] == 1, bool(hit))


class FooledAndSplitTest(unittest.TestCase):
    def swap(self):
        return {"query_id": "s1", "kind": "swap", "transform": "swap", "format": "long", "source_video_id": "A",
                "start_sec": 10.0, "end_sec": 20.0, "query_dur_sec": 10.0,
                "audio_decoy": {"video_id": "B", "start_sec": 5.0}}

    def test_fooled_by_decoy(self):
        r = one(self.swap(), candidates=[{"video_id": "B", "start_sec": 5.0, "score": 2},
                                         {"video_id": "A", "start_sec": 10.0, "score": 1}])
        self.assertTrue(r["fooled"])
        self.assertIsNot(r["rank"], 1)  # 1위가 틀렸으니 R@1 실패
        r = one(self.swap(), candidates=[{"video_id": "A", "start_sec": 10.0, "score": 2}])
        self.assertFalse(r["fooled"])

    def test_non_swap_has_no_fooled(self):
        self.assertIsNone(one(comp(), [seg(0, 8, "A", 100)])["fooled"])

    def test_fooled_rate_and_compilation_excluded_from_r(self):
        man = [self.swap(), comp()]
        res = [{"query_id": "s1", "candidates": [{"video_id": "B", "start_sec": 5.0, "score": 1}]},
               {"query_id": "c1", "segments": [seg(0, 8, "A", 100)]}]
        rows, _ = score_queries(man, res)
        self.assertEqual(summarize_seg(rows)["fooled_rate"], 1.0)
        self.assertEqual(summarize_seg(rows)["n_swap"], 1)
        self.assertEqual(len(single_rows(rows)), 1)
        self.assertEqual(summarize(single_rows(rows))["n"], 1)
        self.assertAlmostEqual(summarize_seg(rows)["SegR"], 1 / 4)  # 구간 4개 중 1개
        self.assertEqual(list(group_by(rows, "kind", summarize_seg)), ["compilation", "swap"])


if __name__ == "__main__":
    unittest.main()
