"""score.py 손계산 테스트. 실행: python -m unittest eval.test_score  (pytest로도 돈다)"""
import unittest

from eval.score import score_queries, summarize, group_by


def m(qid, vid, start, tr, fmt):
    return {"query_id": qid, "source_video_id": vid, "start_sec": start,
            "end_sec": start + 10, "transform": tr, "format": fmt}


def res(qid, *cands):
    return {"query_id": qid, "candidates": [
        {"video_id": v, "start_sec": s, "score": sc} for v, s, sc in cands]}


MANIFEST = [
    m("q1", "A", 100.0, "none", "long"),   # 1위 정답, 오차 1.0
    m("q2", "B", 50.0, "none", "fan"),     # 1위는 다른 영상, 3위 정답(오차 3.0 = 경계, 정답)
    m("q3", "C", 20.0, "noise", "long"),   # 영상은 맞지만 시각이 4초 틀림 → 오답
    m("q4", "D", 0.0, "noise", "fan"),     # 결과 없음 → 실패
]
RESULTS = [
    res("q1", ("A", 101.0, 0.9), ("X", 0.0, 0.1)),
    # 후보를 일부러 score 순이 아닌 순서로 적어 정렬을 확인
    res("q2", ("B", 53.0, 0.5), ("X", 1.0, 0.9), ("Y", 2.0, 0.7)),
    res("q3", ("C", 24.0, 0.8)),
]


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.rows, self.unknown = score_queries(MANIFEST, RESULTS, tol=3.0)

    def test_ranks(self):
        self.assertEqual([r["rank"] for r in self.rows], [1, 3, None, None])

    def test_overall(self):
        s = summarize(self.rows)
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["R@1"], 1 / 4)
        self.assertAlmostEqual(s["R@5"], 2 / 4)
        # 오차는 1위 정답(q1)만: 1.0
        self.assertAlmostEqual(s["err_mean"], 1.0)

    def test_by_transform_and_format(self):
        t = group_by(self.rows, "transform")
        self.assertAlmostEqual(t["none"]["R@1"], 1 / 2)
        self.assertAlmostEqual(t["none"]["R@5"], 1.0)
        self.assertAlmostEqual(t["noise"]["R@5"], 0.0)
        self.assertIsNone(t["noise"]["err_mean"])
        f = group_by(self.rows, "format")
        self.assertAlmostEqual(f["long"]["R@5"], 1 / 2)
        self.assertAlmostEqual(f["fan"]["R@5"], 1 / 2)

    def test_tolerance_argument(self):
        rows, _ = score_queries(MANIFEST, RESULTS, tol=5.0)  # q3 오차 4초가 정답이 됨
        self.assertEqual(rows[2]["rank"], 1)
        rows, _ = score_queries(MANIFEST, RESULTS, tol=0.5)  # q1 오차 1초가 오답이 됨
        self.assertIsNone(rows[0]["rank"])

    def test_missing_and_unknown(self):
        self.assertTrue(self.rows[3]["missing"])
        _, unknown = score_queries(MANIFEST, RESULTS + [res("zz", ("A", 0, 1))])
        self.assertEqual(unknown, ["zz"])

    def test_duplicate_result_rejected(self):
        with self.assertRaises(ValueError):
            score_queries(MANIFEST, RESULTS + [res("q1", ("A", 100.0, 1))])

    def test_median(self):
        man = [m(f"q{i}", "A", 0.0, "none", "long") for i in range(3)]
        rs = [res(f"q{i}", ("A", e, 1)) for i, e in enumerate([0.0, 1.0, 3.0])]
        rows, _ = score_queries(man, rs)
        s = summarize(rows)
        self.assertAlmostEqual(s["err_mean"], 4 / 3)
        self.assertAlmostEqual(s["err_median"], 1.0)


class DedupAndVideoLevelTest(unittest.TestCase):
    def test_same_video_candidates_collapse(self):
        # X의 구간 후보 6개가 위를 채워도 영상으로는 1칸이므로 B는 2위 (중복 제거 안 하면 7위 → R@5 실패)
        cands = [("X", float(i), 0.9 - i * 0.01) for i in range(6)] + [("B", 10.0, 0.5)]
        rows, _ = score_queries([m("q1", "B", 10.0, "none", "long")], [res("q1", *cands)])
        self.assertEqual(rows[0]["rank"], 2)
        self.assertAlmostEqual(summarize(rows)["R@5"], 1.0)

    def test_best_candidate_per_video_decides(self):
        # 정답 영상의 최고 점수 후보가 시각이 틀리면 오답 (낮은 점수 후보가 맞아도 안 봄)
        cands = [("B", 90.0, 0.9), ("B", 10.0, 0.8)]
        rows, _ = score_queries([m("q1", "B", 10.0, "none", "long")], [res("q1", *cands)])
        self.assertIsNone(rows[0]["rank"])
        self.assertEqual(rows[0]["vrank"], 1)

    def test_video_level_and_time_error_median(self):
        man = [m(f"q{i}", "A", 0.0, "none", "long") for i in range(1, 5)]
        rs = [res("q1", ("A", 1.0, 1)),                     # 영상+시각 정답, 오차 1
              res("q2", ("A", 50.0, 1)),                    # 영상만 정답, 오차 50 (tol 초과)
              res("q3", ("Z", 0.0, 1), ("A", 9.0, 0.5)),    # 영상은 2위
              res("q4", ("Z", 0.0, 1))]                     # 영상 못 찾음
        s = summarize(score_queries(man, rs, tol=3.0)[0])
        self.assertAlmostEqual(s["R@1"], 1 / 4)
        self.assertAlmostEqual(s["V@1"], 2 / 4)
        self.assertAlmostEqual(s["V@5"], 3 / 4)
        self.assertAlmostEqual(s["verr_median"], 25.5)  # 1위 영상이 맞은 q1(1), q2(50)의 중앙값


if __name__ == "__main__":
    unittest.main()
