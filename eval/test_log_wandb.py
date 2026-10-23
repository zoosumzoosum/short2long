"""log_wandb의 지표 계산이 score.py와 같은지 확인. wandb 없이도 돈다(업로드 함수는 안 부름).
실행: python -m unittest eval.test_log_wandb
"""
import unittest

from eval.log_wandb import build_summary, parse_config, percentile, query_table_rows
from eval.score import group_by, score_queries, summarize
from eval.test_score import MANIFEST, RESULTS


class LogWandbTest(unittest.TestCase):
    def setUp(self):
        self.rows, _ = score_queries(MANIFEST, RESULTS, 3.0)
        self.results = [dict(r, latency_ms=ms) for r, ms in zip(RESULTS, (10, 20, 30))]
        self.s = build_summary(self.rows, self.results)

    def test_overall_matches_score(self):
        want = summarize(self.rows)
        self.assertEqual(self.s["R@1"], want["R@1"])
        self.assertEqual(self.s["R@5"], want["R@5"])
        self.assertEqual(self.s["V@1"], want["V@1"])

    def test_flat_group_keys(self):
        tr = group_by(self.rows, "transform")["noise"]
        self.assertEqual(self.s["R@1/noise"], tr["R@1"])
        self.assertEqual(self.s["R@5/format=fan"], group_by(self.rows, "format")["fan"]["R@5"])

    def test_none_errors_skipped(self):
        self.assertNotIn("err_mean/noise", self.s)  # noise는 1위 정답이 없다

    def test_latency(self):
        self.assertEqual((self.s["latency_p50_ms"], self.s["latency_max_ms"]), (20, 30))
        self.assertEqual(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90), 9)

    def test_no_latency_no_keys(self):
        self.assertNotIn("latency_p50_ms", build_summary(self.rows, RESULTS))

    def test_query_table(self):
        t = query_table_rows(MANIFEST, RESULTS, self.rows)
        self.assertEqual(len(t), 4)
        self.assertEqual(t[0][5:9], ["A", 101.0, 0.9, True])
        self.assertEqual(t[3][5], None)  # 결과 없는 질의

    def test_parse_config(self):
        c = parse_config(["a=1", "b=x", "c=[1,2]"], '{"d": true}')
        self.assertEqual(c, {"d": True, "a": 1, "b": "x", "c": [1, 2]})


if __name__ == "__main__":
    unittest.main()
