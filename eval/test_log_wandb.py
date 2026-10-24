"""log_wandb의 지표 계산이 score.py와 같은지 확인. wandb 없이도 돈다(업로드 함수는 안 부름).
실행: python -m unittest eval.test_log_wandb
"""
import unittest

from eval.log_wandb import build_summary, parse_config, percentile, query_table_rows
from eval.score import group_by, score_queries, summarize
from eval.test_score import MANIFEST, RESULTS
from eval.test_score_seg import comp


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


class SegMetricsTest(unittest.TestCase):
    def test_seg_metrics_only_for_mm_manifests(self):
        rows, _ = score_queries(MANIFEST, RESULTS, 3.0)
        self.assertNotIn("SegR", build_summary(rows, RESULTS))  # 기존 평가셋은 지표 키가 그대로

    def test_seg_and_fooled_keys(self):
        swap = {"query_id": "s1", "kind": "swap", "transform": "swap", "format": "long", "source_video_id": "A",
                "start_sec": 10.0, "end_sec": 20.0, "query_dur_sec": 10.0, "audio_decoy": {"video_id": "B", "start_sec": 0}}
        res = [{"query_id": "s1", "candidates": [{"video_id": "B", "start_sec": 0.0, "score": 1}]},
               {"query_id": "c1", "segments": [{"q_start": 0, "q_end": 8, "video_id": "A", "start": 100.0, "score": 1}]}]
        rows, _ = score_queries([swap, comp()], res)
        s = build_summary(rows, res)
        self.assertAlmostEqual(s["SegR"], 1 / 4)
        self.assertEqual(s["fooled_rate"], 1.0)
        self.assertEqual(s["SegR/kind=compilation"], 1 / 3)
        self.assertEqual(s["n"], 2)
        self.assertEqual(s["R@1"], 0.0)  # R@1은 compilation 빼고 swap 1개만
        t = query_table_rows([swap, comp()], res, rows)  # segments만 낸 결과도 표가 만들어진다
        self.assertEqual(t[1][5], "A")


if __name__ == "__main__":
    unittest.main()
