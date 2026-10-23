"""audio_fp 테스트. 실행 (저장소 루트, DATABASE_URL이 테스트 DB를 가리킬 때):
    python -m unittest tests.test_audio_fp -v
DB가 필요한 테스트는 트랜잭션 안에서만 넣고 끝에 롤백하므로 DB에 흔적이 남지 않는다.
DATABASE_URL이 없으면 DB 테스트는 건너뛴다.
"""
import os
import unittest

import numpy as np
import psycopg
from scipy.signal import resample_poly

from app import audio_fp

SR = audio_fp.DEFAULT.sr


def fake_music(seconds, seed):
    """음악 비슷한 합성 신호: 0.25초마다 음 3개가 바뀐다. 같은 시드면 같은 신호."""
    rng = np.random.default_rng(seed)
    n_notes = int(seconds / 0.25)
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    seg = int(0.25 * SR)
    tt = np.arange(seg) / SR
    for i in range(n_notes):
        for f in rng.uniform(200, 3500, size=3):
            out[i * seg:(i + 1) * seg] += (np.sin(2 * np.pi * f * tt) * rng.uniform(0.2, 1.0)).astype(np.float32)
    return out / 4


class TestHashes(unittest.TestCase):
    def test_deterministic_and_in_range(self):
        x = fake_music(10, seed=1)
        h1, t1 = audio_fp.fingerprint(x)
        h2, t2 = audio_fp.fingerprint(x)
        self.assertGreater(len(h1), 100)
        self.assertTrue(np.array_equal(h1, h2) and np.array_equal(t1, t2))
        self.assertTrue((h1 >= 0).all() and (h1 < 2 ** 22).all())  # 22비트에 들어감
        self.assertTrue((h1 & 63).min() >= 1)  # 시간차는 1프레임 이상

    def test_peak_cap_per_second(self):
        S = audio_fp.spectrogram(fake_music(10, seed=2))
        t, f = audio_fp.find_peaks(S)
        per_sec = np.bincount(t // (SR // audio_fp.DEFAULT.hop))
        self.assertLessEqual(per_sec.max(), audio_fp.DEFAULT.peaks_per_sec)

    def test_silence_and_short_input(self):
        h, t = audio_fp.fingerprint(np.zeros(SR * 5, dtype=np.float32))
        self.assertEqual(len(h), 0)  # 무음은 해시가 없다 (무음끼리 서로 일치하면 안 됨)
        h, t = audio_fp.fingerprint(np.zeros(100, dtype=np.float32))
        self.assertEqual(len(h), 0)  # 창보다 짧은 입력도 에러 없이


@unittest.skipUnless(os.environ.get("DATABASE_URL"), "DATABASE_URL 없음")
class TestDb(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg.connect(os.environ["DATABASE_URL"])
        self.conn.execute("SET LOCAL synchronous_commit = off")
        self.x = fake_music(60, seed=10)
        audio_fp.index_video(self.conn, "TEST_SELF", self.x)
        audio_fp.index_video(self.conn, "TEST_OTHER", fake_music(60, seed=11))

    def tearDown(self):
        self.conn.rollback()  # 넣은 것을 전부 취소
        self.conn.close()

    def top(self, q, **kw):
        return audio_fp.query_audio(self.conn, q, **kw)[0]

    def test_finds_itself_at_right_time(self):
        start = 20.37  # 프레임 격자(32ms)에 안 맞는 시각
        q = self.x[int(start * SR):int((start + 12) * SR)]
        best = self.top(q)
        self.assertEqual(best["video_id"], "TEST_SELF")
        self.assertAlmostEqual(best["start_sec"], start, delta=0.1)

    def test_survives_noise(self):
        rng = np.random.default_rng(0)
        q = self.x[int(5 * SR):int(17 * SR)]
        q = q + rng.normal(0, q.std() / 3, size=len(q)).astype(np.float32)  # SNR 약 10dB
        best = self.top(q)
        self.assertEqual(best["video_id"], "TEST_SELF")
        self.assertAlmostEqual(best["start_sec"], 5.0, delta=0.1)

    def test_unrelated_query_scores_low(self):
        q = fake_music(12, seed=99)
        own = self.top(self.x[10 * SR:22 * SR])["score"]
        res = audio_fp.query_audio(self.conn, q)
        other = res[0]["score"] if res else 0
        self.assertLess(other, own / 5)

    def test_speed_needs_resampling(self):
        # 1.08배속(음정도 올라감) 질의: 배속을 되돌려 보는 질의만 찾는다
        q = resample_poly(self.x[10 * SR:25 * SR], 100, 108).astype(np.float32)
        plain = audio_fp.query_audio(self.conn, q, speeds=(1.0,))
        fixed = audio_fp.query_audio(self.conn, q, speeds=(1.0, 1.08))
        self.assertEqual(fixed[0]["video_id"], "TEST_SELF")
        self.assertAlmostEqual(fixed[0]["start_sec"], 10.0, delta=0.15)
        self.assertGreater(fixed[0]["score"], 5 * (plain[0]["score"] if plain else 1))


@unittest.skipUnless(os.environ.get("DATABASE_URL"), "DATABASE_URL 없음")
class TestIndexCommit(unittest.TestCase):
    """색인 도중(연결 A가 아직 열려 있을 때) 다른 연결 B가 막히지 않고 넣어 둔 영상을 본다."""

    def setUp(self):
        self.a = audio_fp.connect(os.environ["DATABASE_URL"])
        self.b = psycopg.connect(os.environ["DATABASE_URL"])

    def tearDown(self):
        self.a.execute("DELETE FROM audio_fp WHERE src_id IN (SELECT src_id FROM audio_fp_src WHERE video_id LIKE 'TEST_LOCK%')")
        self.a.execute("DELETE FROM audio_fp_src WHERE video_id LIKE 'TEST_LOCK%'")
        self.a.close()
        self.b.close()

    def test_other_connection_not_blocked_and_sees_committed_video(self):
        x = fake_music(10, seed=5)
        self.assertTrue(audio_fp.index_video(self.a, "TEST_LOCK_1", x))
        self.b.execute("SET lock_timeout = '2s'")  # 막히면 2초 뒤 에러 -> 테스트 실패
        n = self.b.execute("SELECT count(*) FROM audio_fp_src WHERE video_id = 'TEST_LOCK_1'").fetchone()[0]
        self.assertEqual(n, 1)  # A가 연결을 닫지 않았는데도 커밋되어 보임
        self.b.execute("SELECT count(*) FROM audio_fp")  # 표 전체 조회도 막히지 않음
        self.b.rollback()

    def test_already_indexed_video_is_skipped(self):
        x = fake_music(10, seed=6)
        self.assertTrue(audio_fp.index_video(self.a, "TEST_LOCK_2", x))
        self.assertFalse(audio_fp.index_video(self.a, "TEST_LOCK_2", x))  # 두 번째는 건너뜀 (행이 두 배가 되지 않음)
        n = self.b.execute("SELECT count(*) FROM audio_fp_src WHERE video_id = 'TEST_LOCK_2'").fetchone()[0]
        self.assertEqual(n, 1)


if __name__ == "__main__":
    unittest.main()
