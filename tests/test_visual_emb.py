"""visual_emb 테스트 (GPU·모델 없이 돈다). 실행 (저장소 루트):
    python -m unittest tests.test_visual_emb -v
PIL이 없으면 view 만들기 테스트는 건너뛰고, DATABASE_URL이 없으면 DB 테스트는 건너뛴다.
DB 테스트는 임시 이름의 표를 만들고 끝에 지운다 (진짜 표 vis_*와 이름이 겹치지 않음).
"""
import os
import unittest

import numpy as np
import psycopg

from app import visual_emb as ve


def nn_rows(pairs):
    """[(질의 벡터 번호, src_id, 원본 시각, 유사도)] -> vote 입력 (view 열은 0)."""
    return np.array([(q, s, t, sim, 0) for q, s, t, sim in pairs], dtype=np.float64)


class TestBoxes(unittest.TestCase):
    def test_landscape(self):
        b = ve.view_boxes(360, 640)
        self.assertEqual(b["c"], (0, 360, 140, 500))   # 가운데 360x360
        self.assertEqual(b["a"], (0, 360, 0, 360))     # 왼쪽 끝
        self.assertEqual(b["z"], (0, 360, 280, 640))   # 오른쪽 끝
        self.assertEqual(b["f"], (0, 360, 0, 640))

    def test_portrait(self):
        b = ve.view_boxes(256, 144)
        self.assertEqual(b["c"], (56, 200, 0, 144))
        self.assertEqual(b["a"], (0, 144, 0, 144))     # 위
        self.assertEqual(b["z"], (112, 256, 0, 144))   # 아래

    def test_square_dedup(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("PIL 없음")
        img = np.zeros((100, 100, 3), np.uint8)
        kinds = [k for k, _ in ve.make_views(img, "cazf")]
        self.assertEqual(kinds, ["c"])  # c=a=z=f 모두 같은 상자 -> 하나만

    def test_view_shape_and_flip(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("PIL 없음")
        img = np.zeros((360, 640, 3), np.uint8)
        img[:, :320] = 255  # 왼쪽 절반 흰색
        v = dict(ve.make_views(img, "f"))["f"]
        self.assertEqual(v.shape, (ve.SIZE, ve.SIZE, 3))
        vf = dict(ve.make_views(img, "f", flip=True))["f"]
        self.assertGreater(v[:, :50].mean(), 200)   # 원본은 왼쪽이 밝고
        self.assertGreater(vf[:, -50:].mean(), 200)  # 반전하면 오른쪽이 밝다


class TestSqlFile(unittest.TestCase):
    def test_init_sql_matches_ddl(self):
        # db/init/07의 표 정의가 코드의 ddl()과 어긋나지 않는지
        with open("db/init/07_visual_emb.sql", encoding="utf-8") as f:
            sql = f.read()
        self.assertIn(ve.ddl("dinov2b", 768).strip(), sql)


class TestTrim(unittest.TestCase):
    def test_black_bars_removed(self):
        rng = np.random.default_rng(0)
        img = np.zeros((640, 360, 3), np.uint8)
        img[200:440, 20:340] = rng.integers(0, 255, size=(240, 320, 3))  # 가운데에만 그림
        out = ve.trim_border(img)
        self.assertEqual(out.shape[:2], (240, 320))

    def test_no_bars_unchanged(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(100, 160, 3)).astype(np.uint8)
        self.assertEqual(ve.trim_border(img).shape, img.shape)

    def test_all_dark_frame_kept(self):
        # 어두운 장면 전체가 '띠'로 보이면 자르지 않는다 (남는 넓이가 너무 작음)
        img = np.zeros((100, 160, 3), np.uint8)
        self.assertEqual(ve.trim_border(img).shape, img.shape)


class TestVote(unittest.TestCase):
    QP = ve.QueryParams(sim_floor=0.3)

    def test_consistent_offset_wins(self):
        # 질의 프레임 0..9 (시각 0..9). 영상 1은 오프셋 +100에서 줄줄이 닮음, 영상 2는 제각각 이웃
        rows = [(i, 1, 100 + i, 0.8) for i in range(10)]
        rng = np.random.default_rng(0)
        rows += [(i, 2, float(rng.integers(0, 500)), 0.9) for i in range(10)]
        res = ve.vote(nn_rows(rows), list(range(10)), list(map(float, range(10))), self.QP)
        self.assertGreater(res[1][0], res[2][0])
        self.assertAlmostEqual(res[1][1], 100.0, delta=0.5)
        self.assertEqual(res[1][2], 10)

    def test_one_frame_counts_once_per_window(self):
        # 같은 프레임이 같은 오프셋에 view·반전으로 5번 표를 줘도 한 번(가장 큰 값)만
        rows = [(0, 1, 50.0, 0.5 + 0.05 * k) for k in range(5)]
        res = ve.vote(nn_rows(rows), [0] * 5, [0.0], self.QP)
        self.assertAlmostEqual(res[1][0], 0.7 - 0.3)

    def test_floor_removes_weak(self):
        res = ve.vote(nn_rows([(0, 1, 5.0, 0.25)]), [0], [0.0], self.QP)
        self.assertEqual(res, {})

    def test_drift_within_window(self):
        # 배속 1.1처럼 오프셋이 100.0 -> 101.9로 천천히 어긋나도 ±2초 창 안이면 한 덩어리
        rows = [(i, 1, i + 100 + 0.2 * i, 0.8) for i in range(10)]
        res = ve.vote(nn_rows(rows), list(range(10)), list(map(float, range(10))), self.QP)
        self.assertEqual(res[1][2], 10)

    def test_empty(self):
        self.assertEqual(ve.vote(np.zeros((0, 5)), [], [], self.QP), {})

    def test_index_views_filter(self):
        rows = nn_rows([(0, 1, 5.0, 0.9)])
        rows[:, 4] = ve.VIEW_IDS["f"]
        qp = ve.QueryParams(index_views="c")
        self.assertEqual(ve.vote(rows, [0], [0.0], qp), {})


def fake_scene(seed, h=144, w=256):
    """장면 비슷한 가짜 프레임: 6x10 칸마다 무작위 색을 깔고 부드럽게 늘린다. 같은 시드면 같은 그림."""
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 256, size=(6, 10, 3)).astype(np.float32)
    big = np.kron(grid, np.ones((h // 6, w // 10, 1), np.float32))
    return big.astype(np.uint8)


class FakeEmbedder:
    """모델 대신 쓰는 가짜: 8x8 칸 평균 색(192차원)에서 평균을 빼고 길이 1로 맞춘다.
    밝기를 더하거나 곱해도(클리핑 전까지) 같은 벡터가 나오고, 좌우반전하면 다른 벡터가 나온다 -> 질의 쪽 반전 경로를 시험할 수 있다."""
    dim = 192

    def embed(self, imgs):
        n, s = len(imgs), ve.SIZE
        x = imgs.astype(np.float32).reshape(n, 8, s // 8, 8, s // 8, 3).mean(axis=(2, 4)).reshape(n, -1)
        x -= x.mean(axis=1, keepdims=True)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def as_frames(arr):
    return [(float(i), f) for i, f in enumerate(arr)]


@unittest.skipUnless(os.environ.get("DATABASE_URL"), "DATABASE_URL 없음")
class TestDb(unittest.TestCase):
    """가짜 영상 3편(프레임 20장씩)을 임시 스키마에 색인하고, 자르거나 뒤집거나 밝기를 바꾼 질의가 원본·시각을 찾는지 본다.
    모든 작업이 하나의 트랜잭션 안(임시 스키마 포함)이고 끝에 롤백하므로 DB에 흔적이 남지 않는다.
    실제 표(vis_*_dinov2b)와 이름이 같아도 임시 스키마가 먼저 보이므로 건드리지 않는다."""
    MODEL = "dinov2b"

    def setUp(self):
        self.conn = psycopg.connect(os.environ["DATABASE_URL"])  # autocommit 아님: 롤백으로 전부 되돌린다
        self.addCleanup(self.conn.close)
        self.addCleanup(self.conn.rollback)
        self.conn.execute("CREATE SCHEMA t_vis_test")
        self.conn.execute("SET LOCAL search_path = t_vis_test, public")
        self.emb = FakeEmbedder()
        self.conn.execute(ve.ddl(self.MODEL, self.emb.dim))
        self.videos = {name: np.stack([fake_scene(seed * 1000 + i) for i in range(20)])
                       for name, seed in (("VA", 1), ("VB", 2), ("VC", 3))}
        for name, frames in self.videos.items():
            self.assertTrue(self.index(name, frames))

    def index(self, name, frames, params=None):
        times, views, imgs = [], [], []
        for t, img in enumerate(frames):
            for k, v in ve.make_views(img, "cazf"):
                times.append(float(t))
                views.append(ve.VIEW_IDS[k])
                imgs.append(v)
        embs = self.emb.embed(np.stack(imgs))
        return ve.index_rows(self.conn, self.MODEL, "T_" + name, float(len(frames)), times, views, embs,
                             params or ve.index_params(self.MODEL))

    def query(self, arr, **kw):
        qp = ve.QueryParams(**kw)
        cands, _ = ve.query_frames(self.conn, self.MODEL, self.emb, as_frames(arr), qp)
        return cands

    def test_exact_clip_found_with_start_time(self):
        best = self.query(self.videos["VB"][6:16])[0]  # 프레임 6~15 -> 시작 6초
        self.assertEqual(best["video_id"], "T_VB")
        self.assertAlmostEqual(best["start_sec"], 6.0, delta=1.0)

    def test_flipped_and_brightened_clip_found(self):
        clip = np.clip(self.videos["VC"][3:13][:, :, ::-1].astype(np.float32) * 0.8 + 25, 0, 255).astype(np.uint8)
        best = self.query(clip, flip=True)[0]  # 질의 쪽 반전을 켜야 거울상을 되돌려 찾는다
        self.assertEqual(best["video_id"], "T_VC")
        self.assertAlmostEqual(best["start_sec"], 3.0, delta=1.0)

    def test_flipped_clip_needs_flip_option(self):
        # 반전 옵션이 실제로 일하는지: 끄면 점수가 눈에 띄게 낮다
        clip = self.videos["VC"][3:13][:, :, ::-1]
        with_flip = self.query(clip, flip=True)[0]["score"]
        without = [c for c in self.query(clip, flip=False) if c["video_id"] == "T_VC"]
        self.assertTrue(not without or without[0]["score"] < with_flip / 2)

    def test_horizontal_crop_found(self):
        clip = self.videos["VA"][2:12][:, :, 20:196]  # 가로 프레임의 왼쪽으로 치우친 부분 (144x176)
        best = self.query(clip)[0]
        self.assertEqual(best["video_id"], "T_VA")
        self.assertAlmostEqual(best["start_sec"], 2.0, delta=1.0)

    def test_unknown_clip_has_low_score(self):
        known = self.query(self.videos["VA"][0:10])[0]["score"]
        other = self.query(np.stack([fake_scene(99000 + i) for i in range(10)]))
        self.assertTrue(all(c["score"] < known / 3 for c in other))

    def test_duplicate_index_is_skipped(self):
        self.assertFalse(self.index("VA", self.videos["VA"]))

    def test_query_with_different_fps_is_rejected(self):
        with self.assertRaises(ValueError):
            ve.query_frames(self.conn, self.MODEL, self.emb, as_frames(self.videos["VA"][:5]), fps=2.0)

    def test_query_with_changed_model_revision_is_rejected(self):
        src, _ = ve.tables(self.MODEL)
        self.conn.execute(f"UPDATE {src} SET params = jsonb_set(params, '{{revision}}', '\"other\"')")
        with self.assertRaises(ValueError):
            self.query(self.videos["VA"][:5])

    def test_index_without_params_is_rejected(self):
        src, _ = ve.tables(self.MODEL)
        self.conn.execute(f"UPDATE {src} SET params = NULL")
        with self.assertRaises(ValueError):
            self.query(self.videos["VA"][:5])


class TestTableWhitelist(unittest.TestCase):
    def test_unknown_model_rejected(self):
        for bad in ("x; DROP TABLE videos", "dinov2b ", "", "../x"):
            with self.assertRaises(ValueError):
                ve.tables(bad)

    def test_known_models_ok(self):
        for m in ve.MODELS:
            self.assertEqual(ve.tables(m), (f"vis_src_{m}", f"vis_frame_{m}"))

    def test_models_have_pinned_revision(self):
        for spec in ve.MODELS.values():
            self.assertEqual(len(spec.revision), 40)


if __name__ == "__main__":
    unittest.main()
