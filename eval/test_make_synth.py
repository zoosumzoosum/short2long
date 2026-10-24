"""make_synth의 순수 함수 테스트 (ffmpeg 없이). 실행: python -m unittest eval.test_make_synth"""
import random
import unittest

from eval import make_synth as ms


class TransformListTest(unittest.TestCase):
    def test_existing_default_unchanged(self):
        # 기존 평가셋(synth_v1 등) 재현: 기본 변형 목록·순서가 바뀌면 배정이 달라진다
        self.assertEqual(ms.TRANSFORMS, ["none", "bgm", "noise", "speed", "crop"])

    def test_visual_names(self):
        self.assertEqual(ms.VIS_TRANSFORMS, ["vcrop", "zoom", "overlay", "mirror", "frame", "combo"])
        self.assertFalse(set(ms.TRANSFORMS) & set(ms.VIS_TRANSFORMS))


class FilterTest(unittest.TestCase):
    def test_same_seed_same_filter(self):
        self.assertEqual(ms.overlay_filter(random.Random(7)), ms.overlay_filter(random.Random(7)))

    def test_overlay_band_range_and_shapes(self):
        for seed in range(50):
            _, p = ms.overlay_filter(random.Random(seed))
            self.assertTrue(0.15 <= p["band_frac"] <= 0.25)
            self.assertIn(p["band_pos"], ("top", "bottom"))
            self.assertTrue(1 <= len(p["shapes"]) <= 3)

    def test_vcrop_output_size(self):
        vf, p = ms.vcrop_filter(random.Random(1))
        self.assertIn("scale=360:640", vf)
        self.assertEqual(p["out_size"], "360x640")


if __name__ == "__main__":
    unittest.main()
