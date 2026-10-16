import importlib.util
import inspect
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


PACKAGE_DIR = Path(__file__).resolve().parents[1]
EXPECTED_SIGMAS = (
    1.0, 0.978453, 0.954180, 0.926626, 0.895080,
    0.845148, 0.704534, 0.414568, 0.0,
)


class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        comfy = types.ModuleType("comfy")
        management = types.ModuleType("comfy.model_management")
        management.intermediate_device = lambda: torch.device("cpu")
        management.intermediate_dtype = lambda: torch.float32
        comfy.model_management = management
        spec = importlib.util.spec_from_file_location(
            "qwen21_test_package", PACKAGE_DIR / "__init__.py",
            submodule_search_locations=[str(PACKAGE_DIR)],
        )
        cls.package = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "comfy": comfy,
            "comfy.model_management": management,
            spec.name: cls.package,
        }):
            spec.loader.exec_module(cls.package)
        cls.turbo = cls.package.NODE_CLASS_MAPPINGS["QwenImage21TurboScheduler"]
        cls.dynamic = cls.package.NODE_CLASS_MAPPINGS["QwenImage21DynamicScheduler"]
        cls.empty = cls.package.NODE_CLASS_MAPPINGS["QwenImage21EmptyLatent"]

    def test_registration(self):
        self.assertEqual(self.package.NODE_CLASS_MAPPINGS, {
            "QwenImage21EmptyLatent": self.empty,
            "QwenImage21DynamicScheduler": self.dynamic,
            "QwenImage21TurboScheduler": self.turbo,
        })
        self.assertEqual(self.package.NODE_DISPLAY_NAME_MAPPINGS, {
            "QwenImage21EmptyLatent": "Qwen Image 2.1 Empty Latent",
            "QwenImage21DynamicScheduler": "Qwen Image 2.1 Dynamic Scheduler",
            "QwenImage21TurboScheduler": "Qwen Image 2.1 Turbo Scheduler",
        })

    def test_turbo_interface(self):
        self.assertEqual(self.turbo.CATEGORY, "Qwen Image 2.1/Sampling")
        self.assertEqual(self.turbo.INPUT_TYPES(), {"required": {}})
        self.assertEqual(self.turbo.RETURN_TYPES, ("SIGMAS",))
        self.assertEqual(self.turbo.RETURN_NAMES, ("sigmas",))
        self.assertEqual(list(inspect.signature(self.turbo.get_sigmas).parameters), ["self"])

    def test_turbo_schedule(self):
        output = getattr(self.turbo(), self.turbo.FUNCTION)()
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 1)
        sigmas = output[0]
        self.assertIsInstance(sigmas, torch.Tensor)
        self.assertEqual(sigmas.ndim, 1)
        self.assertEqual(sigmas.dtype, torch.float32)
        self.assertEqual(sigmas.device, torch.device("cpu"))
        self.assertEqual(sigmas.numel(), 9)
        torch.testing.assert_close(
            sigmas, torch.tensor(EXPECTED_SIGMAS, dtype=torch.float32), rtol=0, atol=0,
        )
        self.assertEqual(sigmas[0].item(), 1.0)
        self.assertEqual(sigmas[-1].item(), 0.0)
        self.assertTrue(torch.all(sigmas[:-1] > sigmas[1:]).item())
        self.assertEqual(torch.count_nonzero(sigmas == 0).item(), 1)

    def test_turbo_fresh_tensors(self):
        node = self.turbo()
        first, = node.get_sigmas()
        second, = node.get_sigmas()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        first.zero_()
        third, = self.turbo().get_sigmas()
        torch.testing.assert_close(second, third, rtol=0, atol=0)
        torch.testing.assert_close(
            third, torch.tensor(EXPECTED_SIGMAS, dtype=torch.float32), rtol=0, atol=0,
        )

    def test_turbo_does_not_use_resolution_helpers(self):
        with patch.object(self.dynamic, "_get_seq_len", side_effect=AssertionError), \
             patch.object(self.dynamic, "_compute_mu", side_effect=AssertionError), \
             patch.object(torch, "linspace", side_effect=AssertionError):
            sigmas, = self.turbo().get_sigmas()
        torch.testing.assert_close(
            sigmas, torch.tensor(EXPECTED_SIGMAS, dtype=torch.float32), rtol=0, atol=0,
        )

    def test_dynamic_reference_schedule(self):
        defaults = {
            name: definition[1]["default"]
            for name, definition in self.dynamic.INPUT_TYPES()["required"].items()
        }
        self.assertEqual(defaults, {
            "steps": 40, "base_shift": 0.5, "max_shift": 0.9,
            "base_image_seq_len": 256, "max_image_seq_len": 8192,
            "shift_terminal": 0.02, "time_shift_type": "exponential",
            "stretch_to_terminal": True, "clamp_mu": False,
        })
        for shape in (None, (1, 64, 8, 8), (1, 64, 64, 96)):
            for steps in (1, 8, 40):
                for shift_type in ("exponential", "linear"):
                    for stretch in (False, True):
                        with self.subTest(shape=shape, steps=steps, shift=shift_type, stretch=stretch):
                            latent = None if shape is None else {"samples": torch.empty(shape)}
                            sigmas, mu, seq_len = self.dynamic().get_sigmas(
                                **dict(defaults, steps=steps, time_shift_type=shift_type,
                                       stretch_to_terminal=stretch), latent=latent,
                            )
                            expected_seq_len = 4096 if shape is None else shape[-2] * shape[-1]
                            expected_mu = 0.5 + (expected_seq_len - 256) * 0.4 / (8192 - 256)
                            self.assertEqual(seq_len, expected_seq_len)
                            self.assertAlmostEqual(mu, expected_mu)
                            base = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
                            shift = math.exp(expected_mu) if shift_type == "exponential" else expected_mu
                            expected = shift / (shift + (1.0 / base - 1.0))
                            if stretch and steps > 1:
                                expected = 1.0 - (1.0 - expected) * 0.98 / (1.0 - expected[-1])
                            expected = torch.cat((expected.float(), torch.zeros(1)))
                            torch.testing.assert_close(sigmas, expected, rtol=1e-6, atol=1e-7)

    def test_generic_empty_latent_matches_native_schedule(self):
        defaults = {
            name: definition[1]["default"]
            for name, definition in self.dynamic.INPUT_TYPES()["required"].items()
        }
        for source_shape, target_shape, source_scale in (
            ((1, 4, 128, 128), (1, 64, 64, 64), 8),
            ((2, 4, 128, 192), (2, 64, 64, 96), 8),
            ((1, 4, 65, 67), (1, 64, 32, 34), 8),
            ((1, 4, 66, 70), (1, 64, 33, 35), 8),
            ((1, 64, 128, 128), (1, 64, 64, 64), 8),
            ((1, 4, 64, 64), (1, 64, 64, 64), 16),
        ):
            for masked in (False, True):
                with self.subTest(shape=source_shape, scale=source_scale, masked=masked):
                    samples = torch.zeros(source_shape)
                    latent = {"samples": samples, "downscale_ratio_spacial": source_scale}
                    mask = torch.ones(1, 1, 32, 32)
                    if masked:
                        latent["noise_mask"] = mask
                    expected = self.dynamic().get_sigmas(
                        **defaults, latent={"samples": torch.zeros(target_shape)},
                    )
                    actual = self.dynamic().get_sigmas(**defaults, latent=latent)
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    self.assertEqual(actual[1:], expected[1:])
                    self.assertEqual(actual[2], target_shape[-2] * target_shape[-1])
                    self.assertIs(latent["samples"], samples)
                    self.assertEqual(tuple(samples.shape), source_shape)
                    self.assertEqual(torch.count_nonzero(samples).item(), 0)
                    self.assertEqual(latent["downscale_ratio_spacial"], source_scale)
                    if masked:
                        self.assertIs(latent["noise_mask"], mask)
                        self.assertTrue(torch.all(mask == 1).item())

    def test_generic_latent_requires_empty_samples_and_scale(self):
        samples = torch.zeros(1, 4, 128, 128)
        with self.assertRaisesRegex(ValueError, "require.*downscale_ratio_spacial"):
            self.dynamic._get_seq_len({"samples": samples})
        for scale in (0, -8, float("nan"), float("inf"), "invalid"):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "downscale_ratio_spacial"):
                self.dynamic._get_seq_len({"samples": samples, "downscale_ratio_spacial": scale})
        samples[0, 0, 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Encode the source image"):
            self.dynamic._get_seq_len({"samples": samples, "downscale_ratio_spacial": 8})

    def test_native_encoded_latent_uses_its_existing_grid(self):
        samples = torch.ones(1, 64, 33, 35)
        for scale in (None, 8, 16):
            with self.subTest(scale=scale):
                self.assertEqual(self.dynamic._get_seq_len({
                    "samples": samples, "downscale_ratio_spacial": scale,
                }), 33 * 35)

    def test_existing_interfaces_and_latent(self):
        self.assertEqual(self.dynamic.RETURN_TYPES, ("SIGMAS", "FLOAT", "INT"))
        self.assertEqual(self.dynamic.RETURN_NAMES, ("sigmas", "mu", "seq_len"))
        self.assertEqual(self.dynamic.CATEGORY, "Qwen Image 2.1/Sampling")
        self.assertEqual(set(self.dynamic.INPUT_TYPES()["optional"]), {"latent"})
        self.assertEqual(self.empty.RETURN_TYPES, ("LATENT", "INT", "INT", "INT", "INT"))
        self.assertEqual(self.empty.CATEGORY, "Qwen Image 2.1/Latent")
        latent, width, height, actual_width, actual_height = self.empty().generate(100, 70, 2)
        self.assertEqual((width, height, actual_width, actual_height), (6, 4, 96, 64))
        self.assertEqual(tuple(latent["samples"].shape), (2, 64, 4, 6))
        self.assertEqual(latent["downscale_ratio_spacial"], 16)
        self.assertEqual(torch.count_nonzero(latent["samples"]).item(), 0)


if __name__ == "__main__":
    unittest.main()
