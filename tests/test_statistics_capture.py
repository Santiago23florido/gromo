import unittest
from unittest.mock import patch

import torch

import gromo
from gromo.containers.growing_container import GrowingContainer
from gromo.containers.growing_mlp import GrowingMLP
from gromo.modules.linear_growing_module import (
    LinearGrowingModule,
    LinearMergeGrowingModule,
)
from gromo.utils.utils import global_device


class TestStatisticsCapture(unittest.TestCase):
    def setUp(self) -> None:
        gromo.set_skip_capture_under_functorch(False)
        self.device = global_device()

    def tearDown(self) -> None:
        gromo.set_skip_capture_under_functorch(False)

    def _layer(self, in_features: int = 3, out_features: int = 2):
        return LinearGrowingModule(
            in_features,
            out_features,
            device=self.device,
        )

    @staticmethod
    def _statistic_state(statistic):
        tensor = None if statistic._tensor is None else statistic._tensor.detach().clone()
        return tensor, statistic.samples, statistic.updated

    def test_default_capture_behavior_is_unchanged(self):
        """The default still caches both tensors and retains pre-activity gradients."""
        layer = self._layer()
        layer.store_input = True
        layer.store_pre_activity = True
        x = torch.randn(4, 3, device=self.device)

        output = layer(x)
        output.sum().backward()

        self.assertTrue(torch.equal(layer._input, x))
        self.assertIs(layer._pre_activity, output)
        self.assertIsNotNone(layer._pre_activity.grad)

    def test_is_recording_statistics_matches_all_flag_combinations(self):
        layer = self._layer()

        for store_input in (False, True):
            for store_pre_activity in (False, True):
                with self.subTest(
                    store_input=store_input,
                    store_pre_activity=store_pre_activity,
                ):
                    layer.store_input = store_input
                    layer.store_pre_activity = store_pre_activity
                    self.assertEqual(
                        layer.is_recording_statistics,
                        store_input or store_pre_activity,
                    )

                    old_input = layer._input
                    old_pre_activity = layer._pre_activity
                    layer(torch.randn(2, 3, device=self.device))
                    self.assertEqual(layer._input is not old_input, store_input)
                    self.assertEqual(
                        layer._pre_activity is not old_pre_activity,
                        store_pre_activity,
                    )

    def test_paused_computation_is_non_destructive_nested_and_exception_safe(self):
        previous = self._layer(3, 4)
        layer = self._layer(4, 2)
        layer.previous_module = previous
        previous.next_module = layer
        previous.store_input = True
        previous.store_pre_activity = False
        layer.store_input = False
        layer.store_pre_activity = True

        statistics = (
            previous._tensor_s,
            previous.tensor_m,
            layer._tensor_s,
            layer.tensor_m,
            layer.tensor_m_prev,
            layer.cross_covariance,
            layer.covariance_loss_gradient,
            layer.tensor_s_growth,
        )
        for index, statistic in enumerate(statistics):
            statistic._tensor = torch.full(
                statistic._shape or (1,),
                float(index),
                device=self.device,
            )
            statistic.samples = index + 1
            statistic.updated = False
        states = [self._statistic_state(statistic) for statistic in statistics]

        cached_input = torch.randn(1, device=self.device)
        cached_pre_activity = torch.randn(1, device=self.device)
        previous._input = cached_input
        layer._pre_activity = cached_pre_activity
        flags = (
            previous.store_input,
            previous.store_pre_activity,
            layer.store_input,
            layer.store_pre_activity,
        )

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            with layer.paused_computation():
                self.assertFalse(previous.is_recording_statistics)
                self.assertFalse(layer.is_recording_statistics)
                with layer.paused_computation():
                    self.assertFalse(previous.is_recording_statistics)
                    self.assertFalse(layer.is_recording_statistics)
                    layer(previous(torch.randn(2, 3, device=self.device)))
                self.assertFalse(previous.is_recording_statistics)
                self.assertFalse(layer.is_recording_statistics)
                raise RuntimeError("evaluation failed")

        self.assertEqual(
            (
                previous.store_input,
                previous.store_pre_activity,
                layer.store_input,
                layer.store_pre_activity,
            ),
            flags,
        )
        self.assertIs(previous._input, cached_input)
        self.assertIs(layer._pre_activity, cached_pre_activity)
        for statistic, (tensor, samples, updated) in zip(
            statistics,
            states,
            strict=True,
        ):
            self.assertTrue(torch.equal(statistic._tensor, tensor))
            self.assertEqual(statistic.samples, samples)
            self.assertEqual(statistic.updated, updated)

    def test_pause_restores_merge_counters_exactly(self):
        previous = self._layer(2, 3)
        merge = LinearMergeGrowingModule(
            previous_modules=[previous],
            in_features=3,
            device=self.device,
        )
        following = self._layer(3, 2)
        previous.next_module = merge
        following.previous_module = merge
        merge.set_next_modules([following])
        previous.store_pre_activity = True
        following.store_input = True
        counters = (merge.store_input, merge.store_activity)

        with following.paused_computation():
            self.assertEqual((merge.store_input, merge.store_activity), (0, 0))

        self.assertEqual((merge.store_input, merge.store_activity), counters)

    def test_container_pause_covers_all_modules_and_predecessor_links(self):
        model = GrowingMLP(3, 2, 4, 1, device=self.device)
        external_previous = self._layer(3, 3)
        model.layers[0].previous_module = external_previous
        model.layers[0].store_input = True
        external_previous.store_pre_activity = True

        self.assertTrue(model.is_recording_statistics)
        with model.paused_computation():
            self.assertFalse(model.is_recording_statistics)
            self.assertFalse(external_previous.is_recording_statistics)
        self.assertTrue(model.layers[0].store_input)
        self.assertTrue(external_previous.store_pre_activity)

    def test_container_introspection_uses_any_contained_module(self):
        container = GrowingContainer(3, 2, device=self.device)
        container.layer = self._layer()
        self.assertFalse(container.is_recording_statistics)
        container.layer.store_input = True
        self.assertTrue(container.is_recording_statistics)

    def test_functorch_skip_is_opt_in(self):
        layer = self._layer()
        layer.store_input = True
        layer.store_pre_activity = True
        x = torch.randn(3, device=self.device)

        original_input = torch.randn(1, device=self.device)
        original_pre_activity = torch.randn(1, device=self.device)
        layer._input = original_input
        layer._pre_activity = original_pre_activity
        with self.assertRaisesRegex(RuntimeError, "retain_grad"):
            torch.func.jacrev(layer)(x)
        self.assertIsNot(layer._input, original_input)
        self.assertIsNot(layer._pre_activity, original_pre_activity)

        skipped_input = torch.randn(1, device=self.device)
        skipped_pre_activity = torch.randn(1, device=self.device)
        layer._input = skipped_input
        layer._pre_activity = skipped_pre_activity
        gromo.set_skip_capture_under_functorch(True)
        torch.func.jacrev(layer)(x)
        self.assertIs(layer._input, skipped_input)
        self.assertIs(layer._pre_activity, skipped_pre_activity)

    def test_missing_functorch_detection_api_preserves_capture(self):
        layer = self._layer()
        layer.store_input = True
        layer.store_pre_activity = True
        gromo.set_skip_capture_under_functorch(True)

        with patch.object(torch._C._functorch, "peek_interpreter_stack", None):
            output = layer(torch.randn(3, device=self.device))

        self.assertIsNotNone(layer._input)
        self.assertIs(layer._pre_activity, output)
