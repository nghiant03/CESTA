from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import torch

from CESTA.batch import GraphWindowBatch
from CESTA.evaluation.communication import collect_model_communication_config
from CESTA.models.spatial.cesta.model import CESTAClassifier, CommunicationMode


class CESTAControlTest(unittest.TestCase):
    def test_static_topk_selects_highest_probability_available_neighbor(self) -> None:
        model = self._model("static_topk", control_static_topk=1)
        hidden = torch.zeros((1, 1, 3, 2))
        possible = model._possible_message_mask(hidden)

        request = model._rule_request_mask(hidden, possible)

        expected = torch.tensor([[[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]]])
        torch.testing.assert_close(request, expected)

    def test_random_control_is_reproducible_for_a_fixed_seed(self) -> None:
        first = self._model("random", control_request_ratio=0.5, control_seed=17)
        second = self._model("random", control_request_ratio=0.5, control_seed=17)
        hidden = torch.zeros((2, 4, 3, 2))
        possible = first._possible_message_mask(hidden)

        first_request = first._rule_request_mask(hidden, possible)
        second_request = second._rule_request_mask(hidden, possible)

        torch.testing.assert_close(first_request, second_request)
        self.assertGreater(first_request.sum().item(), 0.0)
        self.assertLess(first_request.sum().item(), possible.sum().item())

    def test_receiver_local_threshold_controls_apply_to_all_available_neighbors(self) -> None:
        hidden = torch.zeros((1, 2, 3, 2))
        entropy = torch.tensor([[[0.2, 0.7, 0.8], [0.9, 0.1, 0.4]]])
        margin = torch.tensor([[[0.8, 0.3, 0.2], [0.1, 0.9, 0.6]]])
        change = torch.tensor([[[0.1, 0.8, 0.4], [0.7, 0.2, 0.9]]])
        cases = {
            "entropy": entropy >= 0.5,
            "margin": margin <= 0.5,
            "local_change": change >= 0.5,
        }

        for mode, expected_need in cases.items():
            with self.subTest(mode=mode):
                model = self._model(
                    mode,
                    control_entropy_threshold=0.5,
                    control_margin_threshold=0.5,
                    control_local_change_threshold=0.5,
                )
                possible = model._possible_message_mask(hidden)
                with patch.object(model, "_receiver_control_scores", return_value=(entropy, margin, change)):
                    request = model._rule_request_mask(hidden, possible)
                torch.testing.assert_close(request, expected_need.to(hidden.dtype).unsqueeze(-1) * possible)

    def test_combined_control_uses_normalized_weighted_receiver_score(self) -> None:
        model = self._model(
            "combined",
            control_combined_threshold=0.6,
            control_entropy_weight=2.0,
            control_margin_weight=1.0,
            control_local_change_weight=1.0,
        )
        hidden = torch.zeros((1, 1, 3, 2))
        possible = model._possible_message_mask(hidden)
        entropy = torch.tensor([[[0.8, 0.2, 0.6]]])
        margin = torch.tensor([[[0.2, 0.8, 0.4]]])
        change = torch.tensor([[[0.8, 0.2, 0.6]]])

        with patch.object(model, "_receiver_control_scores", return_value=(entropy, margin, change)):
            request = model._rule_request_mask(hidden, possible)

        expected_need = torch.tensor([[[True, False, True]]])
        torch.testing.assert_close(request, expected_need.to(hidden.dtype).unsqueeze(-1) * possible)

    def test_all_controls_run_end_to_end_and_report_message_counts(self) -> None:
        edge_mask = torch.ones((1, 2, 6), dtype=torch.bool)
        edge_mask[:, :, 1] = False
        batch = GraphWindowBatch(
            x=torch.zeros((1, 2, 6)),
            y=torch.zeros((1, 2, 3), dtype=torch.long),
            node_mask=torch.ones((1, 2, 3), dtype=torch.bool),
            edge_index=torch.tensor(self._edge_index()),
            edge_mask=edge_mask,
        )
        mode_kwargs = {
            "random": {"control_request_ratio": 1.0},
            "static_topk": {"control_static_topk": 1},
            "entropy": {"control_entropy_threshold": 0.0},
            "margin": {"control_margin_threshold": 1.0},
            "local_change": {"control_local_change_threshold": 0.0},
            "combined": {"control_combined_threshold": 0.0},
        }

        for mode, kwargs in mode_kwargs.items():
            with self.subTest(mode=mode):
                model = self._model(mode, **kwargs)
                model.eval()
                logits = model(batch)
                stats = model.last_communication_stats
                self.assertEqual(logits.shape, (1, 2, 3, 2))
                self.assertLessEqual(stats["requested_edge_count"], stats["possible_edge_count"])
                self.assertEqual(len(stats["requested_edge_counts"]), 6)
                self.assertEqual(stats["possible_edge_counts"][1], 0.0)

    def test_control_configuration_is_available_to_evaluation_artifacts(self) -> None:
        model = self._model("entropy", control_entropy_threshold=0.73)

        config = collect_model_communication_config(model)

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config["communication_mode"], "entropy")
        self.assertEqual(config["control_entropy_threshold"], 0.73)
        self.assertEqual(config["control_seed"], 42)

    def test_invalid_control_configuration_fails_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "control_request_ratio"):
            self._model("random", control_request_ratio=1.1)
        with self.assertRaisesRegex(ValueError, "positive control weight"):
            self._model(
                "combined",
                control_entropy_weight=0.0,
                control_margin_weight=0.0,
                control_local_change_weight=0.0,
            )

    @classmethod
    def _model(cls, mode: CommunicationMode, **kwargs: Any) -> CESTAClassifier:
        return CESTAClassifier(
            input_size=6,
            num_nodes=3,
            edge_index=cls._edge_index(),
            edge_prob=[0.9, 0.2, 0.4, 0.8, 0.7, 0.6],
            hidden_size=2,
            num_classes=2,
            dropout=0.0,
            communication_mode=mode,
            **kwargs,
        )

    @staticmethod
    def _edge_index() -> list[list[int]]:
        return [[1, 2, 0, 2, 0, 1], [0, 0, 1, 1, 2, 2]]


if __name__ == "__main__":
    unittest.main()
