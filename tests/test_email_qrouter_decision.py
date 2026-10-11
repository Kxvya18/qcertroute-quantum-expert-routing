"""Invariance checks for the decision-aware quantum measurement rule."""
import unittest

import numpy as np

from ba3active.email_qrouter_shots import _set


class DecisionCertificateTests(unittest.TestCase):
    def test_equal_endpoint_sets_are_constant_between_endpoints(self):
        rng = np.random.default_rng(17)
        checked = 0
        for _ in range(500):
            temporal = rng.dirichlet(np.ones(24))
            graph = rng.dirichlet(np.ones(24))
            left, right = sorted(rng.uniform(size=2))
            if _set(temporal, graph, left) != _set(temporal, graph, right):
                continue
            target = _set(temporal, graph, left)
            for alpha in np.linspace(left, right, 31):
                self.assertEqual(_set(temporal, graph, float(alpha)), target)
            checked += 1
        self.assertGreater(checked, 0)

    def test_same_expert_set_requires_no_gate(self):
        temporal = np.linspace(24.0, 1.0, 24)
        graph = temporal**2
        self.assertEqual(_set(temporal, graph, 0.0), _set(temporal, graph, 1.0))


if __name__ == "__main__":
    unittest.main()
