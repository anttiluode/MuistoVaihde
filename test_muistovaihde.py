"""Run: python -m unittest -v"""
import unittest
from muistovaihde import A_SIZE, B_SIZE, C_SIZE, Switchyard, crt


class TestSwitchyard(unittest.TestCase):
    def test_all_35_phase_pairs_and_crt(self):
        for oa in range(A_SIZE):
            for ob in range(B_SIZE):
                m = Switchyard()
                m.reset_fast(phase_a=oa, phase_b=ob)
                observed = [(e.a, e.b, e.c) for e in m.run(C_SIZE)]
                self.assertEqual(len(set(observed)), C_SIZE)
                self.assertEqual({(a, b) for a, b, _ in observed},
                                 {(a, b) for a in range(A_SIZE) for b in range(B_SIZE)})
                self.assertTrue(all(c == crt(a, b) for a, b, c in observed))

    def test_cold_start_and_coincidence_only(self):
        m = Switchyard()
        self.assertEqual(m.recall({1})[0], set())
        m.teach({1}, set(), laps=10)
        self.assertFalse(m.active_pairs())
        m.teach(set(), {2, 5}, laps=10)
        self.assertFalse(m.active_pairs())

    def test_selective_learning_and_recall(self):
        m = Switchyard()
        m.teach({1}, {2, 5})
        m.teach({3}, {1, 4})
        self.assertEqual(m.active_pairs(), {(1, 2), (1, 5), (3, 1), (3, 4)})
        self.assertEqual(m.recall({1})[0], {2, 5})
        self.assertEqual(m.recall({3})[0], {1, 4})
        self.assertEqual(m.recall({0})[0], set())

    def test_weights_are_labelled_not_absolute_time(self):
        m = Switchyard()
        m.teach({1}, {2, 5})
        for pa in range(A_SIZE):
            for pb in range(B_SIZE):
                self.assertEqual(m.recall({1}, phase_a=pa, phase_b=pb)[0], {2, 5})
                self.assertEqual(m.bind({1}, {2, 5}, phase_a=pa, phase_b=pb)[0],
                                 {(1, 2), (1, 5)})

    def test_ring_c_is_conjunctive_record(self):
        m = Switchyard()
        m.teach({1}, {2, 5})
        self.assertEqual(m.bind({1}, {2, 5})[0], {(1, 2), (1, 5)})
        self.assertEqual(m.bind({1}, {0, 1})[0], set())
        self.assertEqual(m.bind({0}, {2, 5})[0], set())

    def test_forgetting_occurs_without_any_traffic(self):
        m = Switchyard(decay=.001)
        m.teach({1}, {2}, laps=10)
        self.assertEqual(m.recall({1})[0], {2})
        m.reset_fast()
        m.run(2000, "idle")
        self.assertEqual(m.recall({1})[0], set())

    def test_small_capacity_and_same_cue_interference(self):
        m = Switchyard()
        patterns = [({i}, {i % B_SIZE, (i + 2) % B_SIZE}) for i in range(A_SIZE)]
        for a, b in patterns:
            m.teach(a, b)
        for a, b in patterns:
            self.assertEqual(m.recall(a)[0], b)
        q = Switchyard()
        q.teach({1}, {2})
        q.teach({1}, {4})
        self.assertEqual(q.recall({1})[0], {2, 4})  # cannot recall which episode


if __name__ == '__main__':
    unittest.main()
