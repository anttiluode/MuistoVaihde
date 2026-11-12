"""V2 temporal event tests. Run: python -m unittest -v"""
import unittest
from temporal_loop import TemporalLoop, LAP


class TestTemporalLoop(unittest.TestCase):
    def setUp(self):
        self.m = TemporalLoop()

    def teach_default(self):
        self.m.teach(("P", "A", "B", "END"))

    def test_untrained_seed_does_not_generate(self):
        self.assertEqual([e.token for e in self.m.replay("P")], ["P"])

    def test_replay_in_order_and_stop(self):
        self.teach_default()
        events = self.m.replay("P")
        self.assertEqual([e.token for e in events], ["P", "A", "B", "END"])
        self.assertEqual([e.tick for e in events], [0, LAP, 2 * LAP, 3 * LAP])
        self.assertEqual([e.origin for e in events], ["world", "self", "self", "self"])
        self.assertEqual(self.m.predict("END"), None)
        self.assertEqual(self.m.ring, [None] * LAP)
        self.assertEqual([self.m.tick() for _ in range(3 * LAP)], [None] * (3 * LAP))

    def test_shuffled_teaching_learns_shuffled_order(self):
        self.m.teach(("P", "B", "A", "END"))
        self.assertEqual([e.token for e in self.m.replay("P")], ["P", "B", "A", "END"])

    def test_cut_feedback_stops_after_seed(self):
        self.teach_default()
        self.assertEqual([e.token for e in self.m.replay("P", feedback=False)], ["P"])

    def test_self_generated_events_never_become_supervision(self):
        self.teach_default()
        before = {a: dict(row) for a, row in self.m.weights.items()}
        self.m.reset_fast()
        self.m.tick(world="P", learn=True)
        for _ in range(3 * LAP + 2):
            self.m.tick(learn=True)
        self.assertEqual(self.m.weights, before)

    def test_world_correct_match_and_world_mismatch(self):
        self.teach_default()
        self.m.reset_fast()
        self.m.tick(world="P", feedback=True)
        for _ in range(LAP - 1):
            self.m.tick()
        right = self.m.tick(world="A", feedback=True)
        self.assertEqual(right.expected, "A")
        self.assertFalse(right.mismatch)
        self.m.reset_fast()
        self.m.tick(world="P", feedback=True)
        for _ in range(LAP - 1):
            self.m.tick()
        wrong = self.m.tick(world="C", feedback=True, learn=False)
        self.assertEqual((wrong.token, wrong.expected, wrong.mismatch), ("C", "A", True))
        self.assertEqual(wrong.origin, "world")
        self.assertIsNone(wrong.sent)  # C untrained, do not continue a false A prediction
        self.assertEqual(self.m.ring, [None] * LAP)

    def test_world_only_temporal_trace_and_gap_expiry(self):
        self.m.tick(world="P", learn=True, feedback=False)
        for _ in range(LAP - 1):
            self.m.tick(feedback=False)
        event = self.m.tick(world="A", learn=True, feedback=False)
        self.assertEqual(event.wrote, ("P", "A"))
        self.assertGreater(self.m.weights["P"]["A"], 0)
        self.m.reset_fast()
        self.m.tick(world="P", learn=True, feedback=False)
        for _ in range(LAP):
            self.m.tick(feedback=False)
        stale = self.m.tick(world="Q", learn=True, feedback=False)
        self.assertIsNone(stale.wrote)
        self.assertEqual(self.m.weights["P"]["Q"], 0)

    def test_same_cue_two_futures_abstains_instead_of_hiding_conflict(self):
        self.m.teach(("P", "A", "B", "END"))
        self.m.teach(("Q", "A", "C", "END"))
        self.assertIsNone(self.m.predict("A"))
        self.assertEqual([e.token for e in self.m.replay("P")], ["P", "A"])
        self.assertEqual([e.token for e in self.m.replay("Q")], ["Q", "A"])

    def test_explicit_end_not_trained_as_start(self):
        self.teach_default()
        self.assertTrue(all(v == 0 for v in self.m.weights["END"].values()))
        self.assertEqual([e.token for e in self.m.replay("END")], ["END"])


if __name__ == "__main__":
    unittest.main()
