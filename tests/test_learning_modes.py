import tempfile
import unittest
from pathlib import Path

from praxis.learning import agents, db
from praxis.learning.models import FlashcardDraft, Plan, PlanDay


def sample_plan() -> Plan:
    return Plan(
        reasoning="Dependencies first, then application.",
        self_check="Two sequential days with synthesis last.",
        reasoning_types=["dependency-analysis"],
        caveats=[],
        goal="Learn testing",
        summary="Learn foundations and apply them.",
        total_days=2,
        days=[
            PlanDay(
                day_num=1,
                topic="Foundations",
                objective="Explain isolation and deterministic tests.",
                concepts=["test isolation", "fixtures"],
                activities=["Write one isolated test."],
                suggested_resources=[],
            ),
            PlanDay(
                day_num=2,
                topic="Application",
                objective="Apply fixtures in an integration test.",
                concepts=["integration testing"],
                activities=["Build a small integration test."],
                suggested_resources=[],
            ),
        ],
    )


class LearningModesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.original_db_path = db.DB_PATH
        db.DB_PATH = str(Path(self.tmp.name) / "learning.db")
        db.init()

    def tearDown(self):
        db.DB_PATH = self.original_db_path
        self.tmp.cleanup()

    def create_goal(self) -> int:
        goal_id = db.create_goal("Learn software testing", "beginner", 2)
        db.save_plan(goal_id, sample_plan())
        return goal_id

    def test_profiles_isolate_goal_lists(self):
        second_user = db.create_user("Second learner")
        first_goal = db.create_goal("Learn first subject", "beginner", 3)
        second_goal = db.create_goal("Learn second subject", "advanced", 4, second_user)

        self.assertEqual([g.id for g in db.list_goals(1)], [first_goal])
        self.assertEqual([g.id for g in db.list_goals(second_user)], [second_goal])

    def test_plan_builds_concept_graph(self):
        goal_id = self.create_goal()
        graph = db.concept_graph(goal_id)

        self.assertEqual(len(graph["nodes"]), 3)
        self.assertTrue(
            any(
                e["source"] == "fixtures" and e["target"] == "integration testing"
                for e in graph["edges"]
            )
        )

    def test_flashcard_review_reschedules_and_updates_mastery(self):
        goal_id = self.create_goal()
        db.save_flashcards(
            goal_id,
            [
                FlashcardDraft(
                    concept="fixtures",
                    front="Why should a test fixture establish only the state required by the test?",
                    back="A focused fixture limits hidden coupling, keeps failures local, and makes the test setup easier to verify and reuse.",
                    card_type="mechanism",
                    difficulty="application",
                ),
            ],
        )
        cards = db.due_flashcards(goal_id)
        self.assertEqual(len(cards), 1)

        reviewed = db.review_flashcard(cards[0].id, 4)
        mastery = {m.concept: m for m in db.get_mastery(goal_id)}

        self.assertEqual(reviewed.reps, 1)
        self.assertGreater(reviewed.stability, 1.0)
        self.assertEqual(mastery[cards[0].concept].samples, 1)
        self.assertGreater(mastery[cards[0].concept].score, 0)

    def test_note_persistence_adds_new_concepts(self):
        goal_id = self.create_goal()
        note_id = db.save_note(
            goal_id,
            "notes.md",
            "# Property-based testing",
            ["Property-based testing"],
        )

        notes = db.list_notes(goal_id)
        mastery = {m.concept for m in db.get_mastery(goal_id)}
        self.assertEqual(notes[0].id, note_id)
        self.assertIn("Property-based testing", mastery)

    def test_adaptive_questions_get_harder_with_mastery(self):
        questions = agents.build_adaptive_questions(
            ["weak", "medium", "strong"],
            {"weak": 0.1, "medium": 0.5, "strong": 0.9},
            {"medium": ["weak"]},
            3,
        )

        self.assertEqual(questions[0].difficulty, "foundation")
        self.assertIn("core mechanism", questions[0].prompt)
        self.assertIn("using weak", questions[1].prompt)
        self.assertEqual(questions[2].difficulty, "analysis")
        self.assertGreaterEqual(len(questions[2].expected_elements), 3)

    def test_mastery_drift_adjusts_next_pending_day(self):
        goal_id = self.create_goal()
        db.update_mastery(goal_id, "fixtures", 0.1)
        db.update_mastery(goal_id, "fixtures", 0.1)

        change = db.maybe_auto_replan(goal_id)
        plan = db.get_active_plan(goal_id)
        first_day = db.get_plan_days(plan.id)[0]

        self.assertIsNotNone(change)
        self.assertIn("fixtures", change["weak_concepts"])
        self.assertTrue(first_day.activities[0].startswith("Targeted review:"))

    def test_roadmap_uses_real_days_mastery_and_targets(self):
        goal_id = self.create_goal()
        target = db.create_roadmap_target(
            goal_id,
            "Teach fixtures without notes",
            "Explain scope, setup, teardown, and isolation.",
            "mastery",
            0.8,
            "fixtures",
        )
        db.update_roadmap_target(target["id"], True)
        roadmap = db.roadmap(goal_id)

        self.assertEqual(len(roadmap["checkpoints"]), 2)
        self.assertEqual(roadmap["checkpoints"][0]["state"], "ready")
        self.assertTrue(roadmap["targets"][0]["completed"])
        self.assertGreater(roadmap["overall_progress"], 0)

    def test_plan_quality_rejects_vague_unmeasurable_days(self):
        with self.assertRaisesRegex(ValueError, "quality checks"):
            agents.validate_plan_quality(sample_plan(), 2)


if __name__ == "__main__":
    unittest.main()
