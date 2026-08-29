#!/usr/bin/env python3
# framework.py — the world-dreamer architecture, as a shared interface.
#
# A *world dreamer* has exactly two parts:
#
#   WorldModel :  a predictive map  state -> next state (or state -> transition).
#                 It is trained densely, on self-supervised prediction error.
#   Dreamer    :  a planner that *optimizes inside the world model* — it
#                 imagines candidate plans, scores them against the model, and
#                 acts on the best.  Reward is sparse and only touches the
#                 planner; the model never sees it.
#
# The same two-part skeleton is instantiated on two different substrates:
#
#   continuous (pursuit)  — world model = the BDH fast-weight memory
#                           (phi(s) -> v_{t+1}, a linear Hebbian map over
#                           continuous state); dreamer = receding-horizon
#                           shooting over aim headings, scored by imagined
#                           time-to-catch.  See pursuit.py and ../pursuit/bench.py.
#
#   discrete (ARC)        — world model = an induced program (a map from an
#                           input grid to an output grid, composed of discrete
#                           transformation primitives); dreamer = program
#                           search: hypothesize programs, *imagine* their output
#                           on every example, keep the one that predicts all
#                           observations.  See arc.py.
#
# The substrate is the one thing that differs.  The architecture — predict, then
# plan inside the prediction — is shared.  That is the honest scope of "a
# generalized world dreamer": a recipe, not a single trained network that
# transfers from pursuit to ARC.

class WorldModel:
    """A predictive substrate: state -> predicted next state."""

    def observe(self, state, next_state):
        """Train on one real transition (dense, self-supervised)."""
        raise NotImplementedError

    def predict(self, state):
        """Imagined next state (no learning)."""
        raise NotImplementedError


class Dreamer:
    """A planner that optimizes a plan inside a world model."""

    def plan(self, state, world_model):
        """Return the action chosen by imagining rollouts inside the model."""
        raise NotImplementedError


class WorldDreamer:
    """The composed agent: a world model + a dreamer that plans inside it."""

    def __init__(self, world_model, dreamer):
        self.world_model = world_model
        self.dreamer = dreamer

    def step(self, state, next_state):
        """One environment step: learn from the transition, then choose an action."""
        self.world_model.observe(state, next_state)
        return self.dreamer.plan(state, self.world_model)
