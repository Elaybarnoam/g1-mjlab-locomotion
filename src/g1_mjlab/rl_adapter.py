"""The one project-specific correction at the pinned RSL-RL environment seam."""

from typing import Any

from mjlab.rl import RslRlVecEnvWrapper


class StandingVecEnvWrapper(RslRlVecEnvWrapper):  # type: ignore[misc]
    """True termination takes precedence over a coincident time-limit timeout.

    RSL-RL's stored-current-value timeout compensation is otherwise unchanged.
    This is deliberately not advertised as terminal-observation bootstrapping.
    """

    def step(self, actions: Any) -> Any:
        obs, rewards, dones, extras = super().step(actions)
        if "time_outs" in extras:
            extras["time_outs"] = extras["time_outs"] & ~self.env.reset_terminated
        return obs, rewards, dones, extras
