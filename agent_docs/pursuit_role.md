# UAV Pursuit Environment (Roles, Spaces, Rewards)

## Environment
- **World**: 2D square with bounds `[-world_size, world_size]` and time step `dt`.
- **Agents**: `num_hunters` + `num_blockers` + 1 target (evasion). Roles are fixed by index.
- **Episode end**: capture or `max_steps`.
- **Target patrol mode**: if `target_policy_source=patrol`, the target action is overridden to follow waypoints loaded from `target_patrol_path` (with optional named routes).
- **Pursuit sharing**: if any pursuer (hunter or blocker) detects the target, the target’s relative position and velocity are shared with all pursuers for that step.

## Roles
- **Hunter**: pursuer with highest max speed, smaller perception range.
- **Blocker**: pursuer with larger perception range, lower max speed.
- **Target**: evader (single agent), may be learned or patrol-driven.

## State / Observation Space
Per-agent observation vector:
- Own position `(x, y)` and velocity `(vx, vy)`.
- For every other agent: relative position `(dx, dy)`, relative velocity `(dvx, dvy)`, and distance `d`.
  - If the other agent is outside the observer’s perception range, `(dx, dy, dvx, dvy, d)` is zeroed.
  - For pursuers, target observations are shared within the team when any pursuer detects the target.

Dimension: `obs_dim = 4 + (agent_num - 1) * 5`.

Centralized observation (for shared critics): concatenation of all agents’ observations with dimension `obs_dim * agent_num`.

## Action Space
Continuous 2D action for each agent: `Box(low=-1, high=1, shape=(2,))`.
- Actions are scaled by role-specific max speed and applied as velocity updates.
- Positions are updated with `dt` and clipped to world bounds.

## Rewards
Let `d` be the distance from a pursuer to the target, and `min_distance` be the closest hunter distance to the target.
- **Hunter**: `-d + 10` on capture, otherwise `-d`.
- **Blocker**: `-0.7 * d + 6` on capture, otherwise `-0.7 * d`.
- **Target**: `min_distance - 12` on capture, otherwise `min_distance`.
- **Speed penalty**: all agents receive `-speed_penalty * speed^2` to discourage meaningless high-speed motion.

## Capture Condition
Capture occurs when any hunter stays within `capture_radius` of the target for `capture_steps` consecutive steps.


