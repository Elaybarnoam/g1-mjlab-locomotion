# Robot, policy, and PD contract

The active robot is the 29-actuator Unitree G1 supplied by the pinned mjlab revision. Joint order is
resolved from the live model and stored in every qualified contract; native evaluation never assumes
an alphabetical or hard-coded MuJoCo address order.

## Actor vector: 99 values

| Field | Size | Meaning |
| --- | ---: | --- |
| base linear velocity | 3 | pelvis-site velocimeter, pelvis frame, m/s |
| base angular velocity | 3 | pelvis gyro, pelvis frame, rad/s |
| projected gravity | 3 | world gravity direction projected into pelvis frame |
| joint position | 29 | position relative to nominal pose plus encoder-bias convention, rad |
| joint velocity | 29 | joint velocity in contract order, rad/s |
| previous applied action | 29 | action that actually reached the actuator adapter |
| command | 3 | desired planar x/y velocity and yaw rate; all zero for standing-v1 |

The critic adds 12 privileged values for foot/contact information and has width 111. Privileged
values are unavailable to the actor and are not required by native deployment.

## Action and timing

The 29 actor outputs map one-to-one to position-controlled joints:

```text
q_target = q_nominal + action_scale * applied_action - encoder_bias
```

Standing-v1 does not apply an additional explicit software PD torque. MuJoCo position actuators
apply joint-specific stiffness and damping, armature, and force limits at the 0.005-second physics
step. The actor runs every four physics steps, or 50 Hz.

The native adapter reconstructs the exact actor fields from named MuJoCo sensors and addresses,
applies the same previous-action convention, and sends the same position targets. Applying another
PD loop around these targets would double the controller and is unsupported.

The current simulated pelvis linear-velocity observation is not directly measured by a physical
IMU. Sim-to-real work must introduce a deployable estimator/observation contract and retrain or
distill before hardware use.
