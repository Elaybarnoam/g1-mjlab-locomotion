# Walking-v1 generated policy specification

Status: **unqualified development**. Checkpoint: `0865d076a7820aba3bdd1d8d97d0d60194c7989e92919b510514ffde17f929e9`.

Actor: 102 → 512 → 256 → 128 → 29 with ELU. Critic: 114 → 512 → 256 → 128 → 1.
Actor parameters: 220,701. Critic parameters: 223,233.

## Actor input vector

| Index | Value | Unit | Frame/source |
| ---: | --- | --- | --- |
| 0 | base_lin_vel[0] | m/s | pelvis/body |
| 1 | base_lin_vel[1] | m/s | pelvis/body |
| 2 | base_lin_vel[2] | m/s | pelvis/body |
| 3 | base_ang_vel[0] | rad/s | pelvis/body |
| 4 | base_ang_vel[1] | rad/s | pelvis/body |
| 5 | base_ang_vel[2] | rad/s | pelvis/body |
| 6 | projected_gravity[0] | 1 | pelvis/body |
| 7 | projected_gravity[1] | 1 | pelvis/body |
| 8 | projected_gravity[2] | 1 | pelvis/body |
| 9 | joint_pos[0] | rad | joint order |
| 10 | joint_pos[1] | rad | joint order |
| 11 | joint_pos[2] | rad | joint order |
| 12 | joint_pos[3] | rad | joint order |
| 13 | joint_pos[4] | rad | joint order |
| 14 | joint_pos[5] | rad | joint order |
| 15 | joint_pos[6] | rad | joint order |
| 16 | joint_pos[7] | rad | joint order |
| 17 | joint_pos[8] | rad | joint order |
| 18 | joint_pos[9] | rad | joint order |
| 19 | joint_pos[10] | rad | joint order |
| 20 | joint_pos[11] | rad | joint order |
| 21 | joint_pos[12] | rad | joint order |
| 22 | joint_pos[13] | rad | joint order |
| 23 | joint_pos[14] | rad | joint order |
| 24 | joint_pos[15] | rad | joint order |
| 25 | joint_pos[16] | rad | joint order |
| 26 | joint_pos[17] | rad | joint order |
| 27 | joint_pos[18] | rad | joint order |
| 28 | joint_pos[19] | rad | joint order |
| 29 | joint_pos[20] | rad | joint order |
| 30 | joint_pos[21] | rad | joint order |
| 31 | joint_pos[22] | rad | joint order |
| 32 | joint_pos[23] | rad | joint order |
| 33 | joint_pos[24] | rad | joint order |
| 34 | joint_pos[25] | rad | joint order |
| 35 | joint_pos[26] | rad | joint order |
| 36 | joint_pos[27] | rad | joint order |
| 37 | joint_pos[28] | rad | joint order |
| 38 | joint_vel[0] | rad/s | joint order |
| 39 | joint_vel[1] | rad/s | joint order |
| 40 | joint_vel[2] | rad/s | joint order |
| 41 | joint_vel[3] | rad/s | joint order |
| 42 | joint_vel[4] | rad/s | joint order |
| 43 | joint_vel[5] | rad/s | joint order |
| 44 | joint_vel[6] | rad/s | joint order |
| 45 | joint_vel[7] | rad/s | joint order |
| 46 | joint_vel[8] | rad/s | joint order |
| 47 | joint_vel[9] | rad/s | joint order |
| 48 | joint_vel[10] | rad/s | joint order |
| 49 | joint_vel[11] | rad/s | joint order |
| 50 | joint_vel[12] | rad/s | joint order |
| 51 | joint_vel[13] | rad/s | joint order |
| 52 | joint_vel[14] | rad/s | joint order |
| 53 | joint_vel[15] | rad/s | joint order |
| 54 | joint_vel[16] | rad/s | joint order |
| 55 | joint_vel[17] | rad/s | joint order |
| 56 | joint_vel[18] | rad/s | joint order |
| 57 | joint_vel[19] | rad/s | joint order |
| 58 | joint_vel[20] | rad/s | joint order |
| 59 | joint_vel[21] | rad/s | joint order |
| 60 | joint_vel[22] | rad/s | joint order |
| 61 | joint_vel[23] | rad/s | joint order |
| 62 | joint_vel[24] | rad/s | joint order |
| 63 | joint_vel[25] | rad/s | joint order |
| 64 | joint_vel[26] | rad/s | joint order |
| 65 | joint_vel[27] | rad/s | joint order |
| 66 | joint_vel[28] | rad/s | joint order |
| 67 | actions[0] | normalized | previous policy output |
| 68 | actions[1] | normalized | previous policy output |
| 69 | actions[2] | normalized | previous policy output |
| 70 | actions[3] | normalized | previous policy output |
| 71 | actions[4] | normalized | previous policy output |
| 72 | actions[5] | normalized | previous policy output |
| 73 | actions[6] | normalized | previous policy output |
| 74 | actions[7] | normalized | previous policy output |
| 75 | actions[8] | normalized | previous policy output |
| 76 | actions[9] | normalized | previous policy output |
| 77 | actions[10] | normalized | previous policy output |
| 78 | actions[11] | normalized | previous policy output |
| 79 | actions[12] | normalized | previous policy output |
| 80 | actions[13] | normalized | previous policy output |
| 81 | actions[14] | normalized | previous policy output |
| 82 | actions[15] | normalized | previous policy output |
| 83 | actions[16] | normalized | previous policy output |
| 84 | actions[17] | normalized | previous policy output |
| 85 | actions[18] | normalized | previous policy output |
| 86 | actions[19] | normalized | previous policy output |
| 87 | actions[20] | normalized | previous policy output |
| 88 | actions[21] | normalized | previous policy output |
| 89 | actions[22] | normalized | previous policy output |
| 90 | actions[23] | normalized | previous policy output |
| 91 | actions[24] | normalized | previous policy output |
| 92 | actions[25] | normalized | previous policy output |
| 93 | actions[26] | normalized | previous policy output |
| 94 | actions[27] | normalized | previous policy output |
| 95 | actions[28] | normalized | previous policy output |
| 96 | command[0] | m/s,m/s,rad/s | pelvis heading |
| 97 | command[1] | m/s,m/s,rad/s | pelvis heading |
| 98 | command[2] | m/s,m/s,rad/s | pelvis heading |
| 99 | phase_sin[0] | 1 | host gait clock |
| 100 | phase_cos[0] | 1 | host gait clock |
| 101 | walk_blend[0] | 1 | host state |

## Action and PD equation

`q_target = q_nominal + action_scale × clipped_action − encoder_bias`.

The 29 output mappings, normalizer tensors, layer tensors derived shapes, PPO settings, equations,
reference lineage, and worked captured control step are available in `policy-spec.json`. The missing
worked rollout minibatch is explicitly labeled; no optimizer or rollout history was reconstructed.
