# Security policy

## Supported versions

Only the latest release on `main` is supported during the alpha phase.

## Reporting

Do not open a public issue for credentials, arbitrary-code-execution vulnerabilities, unsafe model
deserialization, or a control path that could command physical hardware unexpectedly. Use GitHub's
private vulnerability reporting feature when it is enabled for the repository owner.

This project does not provide a physical-robot control interface. Treat PyTorch checkpoints as
trusted executable inputs only; do not load checkpoints from untrusted sources. ONNX and MuJoCo
artifacts must match their recorded hashes before evaluation or playback.

The immutable standing-v1 evidence records Torch 2.9.0, which has published advisories in APIs this
project does not call. The installable training extra uses a maintained Torch line; historical
environment metadata is evidence and must not be mistaken for a recommended dependency set.
