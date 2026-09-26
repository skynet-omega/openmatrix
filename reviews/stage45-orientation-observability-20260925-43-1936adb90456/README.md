# Stage 4/5: local olfactory observability and matched-input tapes

Campaign 43 is a CPU-only sensor preflight. Two saved poses from the failed 2-s wind life reproduce the original antenna coordinates and Gaussian concentrations exactly from the compiled MuJoCo model. A virtual yaw shift of 2° away from the right-hand source changes bilateral mean concentration by about −0.052 and lateral contrast by +0.001630 at ms1000 but −0.002768 at ms1020. Therefore the unpaired virtual perturbation is not a clean, pose-independent error signal.

For the fixed postwind input window ms1021–1120, the virtual-minus-original lateral contrast remained negative in all 100 samples (range −0.003046 to −0.000659). Two synthetic olfactory tapes were frozen: the virtual sample and a control with the same bilateral mean but original contrast. Their difference is purely antisymmetric and valid in [0,1]; original sensor samples and recorded body commands were copied from the verified donor trace. No CNS or new body simulation was run. These tapes do not demonstrate neural sensitivity or navigation.

| Artifact | SHA-256 | Size |
| --- | --- | ---: |
| `ETAPA45_OBSERVABILIDAD_20260925_43_COMPLETO.zip` | `1936adb90456b4dec4a4a0393ca91a1edda0b27390fd6b87ffb97c3b3c5f6f0b` | 47,751,430 bytes |

The ZIP contains the original 2-s trace, compiled body model, frozen contracts, source, all 100 CPU rows, tapes, external review text, hashes and a self-contained verifier. From a clean extraction, `python -B verify_capsule.py --out CHECK.json` verifies all 28 hashed files, recomputes the CPU results in normal and optimized Python, checks deliberate corruptions and regenerates tape arrays exactly. This was executed successfully in a new directory. It does not contain a complete neural checkpoint or run the organism.

The scientific note is `campanas/etapa45_orientation_observability_20260925_43/README.md` inside the ZIP. Stage 4/5 remain open; current priority is orientation with the existing body, CNS→VNC→MN remains bounded later work, and six-leg muscles/gait later still.
