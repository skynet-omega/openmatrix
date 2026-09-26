# Stage 4/5: mirrored signals and postwind raw-command replay

Campaign 42 is an exposed diagnostic, not Stage 4/5 admission. In the mirrored 400-ms pair, left/right sensory input, ORN, PN, DNb05 and command differed; the source-dependent command difference was positive in 399/400 samples and integrated only 0.057746766°. In the 2-s failed navigation life, replacing the recorded applied yaw command after wind with the recorded pre-filter command in one matched body-only replay lowered final bearing error from 27.558233° to 24.275701°. It still worsened from 19.414163° at wind end and was near the 24.248712° zero-yaw control. No CNS was simulated in this campaign.

The complete ZIP below contains the source traces, executable MuJoCo body, saved physical states, source and contract hashes, scripts, results, external review text, Jev advisory receipt and a clean-extraction verifier. It does not contain a complete 166,700-neuron checkpoint or permit rerunning the original CNS. ChatGPT explicitly did not read or execute the NPZ files; Codex executed and verified the computations locally.

| Artifact | SHA-256 | Size |
| --- | --- | ---: |
| `ETAPA45_SENAL_FILTRO_20260925_42_COMPLETO.zip` | `d5dbb40423fa39c3ca9e2fcb16fc4a72d91705b55a93adf1f82c4d148cc58f21` | 76,603,316 bytes |

From a new extraction, run `python -B verify_capsule.py --out SHORT.json`, or add `--full` to repeat the 2-s physical replay. The final ZIP was extracted at a new location and `--full` passed: 58 hashed files, all three analyses byte-exact in normal and optimized Python, deliberately corrupted fields rejected, and physical trace/final state arrays exactly reproduced. This does not rerun the CNS.

Human-readable interpretation: `campanas/etapa45_signal_chain_20260925_42/REPORT.md` inside the ZIP. The active priority remains orientation with the current body; CNS→VNC→MN is a bounded later line and six-leg muscles/gait later still.
