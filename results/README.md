# results

Raw output of the benchmarks, committed so the numbers in the top-level README can be checked.

| File | Produced by | Content |
| --- | --- | --- |
| `detection_windows.csv` | `driftwatch bench detection` | one row per evaluated window: scenario, seed, window end, phase relative to the drift onset, label counts, evaluation time and one boolean column per detector |
| `detection_summary.csv` | same | per scenario and detector: false positive rate on windows before the onset, true positive rate on windows after it, runs that detected the drift and the median delay |
| `detection_meta.json` | same | benchmark parameters, `MonitorConfig` used, mean evaluation time per window, hardware and package versions |
| `overhead.json` | `driftwatch bench overhead` | SDK call latency, end-to-end throughput, ingest request latency, local store read and write timings, hardware and package versions |

Phases: a window is `pre` when it ends before the onset, `post` when it starts after it and `mixed` when it straddles it. Mixed windows count towards the detection delay but not towards either rate. The no-drift control has only `pre` windows.

Reproduce with `make bench`. The detection benchmark downloads the Adult dataset on first use and took about 25 minutes for 5 seeds on the development machine.
