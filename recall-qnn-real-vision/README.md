# D3 Nano QNN cloud qualification

This is conversion tooling, not a downloadable Recall backend. The working GGUF
backend and all Recall app sources are unchanged.

Run `python tools/qnn_d3/dispatch.py` to publish the conversion source and workflow
to `thaakeno/local-dream`, branch `recall-d3-nano-qnn`. The default branch and Breeze
workflows are unaffected. GitHub Actions installs dependencies and fetches pinned
weights on its Linux runner. The exporter refuses to run outside GitHub Actions.
The AI Hub credential is the encrypted `D3_QAI_HUB_API_TOKEN` Actions secret; it is
never part of these files or uploaded artifacts.

The first run is https://github.com/thaakeno/local-dream/actions/runs/38317802422.
Its first decoder layer compiled as AI Hub job `jpy87q3lg` and linked into an HTP
context as `jgj3wo21p`. Hosted inference `jp2ov496g` completed, but its FP16 output
contains non-finite values. The profile reports 22,887 microseconds for this
single layer and 285,814 microseconds first load. This is an invalid numerical
result, not a usable model or a full-model speed benchmark. The pipeline correctly
stopped before publishing a download or changing Recall.

Tensor diagnostics (`D3_DIAGNOSE_LAYER00=true`) run before
attempting further model conversion. They compare a cloud CPU half-precision
reference and original-weight ONNX reference with instrumented NPU tensors. CPU
here is only a reference for numerical validation; there is no app CPU fallback.
The diagnostic never clamps, substitutes or repairs model outputs. Its graph
includes intermediate maximum-magnitude probes; compiler fusion can change when
observability changes, so it also retains the original final output as a check.

The completed first probe run (38319635105) localizes the first observed invalid
activation to `/layer/linear_attn/Exp_1`; earlier projection, convolution and
normalization probes are finite. PyTorch FP16 and ONNX FP32 references are finite.
The deeper probe run 38352863929 confirms the input is finite and nonpositive
(-291.25 to 0 on HTP), while the Exp output includes negative/non-finite values.
`D3_DIAGNOSE_EXP_ONLY=true` selects a separate isolated test using that exact ONNX
reference input. It compares direct Exp with the algebraically identical sigmoid
ratio on nonpositive inputs; it does not modify or qualify the production model.

Isolated run 38353726340 reproduces the fault without the decoder graph:
`jp0ov6jep` returns 16,730 non-finite values for direct Exp; the sigmoid ratio
returns entirely finite results, max absolute error 0.000976934 and RMSE
0.000198064 against float64 Exp on the same reference fixture. This validates
the candidate at the operation boundary only, not full-model ranking quality.

Run 38354453227 qualified the complete first layer with `D3_STABLE_DECAY=true`.
Its QNN output is entirely finite: max absolute error 0.0276765, RMSE 0.000478563
versus original PyTorch; the lowered ONNX max error is 0.00000596046. Hosted
profile reports estimated inference 22,385 us and first load 291,718 us for
this single layer. Jobs: compile jgk6982o5, link jpe6ly70g, inference jp8j41kx5.
The explicit ONNX lowering statically proves
nonpositive inputs before expressing Exp as sigmoid(x)/sigmoid(-x); unknown or
positive inputs remain untouched. Cloud regression checks cover sign proof,
exclusions, idempotence and FP32 ONNX parity before any new model compilation.
No trained weights, epsilons or output policies are changed.

Current run 38355815962 uses `D3_FEASIBILITY_ONLY=true` to qualify both decoder
types, vision, embedding insertion and readout before compiling the remaining
layers. It reuses the corrected first layer only on exact graph/fixture hashes.
Reports remain non-deployable. Completed model contexts stay on AI Hub by job
ID; successful temporary context downloads are removed on the cloud runner to
avoid uploading gigabytes of duplicate weights into qualification artifacts.

The hosted target reports `chipset:sm8850-ad`, `hexagon:v81`, and FP16 HTP support
(Samsung Galaxy S26 Family, Android 16). It matches the phone's NPU generation,
but hosted timings are not measurements on thaakeno's POCO.

New component reuse requires SHA-256 matches for exported graph/data and exact
test inputs. Reports predating those hashes cannot silently qualify a changed
graph or test fixture.

`dispatch.py --monitor` publishes a separate bounded metadata inspection workflow.
Rerun that workflow to inspect the latest 20 jobs, filtered to this experiment.
It prints component job URLs, statuses and hosted profile timings. Failed-job logs
are fetched on the cloud runner only. It never inspects private Recall media.

The conversion preserves Nano's trained weights and noncausal full-attention mask;
DeltaNet remains causal. It targets all neural components: vision, embedding
lookup/visual-token insertion, 24 decoder layers, final norm and 255-code readout.
The initial fixed buckets are 1024 vision patches (256 merged visual tokens) and
640 prompt tokens. They are a qualification configuration, not full detail-option
coverage. Inputs above a bucket must fail or select a larger compiled bucket,
never be silently truncated.

ONNX exports must match their PyTorch tensor graph. Compiled contexts are executed
on AI Hub with profiling and numerical differences recorded. The report retains
`deployable=false`: component testing with synthetic inputs does not establish
end-to-end parity against official D3 image prompts or quality equivalence.
The published vision export specifically warns that FP16 changes its outputs.
Real-image prompt/M-RoPE/probability parity and ranking checks remain mandatory
before adding a download option. Never advertise an unverified context as
quality-preserving or faster than Recall's existing backend.

Sources:

- [Official Nano checkpoint](https://huggingface.co/vllm-sr/d3-nano/tree/9d9b937f5f117f78c030ea7ba677c0b3d328ec58)
- [Published tensor graphs, Apache-2.0 repository metadata](https://huggingface.co/FluidInference/d3-coreml/tree/62b6a996e15342bdd3d608030d8f445f11a52367/nano/python)
- [AI Hub ONNX, DLC and context-binary compilation](https://workbench.aihub.qualcomm.com/docs/hub/compile_examples.html)
- [AI Hub job inspection and profiling](https://workbench.aihub.qualcomm.com/docs/hub/jobs.html)
