#!/bin/sh
# convert.sh <hf repo> <revision> <int8|fp16> <out dir>: Whisper -> OpenVINO IR (what the GPU runtime loads),
# then a smoke test: load it and transcribe 3 s of silence on the CPU.
set -eu
src=$(python3 -c 'import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], revision=sys.argv[2]))' "$1" "$2")
optimum-cli export openvino --model "$src" --task automatic-speech-recognition-with-past --weight-format "$3" "$4"
python3 - "$4" <<'PY'
import sys, openvino_genai as og
pipe = og.WhisperPipeline(sys.argv[1], "CPU")
print("smoke test:", repr(pipe.generate([0.0] * 48000, language="<|he|>", task="transcribe").texts))
PY
