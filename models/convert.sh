#!/bin/sh
# convert.sh <hf repo> <revision> <int8|fp16> <out dir>: Whisper -> OpenVINO IR (what the GPU runtime loads),
# then a smoke test: load it and transcribe 3 s of silence on the CPU.
set -eu
optimum-cli export openvino --model "$1" --revision "$2" --task automatic-speech-recognition-with-past \
  --weight-format "$3" "$4"
python3 - "$4" <<'PY'
import sys, openvino_genai as og
pipe = og.WhisperPipeline(sys.argv[1], "CPU")
print("smoke test:", repr(pipe.generate([0.0] * 48000, language="<|he|>", task="transcribe").texts))
PY
