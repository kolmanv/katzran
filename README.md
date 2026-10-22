# katzran

A small local speech-to-text service: audio in, text out, nothing kept. Built for Hebrew, Russian and English,
including notes that switch language in the middle. Runs on an Intel iGPU through OpenVINO.

Work in progress. Models: `models/` and the `models` workflow, which converts
[ivrit-ai/whisper-large-v3-turbo](https://huggingface.co/ivrit-ai/whisper-large-v3-turbo) (Apache-2.0) and
[openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (MIT) to OpenVINO IR.
