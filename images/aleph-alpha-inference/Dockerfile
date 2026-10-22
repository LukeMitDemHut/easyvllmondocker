# Thin image: stock vLLM + the Aleph Alpha plugin (provides Kolibri-1 model class
# and the kolibri1 reasoning/tool parsers).
#
# The plugin is a tiny pure-Python wheel and is installed with --no-deps so the
# base image vLLM/torch/transformers stay untouched. At the time of writing,
# upstream also publishes a ready-made image (ghcr.io/aleph-alpha/
# aleph-alpha-inference) that is amd64-only, hence this arm64 recipe.
ARG BASE_VLLM_IMAGE=vllm/vllm-openai:v0.29.0-aarch64-cu129

FROM ${BASE_VLLM_IMAGE}

RUN pip install --no-deps aleph-alpha-inference==1.0.0
