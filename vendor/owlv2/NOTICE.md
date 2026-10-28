OWLv2 object detection model attribution

Source model: https://huggingface.co/google/owlv2-base-patch16-ensemble
Architecture/source: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit
License: Apache License 2.0. The builder verifies the official model card license and every downloaded source file against its immutable revision and official checksum.

Atlas modifications: store weights as fp16 safetensors, load float32 locally; use the installed Transformers implementation, native image/text query embeddings and class/box heads. The class head can run over cached normalized scene embeddings. No remote repository Python code is executed. Input color is excluded unless explicitly required by the user. Revision and file hashes are included in models/detector/atlas-source.json and AI-MODELS.json.
