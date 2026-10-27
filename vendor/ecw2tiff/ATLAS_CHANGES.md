Based on https://github.com/itamarwe/ecw2tiff commit b50288e43fa1cae835e58ecb9e9fa9f7a49ed97f (MIT).
Atlas changes: file-backed indexes, bounded streaming decompression of compressed v1/v2 directories to an atomic reusable disk cache, lazy v3/raw directory reads with per-access validation, Python constructor releases GIL, pure-Rust build without native JPEG SDK.
The full upstream license and provenance are retained.
