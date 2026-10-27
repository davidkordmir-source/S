# Run on Windows: python -m PyInstaller --noconfirm AtlasScout.spec
# All three pinned model snapshots are included. Runtime never downloads weights.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata, collect_dynamic_libs, collect_all, collect_delvewheel_libs_directory

root = Path(SPECPATH)
datas = [(str(root / "ui/dist"), "ui")]
for notice in ("LICENSE", "PROVENANCE.md", "ATLAS_CHANGES.md"):
    datas.append((str(root / "vendor/ecw2tiff" / notice), "third-party/ecw2tiff"))
for notice in ("LICENSE", "NOTICE.md"):
    datas.append((str(root / "vendor/dinov2" / notice), "third-party/dinov2"))
if not (root / '.model-bundle/manifest.json').is_file():
    raise RuntimeError('Run scripts/bundle-models.py before packaging the offline application')
if not (root / '.model-bundle/detail/model.safetensors').is_file():
    raise RuntimeError('The pinned offline DINOv2 model is missing; rerun scripts/bundle-models.py')
datas += [(str(root / '.model-bundle'), 'models')]
if not (root / ".raster-bundle/native-validation.json").is_file():
    raise RuntimeError("Run scripts/bundle-raster.py and validate the raster tools before packaging")
datas += [(str(root / ".raster-bundle"), "raster"), (str(root / "tests/data"), "validation-data")]
raster_data, raster_binaries, raster_imports = collect_all("rasterio")
dll_data, dll_binaries = collect_delvewheel_libs_directory("rasterio")
datas += raster_data + dll_data
binaries = collect_dynamic_libs("pillow_heif") + raster_binaries + dll_binaries
for package in ("sentence_transformers", "transformers", "certifi", "tokenizers"):
    datas += collect_data_files(package)
for distribution in ("sentence-transformers", "transformers", "torch", "scikit-learn", "huggingface-hub", "ecw2tiff", "opencv-python-headless"):
    datas += copy_metadata(distribution)

hiddenimports = raster_imports + collect_submodules("sentence_transformers") + [
    "pillow_heif", "_pillow_heif", "ecw2tiff", "ecw2tiff.ecw2tiff",
    "transformers.models.clip.modeling_clip",
    "transformers.models.clip.processing_clip",
    "transformers.models.clip.image_processing_clip",
    "transformers.models.clip.tokenization_clip",
    "transformers.models.distilbert.modeling_distilbert",
    "transformers.models.distilbert.tokenization_distilbert",
    "transformers.models.bert.tokenization_bert",
    "transformers.models.dinov2.modeling_dinov2",
    "transformers.models.auto.modeling_auto",
    "transformers.models.auto.tokenization_auto",
    "transformers.models.auto.processing_auto",
    "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
    "scipy.special._ufuncs_cxx",
]

a = Analysis([str(root / "scripts/desktop-entry.py")], pathex=[str(root / "src")],
             datas=datas, binaries=binaries, hiddenimports=hiddenimports,
             excludes=["tkinter", "tensorflow", "jax", "IPython", "matplotlib"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AtlasScout",
          debug=False, bootloader_ignore_signals=False, strip=False,
          upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AtlasScout")
