//! Header, stage record and tile index parsing (docs/SPEC.md sections 3 and 4).

use super::entropy::Stream;
use super::io::{ByteSource, FileHandle};
use super::{ColorModel, Error, ImageInfo, Result, StageInfo};

/// Everything needed to locate and interpret tile data.
pub(crate) struct Layout {
    pub(crate) info: ImageInfo,
    pub(crate) stages: Vec<StageInfo>,
    /// Global index of the first tile of each stage.
    pub(crate) stage_first_tile: Vec<u64>,
    /// Dequantization divisor `Q`.
    pub(crate) quant_divisor: u16,
    index: TileIndex,
}

enum TileIndex {
    Relative { data_start: u64, file: FileHandle, start: u64, count: u64 },
    Absolute { file: FileHandle, start: u64, count: u64 },
    Empty,
}

impl Layout {
    pub(crate) fn tile_range(&self, stage: usize, tx: u32, ty: u32) -> Result<(u64, u64)> {
        let bad = || Error::malformed("tile index out of range");
        let st = self.stages.get(stage).ok_or_else(bad)?;
        if tx >= st.tiles_x || ty >= st.tiles_y { return Err(bad()); }
        let i = self.stage_first_tile[stage] + u64::from(ty) * u64::from(st.tiles_x) + u64::from(tx);
        match &self.index {
            TileIndex::Empty => Ok((0, 0)),
            TileIndex::Relative { data_start, file, start, count } => {
                if i + 1 >= *count { return Err(bad()); }
                let mut bytes = [0u8; 16];
                file.read_at(start + i * 8, &mut bytes)?;
                let a = le_u64(&bytes, 0);
                let b = le_u64(&bytes, 8);
                if b < a { return Err(Error::malformed("tile offsets decrease")); }
                let offset = data_start.checked_add(a).ok_or_else(bad)?;
                Ok((offset, b - a))
            }
            TileIndex::Absolute { file, start, count } => {
                if i >= *count { return Err(bad()); }
                let mut bytes = [0u8; 12];
                file.read_at(start + i * 12, &mut bytes)?;
                let offset = le_u64(&bytes, 0);
                let length = u64::from(le_u32(&bytes, 8));
                if length != 0 && offset.checked_add(length).is_none_or(|end| end > file.len()) {
                    return Err(Error::malformed("tile lies outside the file"));
                }
                Ok((offset, length))
            }
        }
    }
}

fn be16(b: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([b[at], b[at + 1]])
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn le_f64(b: &[u8], at: usize) -> f64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    f64::from_le_bytes(a)
}

fn le_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn ascii_field(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// Basic image parameters shared by all versions.
struct Basics {
    color: ColorModel,
    stages: usize,
    width: u32,
    height: u32,
    bands: usize,
    tile_w: u32,
    tile_h: u32,
}

/// Parse and validate the file's header and tile index.
pub(crate) fn parse(file: &FileHandle, cache_dir: Option<&std::path::Path>) -> Result<Layout> {
    if file.len() < 4 {
        return Err(Error::malformed("file too short to be ECW"));
    }
    let mut magic = [0u8; 4];
    file.read_at(0, &mut magic)?;
    if magic[0] == 0x65 {
        parse_v12(file, cache_dir)
    } else if magic == [0, 0, 0, 0x10] {
        parse_v3(file)
    } else {
        Err(Error::unsupported("not an ECW file"))
    }
}

fn read_stage_records(
    file: &FileHandle,
    pos: u64,
    limit: u64,
    basics: &Basics,
) -> Result<(Vec<StageInfo>, u64)> {
    let rec_len = 9 + 4 * basics.bands as u64;
    let total = rec_len * basics.stages as u64;
    if pos.checked_add(total).is_none_or(|e| e > limit) {
        return Err(Error::malformed("stage records do not fit in the file"));
    }
    let bytes = file.read_vec(pos, total)?;
    let mut stages = Vec::with_capacity(basics.stages);
    for (s, rec) in bytes.chunks_exact(rec_len as usize).enumerate() {
        if usize::from(rec[0]) != s {
            return Err(Error::malformed("stage record has the wrong index"));
        }
        let quant_steps = rec[9..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|q| u32::from_be_bytes(*q))
            .collect();
        stages.push(StageInfo {
            width: be32(rec, 1),
            height: be32(rec, 5),
            tiles_x: 0,
            tiles_y: 0,
            quant_steps,
        });
    }
    validate_geometry(basics, &mut stages)?;
    Ok((stages, pos + total))
}

/// Check the stage grid sizes and fill in tile counts.
fn validate_geometry(b: &Basics, stages: &mut [StageInfo]) -> Result<()> {
    if b.color == ColorModel::YCbCr && b.bands < 3 {
        return Err(Error::malformed("YCbCr image with fewer than 3 bands"));
    }
    for st in stages.iter_mut() {
        if st.width >= b.width || st.height >= b.height {
            return Err(Error::malformed("stage grid not smaller than the image"));
        }
        // Grids need not be exact halves of the next finer one: section 6
        // defines reconstruction for any sizes through clamped reads. An
        // empty grid, though, has nothing to clamp to.
        if st.width == 0 || st.height == 0 {
            return Err(Error::malformed("stage grid has zero size"));
        }
        st.tiles_x = st.width.div_ceil(b.tile_w);
        st.tiles_y = st.height.div_ceil(b.tile_h);
    }
    Ok(())
}

fn first_tiles(stages: &[StageInfo]) -> (Vec<u64>, u64) {
    let mut firsts = Vec::with_capacity(stages.len());
    let mut n = 0u64;
    for st in stages {
        firsts.push(n);
        n += u64::from(st.tiles_x) * u64::from(st.tiles_y);
    }
    (firsts, n)
}

fn parse_v12(file: &FileHandle, cache_dir: Option<&std::path::Path>) -> Result<Layout> {
    let flen = file.len();
    if flen < 22 {
        return Err(Error::malformed("truncated header"));
    }
    let version = {
        let mut v = [0u8; 2];
        file.read_at(0, &mut v)?;
        v[1]
    };
    if version != 1 && version != 2 {
        return Err(Error::unsupported(format!(
            "unsupported ECW version {version}"
        )));
    }
    let hlen = if version == 2 { 89 } else { 22 };
    if flen < hlen {
        return Err(Error::malformed("truncated header"));
    }
    let h = file.read_vec(0, hlen)?;
    let color = ColorModel::from_code(h[3])?;
    let basics = Basics {
        color,
        stages: usize::from(h[4]),
        width: be32(&h, 6),
        height: be32(&h, 10),
        bands: usize::from(be16(&h, 14)),
        tile_w: u32::from(be16(&h, 18)),
        tile_h: u32::from(be16(&h, 20)),
    };
    let quant_divisor = be16(&h, 16);
    if basics.stages == 0 {
        return Err(Error::malformed("stage count is zero"));
    }
    if h[5] != 4 {
        return Err(Error::unsupported("subbands per stage is not 4"));
    }
    if basics.bands == 0 || quant_divisor == 0 || basics.tile_w == 0 || basics.tile_h == 0 {
        return Err(Error::malformed("zero band count, divisor or tile size"));
    }
    let mut info = ImageInfo {
        format_version: version,
        width: basics.width,
        height: basics.height,
        bands: basics.bands,
        color_model: color,
        stages: basics.stages,
        tile_width: basics.tile_w,
        tile_height: basics.tile_h,
        target_ratio: 1,
        cell_unit_code: 0,
        cell_size_x: 1.0,
        cell_size_y: 1.0,
        origin_x: 0.0,
        origin_y: 0.0,
        datum: "RAW".into(),
        projection: "RAW".into(),
        epsg: None,
    };
    if version == 2 {
        info.target_ratio = be16(&h, 22);
        info.cell_unit_code = h[24];
        info.cell_size_x = le_f64(&h, 25);
        info.cell_size_y = le_f64(&h, 33);
        info.origin_x = le_f64(&h, 41);
        info.origin_y = le_f64(&h, 49);
        info.datum = ascii_field(&h[57..73]);
        info.projection = ascii_field(&h[73..89]);
    }

    let (stages, pos) = read_stage_records(file, hlen, flen, &basics)?;
    let (stage_first_tile, n_tiles) = first_tiles(&stages);

    // Tile directory.
    if pos.checked_add(4).is_none_or(|e| e > flen) {
        return Err(Error::malformed("missing tile directory"));
    }
    let mut p = [0u8; 4];
    file.read_at(pos, &mut p)?;
    let dir_len = u64::from(u32::from_be_bytes(p));
    let dir_start = pos + 4;
    let data_start = dir_start + dir_len;
    if dir_len == 0 || data_start > flen {
        return Err(Error::malformed("bad tile directory length"));
    }
    // Empty tiles cost nothing, but an index far larger than the file is
    // not plausible; refuse it rather than allocate it.
    if n_tiles > flen + (1 << 20) {
        return Err(Error::malformed("implausibly many tiles for the file size"));
    }
    let index = decode_directory(file, dir_start, data_start, n_tiles + 1, cache_dir)?;
    Ok(Layout { info, stages, stage_first_tile, quant_divisor, index })
}

/// Compressed directories are decoded once into a bounded-buffer disk cache.
/// RAW/v3 directories are read in place; no per-tile vector resides in RAM.
fn decode_directory(file: &FileHandle, start: u64, end: u64, count: u64, cache_dir: Option<&std::path::Path>) -> Result<TileIndex> {
    use std::hash::{Hash, Hasher};
    use std::io::{BufWriter, Write};
    use std::sync::atomic::{AtomicU64, Ordering};
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    static BUILD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let mut encoding = [0u8; 1];
    file.read_at(start, &mut encoding)?;
    if encoding[0] == 5 { return Ok(TileIndex::Empty); }
    if encoding[0] == 1 {
        if count.checked_mul(8).is_none_or(|n| n > end - start - 1) {
            return Err(Error::malformed("raw directory is truncated"));
        }
        let mut last = [0u8; 8];
        file.read_at(start + 1 + (count - 1) * 8, &mut last)?;
        if end.checked_add(u64::from_le_bytes(last)).is_none_or(|n| n > file.len()) {
            return Err(Error::malformed("tile data extends beyond end of file"));
        }
        return Ok(TileIndex::Relative { data_start: end, file: FileHandle::open(file.path())?, start: start + 1, count });
    }
    let metadata = std::fs::metadata(file.path()).map_err(|e| Error::io(&e))?;
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    file.path().hash(&mut hash);
    metadata.len().hash(&mut hash);
    metadata.modified().map_err(|e| Error::io(&e))?.hash(&mut hash);
    start.hash(&mut hash); end.hash(&mut hash); count.hash(&mut hash);
    let key = hash.finish();
    let cache = cache_dir.map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::env::temp_dir().join("AtlasScoutEcwIndex"));
    std::fs::create_dir_all(&cache).map_err(|e| Error::io(&e))?;
    let target = cache.join(format!("v1-{key:016x}.idx"));
    let expected = count.checked_mul(8).and_then(|n| n.checked_add(16))
        .ok_or_else(|| Error::malformed("directory length overflow"))?;
    let valid = || -> bool {
        if let Ok(existing) = FileHandle::open(&target) {
            let mut header = [0u8; 16];
            existing.len() == expected && existing.read_at(0, &mut header).is_ok()
                && &header[..8] == b"ATLIDX01" && le_u64(&header, 8) == count
        } else { false }
    };
    if valid() {
        return Ok(TileIndex::Relative { data_start: end, file: FileHandle::open(&target)?, start: 16, count });
    }
    let _guard = BUILD_LOCK.lock().map_err(|_| Error::malformed("index cache lock poisoned"))?;
    if !valid() {
        let temporary = target.with_extension(format!("{}.{}.tmp", std::process::id(), SERIAL.fetch_add(1, Ordering::Relaxed)));
        let build = || -> Result<()> {
            let output = std::fs::OpenOptions::new().create_new(true).write(true).open(&temporary).map_err(|e| Error::io(&e))?;
            let mut output = BufWriter::with_capacity(256 * 1024, output);
            output.write_all(b"ATLIDX01").map_err(|e| Error::io(&e))?;
            output.write_all(&count.to_le_bytes()).map_err(|e| Error::io(&e))?;
            let mut stream = Stream::open(ByteSource::new(file, start, end, Vec::new()))?;
            let mut remaining = count;
            let mut previous = 0;
            let mut coefficients = vec![0i16; 32768];
            while remaining != 0 {
                let n = remaining.min((coefficients.len() / 4) as u64) as usize;
                stream.decode(&mut coefficients[..n * 4])?;
                for words in coefficients[..n * 4].chunks_exact(4) {
                    let value = words.iter().enumerate().fold(0u64, |v, (k, w)| v | (u64::from(*w as u16) << (16 * k)));
                    if value < previous || end.checked_add(value).is_none_or(|n| n > file.len()) {
                        return Err(Error::malformed("invalid offset in compressed directory"));
                    }
                    previous = value;
                    output.write_all(&value.to_le_bytes()).map_err(|e| Error::io(&e))?;
                }
                remaining -= n as u64;
            }
            output.flush().map_err(|e| Error::io(&e))?;
            output.get_ref().sync_all().map_err(|e| Error::io(&e))?;
            drop(output);
            // Windows does not replace an existing cache via rename.
            if std::fs::rename(&temporary, &target).is_err() && !valid() {
                if target.exists() { std::fs::remove_file(&target).map_err(|e| Error::io(&e))?; }
                std::fs::rename(&temporary, &target).map_err(|e| Error::io(&e))?;
            }
            Ok(())
        };
        let result = build();
        let _ = std::fs::remove_file(&temporary);
        result?;
    }
    Ok(TileIndex::Relative { data_start: end, file: FileHandle::open(&target)?, start: 16, count })
}

// ---------------------------------------------------------------------------
// Version 3

/// Visit the boxes in file range `[start, end)`.
fn scan_boxes(
    file: &FileHandle,
    start: u64,
    end: u64,
    mut visit: impl FnMut([u8; 4], u64, u64) -> Result<()>,
) -> Result<()> {
    let mut pos = start;
    while end.saturating_sub(pos) >= 8 {
        let mut hdr = [0u8; 8];
        file.read_at(pos, &mut hdr)?;
        let len = u64::from(be32(&hdr, 0));
        let kind = [hdr[4], hdr[5], hdr[6], hdr[7]];
        let (payload, box_end) = match len {
            0 => (pos + 8, end),
            1 => {
                if end - pos < 16 {
                    return Err(Error::malformed("truncated large box header"));
                }
                let mut l = [0u8; 8];
                file.read_at(pos + 8, &mut l)?;
                let len = u64::from_be_bytes(l);
                (pos + 16, pos.saturating_add(len))
            }
            n => (pos + 8, pos + n),
        };
        if box_end <= pos || payload > box_end || box_end > end {
            return Err(Error::malformed("invalid box length"));
        }
        visit(kind, payload, box_end)?;
        pos = box_end;
    }
    Ok(())
}

#[derive(Default)]
struct GeoTags {
    scale: Option<(f64, f64)>,
    origin: Option<(f64, f64)>,
    geographic: bool,
    epsg: Option<u16>,
}

fn parse_gtif(file: &FileHandle, start: u64, end: u64) -> Result<GeoTags> {
    let mut g = GeoTags::default();
    let mut pos = start + 5;
    while end.saturating_sub(pos) >= 6 {
        let mut hdr = [0u8; 6];
        file.read_at(pos, &mut hdr)?;
        let tag = be16(&hdr, 0);
        let len = u64::from(be32(&hdr, 2));
        let body = pos + 6;
        if len > end - body {
            break;
        }
        match tag {
            33550 if len >= 16 => {
                let b = file.read_vec(body, 16)?;
                g.scale = Some((le_f64(&b, 0), le_f64(&b, 8)));
            }
            33922 if len >= 48 => {
                let b = file.read_vec(body, 48)?;
                g.origin = Some((le_f64(&b, 24), le_f64(&b, 32)));
            }
            34735 => {
                let b = file.read_vec(body, len)?;
                for grp in b.as_chunks::<8>().0.iter().skip(1) {
                    let v = |i: usize| u16::from_le_bytes([grp[2 * i], grp[2 * i + 1]]);
                    let (key, loc, value) = (v(0), v(1), v(3));
                    if key == 1024 {
                        g.geographic = value == 2;
                    } else if (key == 2048 || key == 3072)
                        && loc == 0
                        && value != 0
                        && value != 32767
                    {
                        g.epsg = Some(value);
                    }
                }
            }
            _ => {}
        }
        pos = body + len;
    }
    Ok(g)
}

fn parse_v3(file: &FileHandle) -> Result<Layout> {
    let flen = file.len();
    let mut ihdr: Option<Vec<u8>> = None;
    let mut gtif: Option<(u64, u64)> = None;
    let mut ecwc: Option<(u64, u64)> = None;
    scan_boxes(file, 0, flen, |kind, start, end| {
        match &kind {
            b"ecwh" if ihdr.is_none() => {
                scan_boxes(file, start, end, |k, s, e| {
                    if &k == b"ihdr" && ihdr.is_none() {
                        if e - s < 22 {
                            return Err(Error::malformed("ihdr box too short"));
                        }
                        ihdr = Some(file.read_vec(s, 22)?);
                    }
                    Ok(())
                })?;
            }
            b"gtif" if gtif.is_none() => gtif = Some((start, end)),
            b"ecwc" if ecwc.is_none() => ecwc = Some((start, end)),
            _ => {}
        }
        Ok(())
    })?;
    let h = ihdr.ok_or_else(|| Error::malformed("missing ihdr box"))?;
    let (cs, ce) = ecwc.ok_or_else(|| Error::malformed("missing ecwc box"))?;

    let color = ColorModel::from_code(h[2])?;
    let basics = Basics {
        color,
        height: be32(&h, 3),
        width: be32(&h, 7),
        bands: usize::from(be16(&h, 11)),
        stages: usize::from(h[13]),
        tile_w: u32::from(be16(&h, 15)),
        tile_h: u32::from(be16(&h, 17)),
    };
    if h[14] != 4 {
        return Err(Error::unsupported("subbands per stage is not 4"));
    }
    if basics.stages == 0 || basics.bands == 0 || basics.tile_w == 0 || basics.tile_h == 0 {
        return Err(Error::malformed(
            "zero stage count, band count or tile size",
        ));
    }
    let geo = match gtif {
        Some((s, e)) => parse_gtif(file, s, e)?,
        None => GeoTags::default(),
    };
    let (cell_x, mut cell_y) = geo.scale.unwrap_or((1.0, -1.0));
    if cell_y > 0.0 {
        cell_y = -cell_y;
    }
    let (origin_x, origin_y) = geo.origin.unwrap_or((0.0, 0.0));
    let info = ImageInfo {
        format_version: 3,
        width: basics.width,
        height: basics.height,
        bands: basics.bands,
        color_model: color,
        stages: basics.stages,
        tile_width: basics.tile_w,
        tile_height: basics.tile_h,
        target_ratio: be16(&h, 20),
        cell_unit_code: if geo.geographic { 2 } else { 1 },
        cell_size_x: cell_x,
        cell_size_y: cell_y,
        origin_x,
        origin_y,
        datum: String::new(),
        projection: match geo.epsg {
            Some(code) => format!("EPSG:{code}"),
            None => "RAW".into(),
        },
        epsg: geo.epsg,
    };

    let (stages, pos) = read_stage_records(file, cs, ce, &basics)?;
    let (stage_first_tile, n_tiles) = first_tiles(&stages);
    if pos.checked_add(5).is_none_or(|e| e > ce) {
        return Err(Error::malformed("truncated codestream header"));
    }
    let mut enc = [0u8; 1];
    file.read_at(pos + 4, &mut enc)?;
    if enc[0] != 1 {
        return Err(Error::unsupported(format!(
            "unsupported tile index encoding {}",
            enc[0]
        )));
    }
    let table_start = pos + 5;
    let _table_len = n_tiles
        .checked_mul(12)
        .filter(|&l| l <= ce - table_start)
        .ok_or_else(|| Error::malformed("tile table does not fit in the codestream box"))?;
    Ok(Layout {
        info,
        stages,
        stage_first_tile,
        quant_divisor: 1,
        index: TileIndex::Absolute { file: FileHandle::open(file.path())?, start: table_start, count: n_tiles },
    })
}
