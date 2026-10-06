//! Small Arrow helpers shared by the `.kiln` tables and the evolution archive: column builders, IPC
//! encode/decode, and tolerant column readers (any integer width, utf8/large_utf8/utf8_view, list/large_list),
//! so files written by pyarrow or polars read back without casts.

use std::io::Cursor;
use std::sync::Arc;

use arrow_array::builder::{
    Float64Builder, ListBuilder, MapBuilder, StringBuilder, UInt8Builder, UInt32Builder,
};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray, UInt8Array,
    UInt16Array, UInt32Array, UInt64Array,
};
use arrow_ipc::reader::{FileReader, StreamReader};
use arrow_ipc::writer::{FileWriter, StreamWriter};
use arrow_schema::{DataType, Field, Schema};

#[allow(clippy::disallowed_types)]
type Meta = std::collections::HashMap<String, String>;
use kiln_ir::common::Diagnostic;

pub const CODE: &str = "E-TRACE-ARROW";

fn err(msg: impl Into<String>) -> Diagnostic {
    Diagnostic::error(CODE, msg)
}

/// A table under construction: named columns with declared nullability. The schema is derived from the
/// arrays, so nested field names always match what the builders produce.
#[derive(Default)]
pub struct Cols {
    fields: Vec<Field>,
    arrays: Vec<ArrayRef>,
}

impl Cols {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, name: &str, nullable: bool, a: ArrayRef) -> &mut Self {
        self.fields
            .push(Field::new(name, a.data_type().clone(), nullable));
        self.arrays.push(a);
        self
    }

    pub fn batch(self, table: &str, version: &str) -> RecordBatch {
        let schema = Schema::new(self.fields).with_metadata(Meta::from([
            ("kiln.table".to_string(), table.to_string()),
            ("kiln.schema_version".to_string(), version.to_string()),
        ]));
        RecordBatch::try_new(Arc::new(schema), self.arrays).expect("columns match their schema")
    }
}

pub fn utf8<'a>(v: impl IntoIterator<Item = Option<&'a str>>) -> ArrayRef {
    Arc::new(StringArray::from_iter(v))
}

pub fn utf8s<'a>(v: impl IntoIterator<Item = &'a str>) -> ArrayRef {
    Arc::new(StringArray::from_iter_values(v))
}

pub fn f64s(v: impl IntoIterator<Item = Option<f64>>) -> ArrayRef {
    Arc::new(Float64Array::from_iter(v))
}

pub fn f64v(v: impl IntoIterator<Item = f64>) -> ArrayRef {
    Arc::new(Float64Array::from_iter_values(v))
}

pub fn u8s(v: impl IntoIterator<Item = Option<u8>>) -> ArrayRef {
    Arc::new(UInt8Array::from_iter(v))
}

pub fn u16s(v: impl IntoIterator<Item = Option<u16>>) -> ArrayRef {
    Arc::new(UInt16Array::from_iter(v))
}

pub fn u32s(v: impl IntoIterator<Item = Option<u32>>) -> ArrayRef {
    Arc::new(UInt32Array::from_iter(v))
}

pub fn u64s(v: impl IntoIterator<Item = Option<u64>>) -> ArrayRef {
    Arc::new(UInt64Array::from_iter(v))
}

pub fn i32s(v: impl IntoIterator<Item = Option<i32>>) -> ArrayRef {
    Arc::new(Int32Array::from_iter(v))
}

pub fn i64s(v: impl IntoIterator<Item = Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from_iter(v))
}

pub fn list_f64<'a>(v: impl IntoIterator<Item = Option<&'a [f64]>>) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new());
    for row in v {
        match row {
            Some(xs) => {
                b.values().append_slice(xs);
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

pub fn list_u32<'a>(v: impl IntoIterator<Item = Option<&'a [u32]>>) -> ArrayRef {
    let mut b = ListBuilder::new(UInt32Builder::new());
    for row in v {
        match row {
            Some(xs) => {
                b.values().append_slice(xs);
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

pub fn list_utf8<'a>(v: impl IntoIterator<Item = Option<&'a [String]>>) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new());
    for row in v {
        match row {
            Some(xs) => {
                for x in xs {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

/// `map<utf8, f64>`, keys in the given (sorted) order.
pub fn map_utf8_f64<'a, I>(v: impl IntoIterator<Item = I>) -> ArrayRef
where
    I: IntoIterator<Item = (&'a str, f64)>,
{
    let mut b = MapBuilder::new(None, StringBuilder::new(), Float64Builder::new());
    for row in v {
        for (k, x) in row {
            b.keys().append_value(k);
            b.values().append_value(x);
        }
        b.append(true).expect("map row");
    }
    Arc::new(b.finish())
}

/// `map<u8, f64>`.
pub fn map_u8_f64<'a, I>(v: impl IntoIterator<Item = I>) -> ArrayRef
where
    I: IntoIterator<Item = &'a (u8, f64)>,
{
    let mut b = MapBuilder::new(None, UInt8Builder::new(), Float64Builder::new());
    for row in v {
        for (k, x) in row {
            b.keys().append_value(*k);
            b.values().append_value(*x);
        }
        b.append(true).expect("map row");
    }
    Arc::new(b.finish())
}

pub fn to_ipc_file(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut w = FileWriter::try_new(&mut out, &batch.schema()).expect("ipc writer");
        w.write(batch).expect("ipc write");
        w.finish().expect("ipc finish");
    }
    out
}

pub fn to_ipc_stream(batch: &RecordBatch) -> Vec<u8> {
    let mut out = Vec::new();
    {
        let mut w = StreamWriter::try_new(&mut out, &batch.schema()).expect("ipc writer");
        w.write(batch).expect("ipc write");
        w.finish().expect("ipc finish");
    }
    out
}

const FILE_MAGIC: &[u8] = b"ARROW1";

/// Batches of an Arrow IPC file, or of one or more concatenated IPC streams (the append-only archive form).
/// A truncated last stream (a writer died mid-batch) yields the complete batches before it and
/// `partial = true`.
pub fn read_ipc(bytes: &[u8]) -> Result<(Vec<RecordBatch>, bool), Diagnostic> {
    if bytes.starts_with(FILE_MAGIC) {
        let r = FileReader::try_new(Cursor::new(bytes.to_vec()), None)
            .map_err(|e| err(format!("bad Arrow IPC file: {e}")))?;
        let batches: Result<Vec<_>, _> = r.collect();
        return batches
            .map(|b| (b, false))
            .map_err(|e| err(format!("bad Arrow IPC batch: {e}")));
    }
    let mut cur = Cursor::new(bytes);
    let mut out = Vec::new();
    while (cur.position() as usize) < bytes.len() {
        let start = cur.position();
        let reader = match StreamReader::try_new(&mut cur, None) {
            Ok(r) => r,
            Err(e) if out.is_empty() && start == 0 => {
                return Err(err(format!("not an Arrow IPC file or stream: {e}")));
            }
            Err(_) => return Ok((out, true)),
        };
        for b in reader {
            match b {
                Ok(b) => out.push(b),
                Err(_) => return Ok((out, true)),
            }
        }
        if cur.position() == start {
            break;
        }
    }
    Ok((out, false))
}

pub struct Table<'a> {
    pub name: &'a str,
    pub batches: &'a [RecordBatch],
}

impl<'a> Table<'a> {
    pub fn new(name: &'a str, batches: &'a [RecordBatch]) -> Self {
        Self { name, batches }
    }

    pub fn has(&self, col: &str) -> bool {
        self.batches
            .first()
            .is_some_and(|b| b.column_by_name(col).is_some())
    }

    pub fn rows(&self) -> usize {
        self.batches.iter().map(RecordBatch::num_rows).sum()
    }

    fn columns(&self, col: &str) -> Option<Vec<ArrayRef>> {
        self.batches
            .iter()
            .map(|b| b.column_by_name(col).cloned())
            .collect()
    }

    fn need(&self, col: &str) -> Result<Vec<ArrayRef>, Diagnostic> {
        self.columns(col).ok_or_else(|| {
            err(format!("table {} has no column {col:?}", self.name))
                .hint("the file was written with an incompatible schema")
        })
    }

    fn bad(&self, col: &str, dt: &DataType) -> Diagnostic {
        err(format!(
            "table {} column {col:?} has unsupported type {dt}",
            self.name
        ))
    }

    fn opt_ints(&self, col: &str) -> Result<Option<Vec<Option<i128>>>, Diagnostic> {
        let Some(cs) = self.columns(col) else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(self.rows());
        for a in cs {
            out.extend(ints(&a).ok_or_else(|| self.bad(col, a.data_type()))?);
        }
        Ok(Some(out))
    }

    fn narrow<T: TryFrom<i128>>(&self, col: &str, v: i128) -> Result<T, Diagnostic> {
        T::try_from(v).map_err(|_| {
            err(format!(
                "table {} column {col:?}: {v} out of range for {}",
                self.name,
                std::any::type_name::<T>()
            ))
        })
    }

    /// Nullable column of any integer type as `T`; values outside `T`'s range are rejected.
    pub fn opt_int<T: TryFrom<i128>>(&self, col: &str) -> Result<Vec<Option<T>>, Diagnostic> {
        match self.opt_ints(col)? {
            None => Ok((0..self.rows()).map(|_| None).collect()),
            Some(v) => v
                .into_iter()
                .map(|x| x.map(|x| self.narrow(col, x)).transpose())
                .collect(),
        }
    }

    pub fn int<T: TryFrom<i128>>(&self, col: &str) -> Result<Vec<T>, Diagnostic> {
        self.need(col)?;
        self.opt_int(col)?
            .into_iter()
            .map(|x| x.ok_or_else(|| err(format!("table {} column {col:?} has nulls", self.name))))
            .collect()
    }

    pub fn opt_u64(&self, col: &str) -> Result<Vec<Option<u64>>, Diagnostic> {
        self.opt_int(col)
    }

    pub fn u64(&self, col: &str) -> Result<Vec<u64>, Diagnostic> {
        self.int(col)
    }

    pub fn u32(&self, col: &str) -> Result<Vec<u32>, Diagnostic> {
        self.int(col)
    }

    pub fn opt_u32(&self, col: &str) -> Result<Vec<Option<u32>>, Diagnostic> {
        self.opt_int(col)
    }

    pub fn opt_i64(&self, col: &str) -> Result<Vec<Option<i64>>, Diagnostic> {
        self.opt_int(col)
    }

    pub fn i64(&self, col: &str) -> Result<Vec<i64>, Diagnostic> {
        self.need(col)?;
        Ok(self
            .opt_i64(col)?
            .into_iter()
            .map(|x| x.unwrap_or(0))
            .collect())
    }

    pub fn opt_f64(&self, col: &str) -> Result<Vec<Option<f64>>, Diagnostic> {
        match self.columns(col) {
            None => Ok(vec![None; self.rows()]),
            Some(cs) => {
                let mut out = Vec::with_capacity(self.rows());
                for a in cs {
                    match a.data_type() {
                        DataType::Float64 => {
                            out.extend(a.as_primitive::<Float64Type>().iter());
                        }
                        DataType::Float32 => out.extend(
                            a.as_primitive::<Float32Type>()
                                .iter()
                                .map(|x| x.map(f64::from)),
                        ),
                        DataType::Null => out.extend(std::iter::repeat_n(None, a.len())),
                        dt => match ints(&a) {
                            Some(v) => out.extend(v.into_iter().map(|x| x.map(|x| x as f64))),
                            None => return Err(self.bad(col, dt)),
                        },
                    }
                }
                Ok(out)
            }
        }
    }

    pub fn f64(&self, col: &str) -> Result<Vec<f64>, Diagnostic> {
        self.need(col)?;
        Ok(self
            .opt_f64(col)?
            .into_iter()
            .map(|x| x.unwrap_or(f64::NAN))
            .collect())
    }

    pub fn opt_str(&self, col: &str) -> Result<Vec<Option<String>>, Diagnostic> {
        match self.columns(col) {
            None => Ok(vec![None; self.rows()]),
            Some(cs) => {
                let mut out = Vec::with_capacity(self.rows());
                for a in cs {
                    out.extend(strs(&a).ok_or_else(|| self.bad(col, a.data_type()))?);
                }
                Ok(out)
            }
        }
    }

    pub fn str(&self, col: &str) -> Result<Vec<String>, Diagnostic> {
        self.need(col)?;
        Ok(self
            .opt_str(col)?
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect())
    }

    /// Lists of any element type readable by `elem`; a missing column reads as all-null.
    fn lists<T>(
        &self,
        col: &str,
        elem: impl Fn(&ArrayRef) -> Option<Vec<Option<T>>>,
    ) -> Result<Vec<Option<Vec<T>>>, Diagnostic> {
        let Some(cs) = self.columns(col) else {
            return Ok((0..self.rows()).map(|_| None).collect());
        };
        let mut out = Vec::with_capacity(self.rows());
        for a in cs {
            let rows: Vec<Option<ArrayRef>> = match a.data_type() {
                DataType::List(_) => a.as_list::<i32>().iter().collect(),
                DataType::LargeList(_) => a.as_list::<i64>().iter().collect(),
                DataType::FixedSizeList(_, _) => a.as_fixed_size_list().iter().collect(),
                DataType::Null => vec![None; a.len()],
                dt => return Err(self.bad(col, dt)),
            };
            for r in rows {
                out.push(match r {
                    None => None,
                    Some(v) => Some(
                        elem(&v)
                            .ok_or_else(|| self.bad(col, v.data_type()))?
                            .into_iter()
                            .flatten()
                            .collect(),
                    ),
                });
            }
        }
        Ok(out)
    }

    pub fn list_f64(&self, col: &str) -> Result<Vec<Option<Vec<f64>>>, Diagnostic> {
        self.lists(col, |a| match a.data_type() {
            DataType::Float64 => Some(a.as_primitive::<Float64Type>().iter().collect()),
            DataType::Float32 => Some(
                a.as_primitive::<Float32Type>()
                    .iter()
                    .map(|x| x.map(f64::from))
                    .collect(),
            ),
            _ => ints(a).map(|v| v.into_iter().map(|x| x.map(|x| x as f64)).collect()),
        })
    }

    pub fn list_u32(&self, col: &str) -> Result<Vec<Option<Vec<u32>>>, Diagnostic> {
        self.lists(col, ints)?
            .into_iter()
            .map(|l| {
                l.map(|l| l.into_iter().map(|x| self.narrow(col, x)).collect())
                    .transpose()
            })
            .collect()
    }

    pub fn list_str(&self, col: &str) -> Result<Vec<Option<Vec<String>>>, Diagnostic> {
        self.lists(col, strs)
    }

    /// `map<utf8, number>` (or a list of `{key, value}` structs) as sorted pairs.
    pub fn map_str_f64(&self, col: &str) -> Result<Vec<Vec<(String, f64)>>, Diagnostic> {
        let Some(cs) = self.columns(col) else {
            return Ok(vec![Vec::new(); self.rows()]);
        };
        let mut out = Vec::with_capacity(self.rows());
        for a in cs {
            match a.data_type() {
                DataType::Map(_, _) => {
                    let m = a.as_map();
                    let keys = strs(m.keys()).ok_or_else(|| self.bad(col, a.data_type()))?;
                    let vals = Table::new(self.name, &[]).floats(m.values(), col)?;
                    let off = m.value_offsets();
                    for i in 0..m.len() {
                        let (s, e) = (off[i] as usize, off[i + 1] as usize);
                        out.push(if m.is_null(i) {
                            Vec::new()
                        } else {
                            (s..e)
                                .filter_map(|j| Some((keys[j].clone()?, vals[j]?)))
                                .collect()
                        });
                    }
                }
                DataType::Null => out.extend((0..a.len()).map(|_| Vec::new())),
                dt => return Err(self.bad(col, dt)),
            }
        }
        Ok(out)
    }

    fn floats(&self, a: &ArrayRef, col: &str) -> Result<Vec<Option<f64>>, Diagnostic> {
        match a.data_type() {
            DataType::Float64 => Ok(a.as_primitive::<Float64Type>().iter().collect()),
            DataType::Float32 => Ok(a
                .as_primitive::<Float32Type>()
                .iter()
                .map(|x| x.map(f64::from))
                .collect()),
            dt => ints(a)
                .map(|v| v.into_iter().map(|x| x.map(|x| x as f64)).collect())
                .ok_or_else(|| self.bad(col, dt)),
        }
    }

    /// `map<u8, f64>` as pairs.
    pub fn map_u8_f64(&self, col: &str) -> Result<Vec<Vec<(u8, f64)>>, Diagnostic> {
        let Some(cs) = self.columns(col) else {
            return Ok(vec![Vec::new(); self.rows()]);
        };
        let mut out = Vec::with_capacity(self.rows());
        for a in cs {
            let m = match a.data_type() {
                DataType::Map(_, _) => a.as_map(),
                dt => return Err(self.bad(col, dt)),
            };
            let keys = ints(m.keys()).ok_or_else(|| self.bad(col, a.data_type()))?;
            let vals = self.floats(m.values(), col)?;
            let off = m.value_offsets();
            for i in 0..m.len() {
                let (s, e) = (off[i] as usize, off[i + 1] as usize);
                let mut row = Vec::with_capacity(e - s);
                for j in s..e {
                    if let (Some(k), Some(v)) = (keys[j], vals[j]) {
                        row.push((self.narrow(col, k)?, v));
                    }
                }
                out.push(row);
            }
        }
        Ok(out)
    }
}

fn ints(a: &ArrayRef) -> Option<Vec<Option<i128>>> {
    macro_rules! conv {
        ($t:ty) => {
            a.as_primitive::<$t>()
                .iter()
                .map(|x| x.map(i128::from))
                .collect()
        };
    }
    Some(match a.data_type() {
        DataType::Int8 => conv!(Int8Type),
        DataType::Int16 => conv!(Int16Type),
        DataType::Int32 => conv!(Int32Type),
        DataType::Int64 => conv!(Int64Type),
        DataType::UInt8 => conv!(UInt8Type),
        DataType::UInt16 => conv!(UInt16Type),
        DataType::UInt32 => conv!(UInt32Type),
        DataType::UInt64 => conv!(UInt64Type),
        DataType::Null => vec![None; a.len()],
        _ => return None,
    })
}

fn strs(a: &ArrayRef) -> Option<Vec<Option<String>>> {
    Some(match a.data_type() {
        DataType::Utf8 => a
            .as_string::<i32>()
            .iter()
            .map(|x| x.map(String::from))
            .collect(),
        DataType::LargeUtf8 => a
            .as_string::<i64>()
            .iter()
            .map(|x| x.map(String::from))
            .collect(),
        DataType::Utf8View => a
            .as_string_view()
            .iter()
            .map(|x| x.map(String::from))
            .collect(),
        DataType::Dictionary(_, v) if v.as_ref() == &DataType::Utf8 => {
            let d = a.as_any_dictionary();
            let values = strs(d.values())?;
            let keys = ints(&arrow_array::make_array(d.keys().to_data()))?;
            keys.into_iter()
                .map(|k| {
                    k.and_then(|k| usize::try_from(k).ok())
                        .and_then(|k| values.get(k).cloned().flatten())
                })
                .collect()
        }
        DataType::Null => vec![None; a.len()],
        _ => return None,
    })
}

/// The dictionary-encoded utf8 type used for repeated path strings.
pub fn dict_utf8<'a>(v: impl IntoIterator<Item = &'a str>) -> ArrayRef {
    let a: arrow_array::DictionaryArray<UInt32Type> = v.into_iter().collect();
    Arc::new(a)
}

/// One cell as text (CSV export): numbers, strings, dictionary strings, and `;`-joined lists and maps.
pub fn display(a: &dyn Array, row: usize) -> String {
    if a.is_null(row) {
        return String::new();
    }
    macro_rules! prim {
        ($t:ty) => {
            a.as_primitive::<$t>().value(row).to_string()
        };
    }
    match a.data_type() {
        DataType::UInt8 => prim!(UInt8Type),
        DataType::UInt16 => prim!(UInt16Type),
        DataType::UInt32 => prim!(UInt32Type),
        DataType::UInt64 => prim!(UInt64Type),
        DataType::Int8 => prim!(Int8Type),
        DataType::Int16 => prim!(Int16Type),
        DataType::Int32 => prim!(Int32Type),
        DataType::Int64 => prim!(Int64Type),
        DataType::Float32 => prim!(Float32Type),
        DataType::Float64 => prim!(Float64Type),
        DataType::Utf8 => a.as_string::<i32>().value(row).to_string(),
        DataType::LargeUtf8 => a.as_string::<i64>().value(row).to_string(),
        DataType::Dictionary(_, _) => {
            let d = a.as_any_dictionary();
            let k = ints(&arrow_array::make_array(d.keys().to_data())).and_then(|v| v[row]);
            k.and_then(|k| usize::try_from(k).ok())
                .map_or_else(String::new, |k| display(d.values().as_ref(), k))
        }
        DataType::List(_) => {
            let v = a.as_list::<i32>().value(row);
            (0..v.len())
                .map(|i| display(v.as_ref(), i))
                .collect::<Vec<_>>()
                .join(";")
        }
        DataType::Map(_, _) => {
            let m = a.as_map();
            let off = m.value_offsets();
            (off[row] as usize..off[row + 1] as usize)
                .map(|j| {
                    format!(
                        "{}={}",
                        display(m.keys().as_ref(), j),
                        display(m.values().as_ref(), j)
                    )
                })
                .collect::<Vec<_>>()
                .join(";")
        }
        dt => format!("<{dt}>"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> RecordBatch {
        let mut c = Cols::new();
        c.push("a", false, u32s([Some(1), Some(2)]))
            .push("s", true, utf8([Some("x"), None]))
            .push("l", false, list_f64([Some(&[1.0, 2.0][..]), Some(&[][..])]))
            .push("m", false, map_utf8_f64([vec![("k", 1.5)], vec![]]))
            .push("d", false, dict_utf8(["p", "q"]));
        c.batch("t", "1.0")
    }

    #[test]
    fn integer_readers_do_not_wrap() {
        let mut l = ListBuilder::new(arrow_array::builder::UInt64Builder::new());
        l.values().append_value(1 << 32);
        l.append(true);
        l.values().append_value(1);
        l.append(true);
        let mut c = Cols::new();
        c.push("big", false, u64s([Some(1 << 32), Some(1 << 63)]))
            .push("l", false, Arc::new(l.finish()));
        let b = c.batch("t", "1.0");
        let batches = [b];
        let t = Table::new("t", &batches);
        assert!(t.u32("big").is_err());
        assert!(t.opt_u32("big").is_err());
        assert!(t.opt_i64("big").is_err());
        assert!(t.list_u32("l").is_err());
        assert_eq!(t.u64("big").unwrap(), vec![1 << 32, 1 << 63]);
        assert_eq!(t.opt_f64("big").unwrap()[1], Some(2f64.powi(63)));
        assert!(t.int::<u16>("big").is_err());
    }

    #[test]
    fn file_and_concatenated_streams_round_trip() {
        let b = sample();
        let (f, partial) = read_ipc(&to_ipc_file(&b)).unwrap();
        assert!(!partial);
        assert_eq!(f, vec![b.clone()]);
        let mut s = to_ipc_stream(&b);
        s.extend(to_ipc_stream(&b));
        let (batches, partial) = read_ipc(&s).unwrap();
        assert_eq!((batches.len(), partial), (2, false));
        let t = Table::new("t", &batches);
        assert_eq!(t.u32("a").unwrap(), vec![1, 2, 1, 2]);
        assert_eq!(t.opt_str("s").unwrap()[1], None);
        assert_eq!(t.str("d").unwrap(), vec!["p", "q", "p", "q"]);
        assert_eq!(t.map_str_f64("m").unwrap()[0], vec![("k".to_string(), 1.5)]);
        assert_eq!(t.list_f64("l").unwrap()[0], Some(vec![1.0, 2.0]));
        assert_eq!(t.opt_f64("missing").unwrap(), vec![None; 4]);
        s.truncate(s.len() - 20);
        let (batches, partial) = read_ipc(&s).unwrap();
        assert!(partial && batches.len() == 1);
    }
}
