# Raw inputs (not in git)

`MANIFEST.csv` lists every raw input: NYC Open Data dataset ID, landing page,
download date, number format and SHA-256 of the file used.

You do **not** need these files to rebuild the records: the Manhattan subset of
each input, in its original formatting, is deposited with the Data Records as
`source_snapshot/` (created by `pipeline/a_build_records/00_make_source_snapshot.py`)
and `run_all.py records` uses it by default.

`number_format = it` means the file was exported from the NYC Open Data web
interface with Italian locale formatting ('.' thousands separator, ',' decimal
separator). The parser detects the format and fails loudly on ambiguous input.
NYC Open Data updates datasets in place: a new download will differ slightly
from the snapshot.
