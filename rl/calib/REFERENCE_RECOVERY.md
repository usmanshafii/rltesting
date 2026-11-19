# Restored baseline-test input

The full repository regression found two environmental failures after the RL tests passed:
a missing git-ignored ASTM CSV and a C:/D: temporary-overlay path mismatch.

The original NREL workbook was recovered from
https://raw.githubusercontent.com/JuliusMuschaweck/JMO_Spectrum/master/AM0AM1_5.xls
because the historical PVEducation host returned HTTP 403. Its SHA-256 exactly matches
the workbook registered in docs/sources/README.md:
076cc3d48b701bcc45fe72051afc1846f8ea57cb15a568ffaa10f67eda951b77.
The mirror supplies identical document bytes; it is not used as a new numerical authority.

The registered workbook was restored to docs/sources/astm/NREL_AM0AM1_5_ASTM_G173_E490.xls.
The Spectra sheet's first four columns (all 2,004 rows, two headers plus 2,002 numeric
rows) were exported to docs/sources/astm/ASTM_G173-03_from_NREL_workbook.csv, retaining
full Python float precision and LF line endings. Every exported numeric cell was
compared exactly to the verified workbook. See reference_recovery.json.

The restored CSV's serialization hash differs from the historical CSV export hash in
the register. It is a fresh value-for-value export from the identical registered
workbook; no claim of byte identity with the old CSV is made. The historical source
register and the physics implementation remain unchanged. Both third-party files
remain git-ignored under the existing repository rules.

The extraction command is retained below for the AI-use/source-reading record.
xlrd was used only as an isolated analysis tool, not added to the application or the
pinned RL runtime. This is not a new project runtime dependency.

~~~python
import csv, io
import xlrd
book = xlrd.open_workbook(file_contents=verified_workbook_bytes)
sheet = book.sheet_by_name("Spectra")
rows = [sheet.row_values(i)[:4] for i in range(sheet.nrows)]
out = io.StringIO(newline="")
csv.writer(out, lineterminator="\n").writerows(rows)
data = out.getvalue().encode("utf-8")
parsed = list(csv.reader(io.StringIO(data.decode())))[2:]
assert len(parsed) == 2002
assert all(float(v) == original for row, rr in zip(parsed, rows[2:])
           for v, original in zip(row, rr))
~~~

The overlay test passes when pytest's --basetemp is on D:, the same drive as the
repository. No change was made to render/sampling.py or to its test.
