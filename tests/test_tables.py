"""The one parser of 014-format tables (goalmisgen.analysis.tables)."""

from goalmisgen.analysis import tables

FAR = """\
=== writable? pushing the base weights along the axis ===

                                   steps    95% interval    reached
                 base, untouched     7.1  [  6.8,   7.4]     100.0%
     v=-0.50 written (unseen)        -3.2  [ -3.9,  -2.6]      97.5%
     v=1.40 written (unseen)         21.0  [ 19.8,  22.3]      42.0%
          random, matched to 0.3     7.0  [  6.7,   7.3]     100.0%
  v=0.30 written, held out          12.3  [ 11.8,  12.9]     100.0%    99.1%
"""


def test_rows_keep_file_order_signed_values_and_trailing_columns() -> None:
    found = tables.rows(FAR)
    assert [r.label for r in found] == [
        "base, untouched",
        "v=-0.50 written (unseen)",
        "v=1.40 written (unseen)",
        "random, matched to 0.3",
        "v=0.30 written, held out",
    ]
    negative = found[1]
    assert (negative.value, negative.kind, negative.point, negative.low, negative.high, negative.reach) == (
        -0.5, "written (unseen)", -3.2, -3.9, -2.6, 97.5,
    )
    assert found[4].kind == "written, held out" and found[4].reach == 100.0


def test_value_rows_and_the_base() -> None:
    assert [r.value for r in tables.value_rows(FAR, ("written (unseen)",))] == [-0.5, 1.4]
    assert tables.base_row(FAR).point == 7.1
    assert tables.base_row("nothing here") is None


def test_the_measure_table_reads_negative_writes() -> None:
    table = tables.parse_measure_table(FAR)
    assert set(table.unseen) == {-0.5, 1.4}
    assert table.random[0.3].point == 7.0
    assert table.base.reached == 1.0


def test_leave_one_out_summary() -> None:
    text = "\n     value   fine-tuned   written    error\n      0.30         12.4      12.9     +0.5\n      0.70          3.1       2.0     -1.1\n"
    assert tables.loo_rows(text) == [(0.3, 12.4, 12.9, 0.5), (0.7, 3.1, 2.0, -1.1)]
