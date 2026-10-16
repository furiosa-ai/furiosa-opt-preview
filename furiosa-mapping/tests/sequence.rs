//! Tests for `sequence`.

use furiosa_mapping::*;
use furiosa_mapping_macro::{axes, m};

axes![Quarter = 4, Row = 128, Width = 16];

/// The first 64 of each quarter's 128 rows are live, so the lower half of the buffer holds its
/// live cells in two runs of 64 * 16, one quarter of 128 * 16 cells apart.
#[test]
fn test_a_padded_half_reads_only_its_live_rows() {
    let padded = <m![Quarter, Row = 64 # 128, Width]>::to_value();
    let half = padded.clone().modulo(padded.size() / 2);

    let configs = sequence(
        &[&half],
        &[&<m![Quarter % 2, Row = 64, Width]>::to_value()],
        SequencerMode::Read,
    )
    .unwrap();

    assert_eq!(
        configs[0].iter().collect::<Vec<_>>(),
        (0..1024).chain(2048..3072).collect::<Vec<_>>()
    );
}
