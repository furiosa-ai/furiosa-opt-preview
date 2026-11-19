//! PE placement checks for scratchpad scalar accesses.

use furiosa_mapping::{Mapping, MappingExt};

/// Checks that only position 0 is live, allowing trailing padding.
pub(super) fn only_position_zero_is_live(placement: &Mapping) -> bool {
    // Padding trails live cells, so removing it exposes the PE 0 identity.
    placement.clone().normalize().remove_outermost_padding() == Mapping::identity()
}

/// A PE mapping incompatible with scratchpad scalar access.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("scratchpad scalar access requires only PE 0 live, got {pe}")]
pub struct SpmScalarPeError {
    pub pe: Mapping,
}

/// Validates that scalar accesses can read and write PE 0 alone.
pub fn validate_spm_scalar_pe(pe: &Mapping) -> Result<(), SpmScalarPeError> {
    if only_position_zero_is_live(pe) {
        return Ok(());
    }
    Err(SpmScalarPeError { pe: pe.clone() })
}

#[cfg(test)]
mod tests {
    use furiosa_mapping::*;

    use super::*;

    axes![K = 4, CL = 2];

    #[test]
    fn accepts_one_live_pe_under_any_padding_kind() {
        validate_spm_scalar_pe(&<m![1]>::to_value()).unwrap();
        validate_spm_scalar_pe(&<m![1 # 4]>::to_value()).unwrap();
        validate_spm_scalar_pe(&<m![1 #{0} 4]>::to_value()).unwrap();
        validate_spm_scalar_pe(&<m![1 #{!} 4]>::to_value()).unwrap();
        validate_spm_scalar_pe(&<m![1 # 4, 1 # 2]>::to_value()).unwrap();
    }

    #[test]
    fn rejects_a_placement_that_spans_more_than_pe_zero() {
        for pe in [
            <m![4]>::to_value(),
            <m![K % 4]>::to_value(),
            <m![CL]>::to_value(),
            Mapping::Broadcast { size: 4 },
        ] {
            assert_eq!(validate_spm_scalar_pe(&pe), Err(SpmScalarPeError { pe: pe.clone() }));
        }
    }
}
