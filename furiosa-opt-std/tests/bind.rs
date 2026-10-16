use std::alloc::System;
use std::marker::PhantomData;

use furiosa_opt_std::prelude::*;
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static ALLOCATOR: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

// One test, since the allocation count spans every thread.
fn without_allocations<T>(run: impl FnOnce() -> T) -> T {
    let region = Region::new(ALLOCATOR);
    let result = run();
    let change = region.change();
    assert_eq!(change.allocations + change.reallocations, 0, "binding allocated");
    result
}

/// One buffer of a distinct size, or none.
struct Value(Option<HbmBuffer>);

impl DeviceSend for Value {
    fn arguments(&self, access: Access) -> Result<impl Iterator<Item = (HbmBufferSlice<'_>, Access)>, Error> {
        Ok(std::iter::once((
            self.0.as_ref().ok_or(Error::Unplaced)?.slice(.., ..)?,
            access,
        )))
    }
}

fn value(device: &Device, size: usize) -> Value {
    Value(Some(HbmBuffer::alloc(device, size).unwrap()))
}

#[derive(DeviceSend)]
struct Entries<Iterator, const N: usize>([Iterator; N]);

#[derive(DeviceSend)]
struct State<'a, Result, const N: usize> {
    first: &'a Result,
    entries: Entries<Result, N>,
    last: (Result, ()),
    marker: PhantomData<&'a ()>,
}

/// Derived bindings yield arguments in declaration order, without allocating.
#[test]
fn binding_preserves_order_without_allocating() {
    let device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
    let first = value(&device, 1);
    let mut state = State {
        first: &first,
        entries: Entries(std::array::from_fn::<_, 1024, _>(|index| value(&device, index + 2))),
        last: (value(&device, 1026), ()),
        marker: PhantomData,
    };
    let taken = std::mem::replace(&mut state.entries.0[500], Value(None));
    assert!(matches!(state.arguments(Access::Read), Err(Error::Unplaced)));
    state.entries.0[500] = taken;
    let bound = without_allocations(|| {
        let arguments = state.arguments(Access::Write).unwrap().enumerate();
        arguments
            .filter(|(index, (memory, _))| memory.size() == index + 1)
            .count()
    });
    assert_eq!(bound, 1026);
}
