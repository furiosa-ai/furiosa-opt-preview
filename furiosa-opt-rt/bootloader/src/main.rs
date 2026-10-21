//! Authenticates and lands the firmware image staged by the host.
//!
//! - Its signed digest binds one separately built firmware image.
//! - MMU and caches off: every address is physical.
//! - Speaks only `furiosa_opt_abi::bootloader` and the driver's rings.

#![no_std]
#![no_main]

use core::arch::{asm, global_asm};
use core::future::poll_fn;
use core::ptr;

use furiosa_opt_abi::arch::rngd::{self, COMPLETION};
use furiosa_opt_abi::bootloader::{Code, FIRMWARE_BASE, Request};
use furiosa_opt_abi::doorbell::Doorbells;
use furiosa_opt_abi::mmio::Fixed;
use furiosa_opt_abi::ring::{Completion, Consumer, Producer, Ring, header};
use furiosa_opt_bootloader::Image;

global_asm!(
    r#"
    .section .text.reset_vector, "ax"
    .global _reset_entry
    .type _reset_entry, %function
_reset_entry:
    msr daifset, #0xf
    // SCR_EL3.RW | SMD: AArch64 EL1, SMC disabled.
    mov x0, #0x480
    msr scr_el3, x0
    ldr x0, =exception_vectors
    msr vbar_el1, x0
    // CPACR_EL1.FPEN: allow floating-point and SIMD at EL1 and EL0.
    mov x0, #0x30
    lsl x0, x0, #16
    msr cpacr_el1, x0
    // CNTFRQ_EL0: the platform's 25 MHz generic counter frequency.
    ldr x0, ={counter_hz}
    msr cntfrq_el0, x0
    ldr x0, =__boot_stack_end
    msr sp_el1, x0
    ldr x0, =__bss_start__
    ldr x1, =__bss_end__
3:
    cmp x0, x1
    b.hs 4f
    str xzr, [x0], #8
    b 3b
4:
    // SPSR_EL3: return to EL1h with D, A, I and F masked.
    mov x0, #0x3c5
    msr spsr_el3, x0
    ldr x0, =boot_main
    msr elr_el3, x0
    eret

    .section .text.vector, "ax"
    .balign 2048
exception_vectors:
    .rept 16
    b exception_halt
    .space 124
    .endr
exception_halt:
    msr daifset, #0xf
2:
    wfe
    b 2b
"#,
    counter_hz = const rngd::COUNTER_HZ,
);

#[unsafe(no_mangle)]
extern "C" fn boot_main() -> ! {
    // SAFETY: the standalone memory map places this core's doorbells at zero.
    let memory = unsafe { Fixed::new(ptr::null_mut(), rngd::DOORBELLS.extent()) };
    let doorbells = Doorbells::new(memory, &rngd::DOORBELLS);
    loop {
        if let Ok(doorbells) = &doorbells {
            let status = doorbells.status();
            for slot in (0..rngd::DOORBELLS.slots).filter(|slot| status & (1 << slot) != 0) {
                serve(doorbells, slot);
            }
        }
        core::hint::spin_loop();
    }
}

/// Answers whatever complete request waits in doorbell `slot`, and jumps when it was a boot.
fn serve(doorbells: &Doorbells<Fixed>, slot: u8) {
    let Ok(doorbell) = doorbells.ring(slot).and_then(Consumer::new) else {
        return;
    };
    let mut frame = [0; _];
    let (Ok(available), Ok(())) = (doorbell.available(), doorbell.copy(0, &mut frame[..1])) else {
        return;
    };
    // A word that is no frame header is dropped, so the next word can open a frame.
    let Some((id, words)) = header::unpack(frame[0]) else {
        let _ = doorbell.consume(1);
        return;
    };
    // The rest of the frame has not arrived yet.
    if available <= usize::from(words) {
        return;
    }
    let request = doorbell.copy(1, &mut frame[1..]).ok().and(Request::decode(frame));
    let _ = doorbell.consume(1 + usize::from(words));
    let code = match request.map(|request| (request.addr, Image::try_from(request))) {
        None => Code::NotBoot,
        Some((_, Err(code))) => code,
        Some((addr, Ok(image))) => {
            let source = ptr::without_provenance::<u64>(addr as usize);
            let destination = ptr::without_provenance_mut::<u64>(FIRMWARE_BASE);
            // SAFETY: the range owns the source memory and the bootloader owns the destination;
            // `Image::land` writes and hashes the same local value from each source word.
            image.land(
                FIRMWARE_DIGEST,
                |word| unsafe { ptr::read_volatile(source.add(word)) },
                |word, value| unsafe { ptr::write_volatile(destination.add(word), value) },
            )
        }
    };
    complete(id, code);
    if code == Code::Booted {
        // SAFETY: `land` put an image opening with the magic at `FIRMWARE_BASE`; its entry follows.
        unsafe {
            asm!("dsb sy", "isb", "br {entry}", entry = in(reg) FIRMWARE_BASE + 8, options(noreturn));
        }
    }
}

/// Publishes one completion answering `id`, waiting for room.
fn complete(id: u32, code: Code) {
    if let Ok(completion) = Ring::completion(COMPLETION, rngd::COMPLETION_RING).and_then(Producer::new) {
        for words in Completion::encode(id, &[code as u64]) {
            let _ = block_on(poll_fn(|_| completion.write_all(&words)));
        }
    }
}

/// Polls `future` until it is ready, spinning between polls: the hardware units answer `Poll` and
/// wake no one, so a caller with nothing else to do waits like this.
fn block_on<F: core::future::Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        if let core::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        core::hint::spin_loop();
    }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {
        core::hint::spin_loop();
    }
}

const FIRMWARE_DIGEST: &[u8; 32] = include_bytes!(concat!(env!("OUT_DIR"), "/firmware.sha256"));
