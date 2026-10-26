# furiosa-opt-bootloader

The signed code a PE runs from reset. The chip's secure boot runs only signed code, so the
bootloader is the one signed part of the runtime: it authenticates the firmware image the host
staged in device memory, copies it into the scratchpad and enters it. Before that it serves
nothing but the boot request; `furiosa_opt_abi::bootloader` holds the contract with the host.

The bootloader authenticates exactly one firmware image, because its signed bytes carry that
image's SHA-256 digest. A firmware change therefore needs a rebuilt and re-signed bootloader, even
though the two images stay separate.

## Build

`make furiosa-opt-vendor SIGN=dev` (or `SIGN=release`, with the release credentials) builds the
firmware, binds its digest into the bootloader through `FURIOSA_OPT_FIRMWARE`, then signs and
vendors both images.
`make furiosa-opt-bootloader-test` runs the host tests.

## Signing

The device accepts a bootloader only with a signature from a key it trusts, so the runtime
embeds `furiosa-opt-bootloader.signed`, never the bare `bootloader.bin`. The signer lives in
`sign/`; the public distribution carries the signed image but not the signer, its certificates or
the release-key configuration.
