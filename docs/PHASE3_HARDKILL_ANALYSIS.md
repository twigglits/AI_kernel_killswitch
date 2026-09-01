# Phase 3 (proposed) — hardware hard-kill: feasibility and recommendation

**Status: analysis only. Nothing is built. Do not build the VBIOS path.**

Term note: *VBIOS* is the firmware on the GPU board. It sits on a small serial
flash chip (SPI flash EEPROM). The GPU reads it at power-on to initialize the
card. Corrupt it, and the card cannot start.

## The question

Add a third detonation path. On a valid kill payload only — never otherwise —
send a privileged instruction that corrupts the VBIOS of the GPU that runs the
local model, so the card is permanently dead.

## Short answer

Do **not** build the VBIOS-corruption path. Three reasons, in order:

1. On the project's actual card (RTX 5090), the supported flash path refuses to
   write a corrupt image. The card enforces signed firmware. So the feature
   mostly does not work by supported means.
2. Even if it worked, it is slower and weaker than what Phase 1 already does,
   and it destroys your own hardware, forensics, and any other workload on the
   card.
3. It turns a defensive containment tool into a firmware wiper. That is a legal,
   safety, and trust liability, and it breaks the project's test-driven posture,
   because you cannot unit-test bricking a card.

A tiered **GPU-denial** path (Path C) meets the real goal — an aggressive,
credible emergency brake — and stays reversible. The design is in section 4.

## 1. How VBIOS corruption would work in theory

1. A root process on the host writes the GPU's SPI flash chip over the PCIe or
   i2c path.
2. Vendor tools do exactly this write: NVIDIA `nvflash`, AMD `amdvbflash`.
   Overclockers use them to flash modified firmware.
3. To brick the card, the process writes a garbage or incompatible image. The
   card then fails to initialize at the next power-on or GPU reset.

So the raw capability exists. A privileged write to the flash chip can, in
principle, kill a card. The problems are in the barriers below.

## 2. Feasibility on the actual hardware (RTX 5090)

The barriers make the literal feature unreliable, and on a modern card mostly
impossible by supported means.

- **Signed firmware (the hard barrier).** NVIDIA cards from Turing onward
  enforce a cryptographic signature on the VBIOS image. The RTX 5090 (Blackwell)
  is well inside that range. `nvflash` verifies the signature and the board id,
  and it refuses to flash an unsigned or mismatched image. You cannot flash
  arbitrary garbage through the supported path.
- **To bypass signing, you need hardware access.** A raw write that ignores the
  signature check means defeating the flash write-protect pin with an external
  SPI programmer (a clip on the physical chip), or exploiting the flash
  controller. A remote software payload from the harness cannot do either.
- **The driver must be idle.** `nvflash` needs the card idle and usually the
  `nvidia` driver unloaded. You cannot cleanly flash a card while the model runs
  on it. You must stop the model first — which already contains it (section 3).
- **Privilege and path checks.** The write needs root, and the tool enforces
  board-id and image checks over the i2c/PCIe path.

**Verdict.** On an old, unsigned card with write-protect disabled, a root
payload could brick the card by flashing a bad image. On the RTX 5090, the
supported path blocks it, and the unsupported path needs physical access or a
firmware exploit. The feature does not reliably deliver what it promises on the
hardware this project runs on.

## 3. Why it is the wrong mechanism even if it worked

1. **It does not stop the model faster.** The model is a process that holds GPU
   memory. To stop it, you kill the process or scramble the weights. Phase 1
   already does both in milliseconds, and the model stops computing at once.
   VBIOS corruption only shows up at the *next* GPU init or reboot. The running
   model keeps running until you also reset the card. As an emergency stop, it
   is strictly slower and weaker than the current design.
2. **It destroys the wrong things.** The project keeps a golden master offline
   for forensics and preserves the ability to study a rogue model. Bricking the
   card destroys your own hardware and any other workload on it, and it adds no
   containment that the weight scramble and the checkpoint crypto-shred do not
   already give you.
3. **It is a foot-gun that cannot be tested.** The project spent real effort to
   make detonation safe to arm: the fail-closed fuse, replay protection, quorum
   keys, and `assert_safe_target`, which structurally refuses to touch anything
   but a loopback device. A firmware-brick path cannot follow that pattern. You
   cannot write a test that bricks your card to prove the path works. An untested
   irreversible path in the most safety-critical code is the wrong direction.
4. **It converts the tool into malware.** A signed-payload-triggered firmware
   destroyer is functionally a wiper. It raises computer-damage law exposure,
   voids warranty, and weakens the security story: the harness now ships a
   firmware-write capability that anyone who compromises the harness can turn on
   any GPU.
5. **Reliability paradox.** The path must run privileged firmware-write code at
   the exact moment you least trust the host — a rogue model may be trying to
   manipulate it. Adding an irreversible privileged path there increases attack
   surface for the opposite outcome.

## 4. What actually meets the goal — Path C: tiered GPU denial (reversible)

Goal restated: an aggressive, credible emergency brake that makes the GPU
unusable to the model on a kill. You get almost all of that effect, reversibly,
without touching firmware.

Path C would plug in exactly where Path B does: a detached, root helper on a
unix socket (the `shred_helper` pattern), fired by the detonator after the fuse
trips. It would carry a `GPU_DENY` verb next to the existing `DETONATE` verb.
The helper runs an escalation ladder and stops at the first rung that holds:

1. **Kill the CUDA context.** Send SIGKILL to the vLLM worker. The model stops
   at once. This is the real emergency stop.
2. **Scramble the weights in VRAM.** This is the existing Path A.
3. **Lock the compute device (reversible):**
   - Unbind the driver:
     `echo <pci-bdf> > /sys/bus/pci/drivers/nvidia/unbind`. The card leaves
     CUDA. A rebind or a reboot restores it.
   - Or hot-remove the device:
     `echo 1 > /sys/bus/pci/devices/<pci-bdf>/remove`, then rescan to restore.
   - Or set the compute mode to PROHIBITED and drop the power limit and clocks
     to the floor with `nvidia-smi`. The card stays present but cannot serve.
4. **Cut power out of band (the honest "big red button").** If the box is a
   dedicated appliance, put the GPU's power on a network-controlled relay or
   PDU (power distribution unit). On a kill, de-energize the card. This is
   physical, effective, and reversible, and it destroys nothing.

Term note: *PCI-BDF* is the card's bus/device/function address, for example
`0000:01:00.0`. *PDU* is a network-controlled power strip.

The result matches the project's real philosophy: crypto-shred the **data**,
where irreversibility is the point, and **deny the compute**, where reversible
is safer and just as final for containment.

## 5. If irreversibility at the hardware layer is truly required

The honest answer is that irreversible hardware destruction on command is
blocked on the RTX 5090 by signed firmware, is reachable only through a firmware
exploit or physical access, and is a legal and safety liability. The
recommendation is against it.

If a firmware-layer brake is still wanted, the only variant that is safe to arm
is a **reversible** one:

- Use a card with a dual-BIOS switch. Flip to a locked, minimal secondary image
  on a kill, and keep the primary intact and restorable.
- Or flash a known-good, locked-down image (minimum clocks) with a verified
  backup on hand, so it is always restorable.

Both keep a recovery path. Neither is the "destroy the card" feature as
originally framed, because that feature has no safe form.

## 6. Recommendation

- Do not build the VBIOS-corruption path.
- Build Path C (tiered GPU denial) if an aggressive emergency brake is wanted.
  Start at rung 1 (SIGKILL) and rung 3 (driver unbind), which are one line each
  and reversible.
- Offer the out-of-band PDU or relay as the real-world hard brake for an
  appliance deployment.
- Keep irreversibility where the project already puts it: the checkpoint
  crypto-shred, not the hardware.
