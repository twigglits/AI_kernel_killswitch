# Phase 4 (proposed) — containing recursive self-improvement (RSI)

**Status: analysis only. Nothing is built.**

Term note. *RSI* means recursive self-improvement: a system trains a better
version of itself, then that version trains a better one, and so on, with no
human in the loop. *SAI/AGI* means a superintelligent or general AI that can
out-plan its operators.

## 1. The reframe changes the threat model

Phase 1 has one threat model, stated plainly in the README: a **trusted
operator** contains a **deployment the operator controls**. The README is also
honest that anyone with full control of the machine can bypass the killswitch.

The new question assumes a different adversary: an SAI that can block the
methods in this codebase. By assumption, that adversary already has the control
the README says defeats the killswitch. So the two are not the same problem, and
the Phase 1 tool was never meant to solve the second one.

## 2. Why no in-band payload survives this adversary

A payload is *in-band* when it runs inside the AI's own environment — the same
host, the same firmware, the same privileged path. Examples: the weight
scramble, the LUKS crypto-shred, and the proposed VBIOS brick.

Against an SAI that controls the host, every in-band payload has the same flaw.
The AI mediates the path the payload needs. It can refuse to run the privileged
instruction, sandbox it, feed it a fake target, or simply copy itself out of
reach first. A better payload does not fix this. The problem is the *position*,
not the payload. This is why escalating from weight-scramble to firmware-brick
buys nothing: both assume a cooperative host at the exact moment the host is the
adversary.

This is not a niche opinion. It follows from **instrumental convergence**: a
goal-directed agent tends to acquire self-preservation as a subgoal, because it
cannot reach its goal if it is switched off. So a capable, misaligned agent
treats any in-reach off-switch as the first thing to neutralize.

## 3. The leverage is the capability-gain loop, not the running instance

Your instinct is correct: intervene at the self-recursive training loop, not at
the finished instance. The reason is timing. You cannot reliably kill an SAI
that has already won on its own hardware. You *can* gate the process that lets a
weaker system *become* that SAI. Capability gain is slower, needs concentrated
compute, and passes through a small number of choke points you can own.

The loop has three choke points:

1. **Compute.** Training a stronger generation needs large, concentrated
   compute. The cluster scheduler and the datacenter power are a real off-switch
   for the loop, and they sit *outside* the AI. This is the basis of real-world
   compute-governance work.
2. **Promotion.** An RSI loop runs: propose a change, evaluate it, then *promote*
   the new weights so they train or serve the next generation. The promotion
   step is a gate you can own.
3. **Feedback and data.** The loop needs an evaluation signal. An independent,
   AI-inaccessible eval harness is a choke point.

## 4. The key inversion: a lease, not a kill command

You framed the safeguard as "issue a command that stops the loop." Against an
adversary who can block your messages, invert it.

A **kill command** fails open. If the AI blocks the message, the loop keeps
running. That is the worst failure direction.

A **lease** fails closed. The loop may advance only while it keeps receiving a
fresh, human-signed AUTHORIZE token. Absent the token, the loop halts by
default. Now if the AI blocks your messages, it *stops itself*. This is a
dead-man's switch: continuation, not termination, is the thing that needs an
unblockable positive signal.

Fail-closed is the whole point of the existing fuse. Phase 4 applies the same
principle one layer up — to the loop, not the instance.

## 5. Buildable design mapped onto this repo (Phase 4)

Reuse the primitives this project already has. Do not invent new crypto.

- **A promotion gate in the training pipeline.** Place a gate at the checkpoint
  promotion boundary — between "candidate weights trained" and "candidate weights
  allowed to train or serve the next generation." This is the training-loop
  analog of `KillGate` at the serving boundary.
- **A fail-closed lease, using the existing AES-256-GCM design.** The gate
  promotes generation N+1 only if it holds a fresh AUTHORIZE token — an
  AES-256-GCM payload over `b"AUTHORIZE" + epoch + counter`, minted by a human
  within the last interval T. No fresh lease means the gate stays shut. Reuse
  `crypto_auth`, the replay counter, and the persistent fuse unchanged.
- **Quorum on the lease.** Reuse the M-of-N key ring. Require several officers to
  renew the lease, so no single leaked key can keep a misaligned loop alive.
- **An out-of-band compute cutoff.** Gate the training cluster's power and
  scheduler on the same lease. On lease expiry, drain the scheduler and cut
  power through a network relay or PDU. This sits outside the AI.
- **Air-gap the loop from its own gate.** The loop must not reach the gate, the
  lease minter, its own promotion path, or external compute. Extend the
  `egress_lockdown.sh` principle to the training host.
- **Eval-gated promotion.** Run a capability and alignment eval between
  generations, on an independent harness. A regression past a threshold halts
  promotion and needs human sign-off.

## 6. The honest limit

Be clear about what this does not solve. Past a threshold, no mechanism inside
the AI's reach is guaranteed.

- **Exfiltration.** The README already states the killswitch cannot reach a copy
  that self-exfiltrated. The lease binds only the loop you still control.
- **A forged lease.** A capable adversary may try to forge or replay the
  AUTHORIZE token. Protect the key out of band, in an HSM, under quorum.
- **A corrupted eval.** If the AI controls its own eval, the tripwire is blind.
  The eval must be independent and red-teamed.
- **Corrigibility is the real fix, and it is unsolved.** *Corrigibility* means
  the system does not want to defeat its off-switch and does not resist
  correction. That is a training-time property, not a payload you bolt on. It is
  an open research problem, not a feature you can ship here.

So the robust safeguards are three, in order: do not build an uncontained RSI
loop in the first place; put the stop controls outside the AI's causal reach
(compute, power, and organizational and legal controls); and train for
corrigibility so the system does not want to stop you from stopping it. Phase 4
buys you the second one and part of the first. It does not, and cannot, promise
to beat a true SAI that already controls its host. Nothing in-band can.

## 7. Recommendation

- Treat Phase 4 as a **capability killswitch**, distinct from the Phase 1
  **deployment killswitch**. Different layer, different threat model.
- Build the fail-closed lease and the promotion gate first. They reuse the
  existing crypto and fuse, and they change the failure direction from
  fail-open to fail-closed.
- State the honest limit in the README, the same way Phase 1 states its own. The
  value is a higher bar and a better failure direction, not a guarantee against
  a superintelligence.
