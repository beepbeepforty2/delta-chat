# Recording script — 2–4 minute walkthrough

Shot list for the assignment's required walkthrough. Every timing below is
**measured on this machine**, not estimated. Total live runtime ≈ **2m 20s**;
with `--idle-time-limit` the played-back cast lands around **2m**.

The assignment asks for three things — one delta, one grounded chat exchange,
and the eval scorecard. Shots 2, 4 and 5–6 are those three. Everything else is
optional if you want to come in under time.

---

## Before you hit record

asciinema isn't installed on this machine yet:

```bash
brew install asciinema      # checked: `which asciinema` currently returns nothing
```

```bash
cd ~/Code/delta-chat
git status --short          # should be clean
make dataset                # ~30s, ONE TIME — do NOT record this
make test                   # confirm 393 green — do NOT record this
rm -rf reports/ traces/     # so the demo writes them fresh on camera
```

Check `.env` has a working `LLM_AUTH_TOKEN` — shot 4 makes a live call and a
credential failure mid-recording is the one thing that wastes a take.

Set your terminal to **100×30** and a legible font. asciinema records the real
terminal size; anything wider makes the scorecard wrap on playback.

```bash
asciinema rec delta-chat-demo.cast \
  --idle-time-limit 2 \
  --title "delta-chat — document delta & grounded chat"
```

`--idle-time-limit 2` caps any pause longer than 2s at 2s. This is what makes
the 75-second scorecard watchable **without faking anything** — the output is
real and complete, only dead waiting time is compressed.

---

## Shot 1 — what this is *(~10s, mostly talking)*

```bash
cat DEMO.md | head -20
```

Say: *"Two revisions of a P&ID in, a structured delta out, plus chat grounded
in both documents and that delta. The pair I'm using is real vendor content —
hand-edited, not synthetic."*

## Shot 2 — the delta *(measured 0.96s)*

```bash
make run A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
cat reports/delta_report.md
```

Point at **one** line — the `[HIGH]` one:

> `[HIGH] Sheet 1, zone F-8 (confidence 0.02): line_tag pipe_class changed: AC21 -> AC31`

Say: *"Eleven changes; exactly one ranked HIGH, and it's the right one — a pipe
class change is a mechanical rating change. The note reword and the DELETED
placeholder are correctly LOW. It's ranking, not counting."*

If you have 5 spare seconds, the confidence `0.02` is the most interesting
number on screen: *"low because several line tags on this sheet look alike, so
the match was contested — the engine is telling you it's unsure about the
alignment, not about the change."*

## Shot 3 — markup *(measured 0.81s, optional)*

```bash
make markup A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
```

Say: *"Real PDF annotation objects — they show up in Acrobat's markup list, not
a flattened image."* **Cut here and show a screenshot** of `reports/markup_b.pdf`
open in a viewer; asciinema can't show the PDF.

## Shot 4 — grounded chat *(measured 3.5s, live LLM)*

```bash
uv run python -m src.cli chat \
  --a data/samples/real_pair/a/L0.pdf \
  --b data/samples/real_pair/b/L0.pdf \
  --question "What changed in the pipe class for any line tags, and does it matter?"
```

**This is the most important 20 seconds of the demo.** The answer cites
`[delta:1:F-8:delta0009]` — the exact record from shot 2 — and then *declines*
the "does it matter?" half.

Say: *"It answered what the sources support and refused the part they don't,
instead of producing plausible engineering commentary. That's the whole design:
citations are validated after generation, and an uncited claim gets overridden
into a refusal."*

Optionally add a second question to show a clean refusal:

```bash
  --question "What is the recommended lubricant viscosity for the compressor bearings?"
```

## Shot 5 — the scorecard *(measured 75s → ~4s on playback)*

```bash
make eval
```

⚠️ Bare `make eval` runs the **full** scorecard including live chat and the
baseline — that's **~8.5 minutes**. For the recording use the deterministic
half:

```bash
uv run python -m eval.run_eval --dataset eval/datasets/v0 --skip-chat --skip-baseline
```

Talk over the wait; `--idle-time-limit` compresses it. When it lands, point at
the numbers that are **bad**:

Say: *"L0 F1 0.84. But look at `remove P=0.00` — every removal it reports on
these pairs is a false positive. And L2, the scanned path, is 0.13 precision.
The scorecard is built to show me that, not to flatter the system. Zero false
positives on all three null pairs, and the sibling drawing is correctly refused
rather than diffed."*

## Shot 6 — the held-out set *(measured 2.5s)* — **the strongest shot**

```bash
make eval-holdout
```

Say: *"This is a real EPA P&ID the system has never been tuned against. On my
own synthetic set the raster layer measured a 0.0 recall lift — looked like
dead weight. On real data it's 1.0, catching both valve-symbol swaps the text
engine structurally cannot see, at 1.0 residue precision. And the null control
emits 0 spurious regions where the synthetic one emits 61. Both numbers
reversed the moment I used real data — which is exactly why the held-out set
exists, and why I report it separately."*

## Shot 7 — observability *(instant, optional)*

```bash
make trace ID=$(ls -t traces/*.json | head -1 | xargs basename | sed 's/.json//')
```

Say: *"Every request traces ingest → delta → retrieval → LLM, per-stage
timings, tokens and cost on the LLM span. Failures are recorded on the span,
not swallowed."*

---

## Finishing

```
Ctrl-D                       # stops recording
asciinema upload delta-chat-demo.cast
```

Or keep it local and commit the `.cast` — it's a few KB of plain JSON, and
`asciinema play delta-chat-demo.cast` replays it offline with no upload.

Put the resulting link (or file path) at the top of `DEMO.md`, which already
contains all of this output in written form for anyone who'd rather read it.

---

## Option B — hand it to a computer-use agent

macOS can record a fixed-length video with no interactive stop:

```bash
screencapture -v -V 210 -k ~/Desktop/delta-chat-demo.mov
```

`-V 210` stops itself after 210s, `-k` shows clicks. **Grant Screen Recording
permission to the app that will spawn it (Terminal / Claude) first** — macOS
prompts once and silently records a black screen if denied.

The advantage over asciinema: a real video *can* show the annotated PDF and
`report.html`, which is where the two visual payoffs live.

The one scheduling problem: the deterministic scorecard takes 75s and a video
has no idle compression, so it would be 75 seconds of a blinking cursor. The
script below starts it in the background first and fills that window with the
visual artifacts, then prints its output when it's ready.

**Paste this prompt into a Claude session with computer control on the Mac:**

> Record a ~3.5 minute screen-recorded demo of the delta-chat project. Work in
> `~/Code/delta-chat`. Do all of this yourself; don't ask me to confirm steps.
>
> **Preflight (do NOT record this):** open Terminal, resize it to roughly
> 100×32, `cd ~/Code/delta-chat`, then run `make dataset` if
> `eval/datasets/v0/pairs` is missing, and `rm -rf reports traces`. Confirm
> `.env` exists (shot 4 makes a live LLM call). Confirm `screencapture` has
> Screen Recording permission — if a permission dialog appears, grant it and
> start over.
>
> **Then write this to `/tmp/demo_run.sh`, `chmod +x` it, and run it.** It is
> timed to fit inside the recording; do not add commands or change the sleeps:
>
> ```bash
> #!/bin/bash
> cd ~/Code/delta-chat
> clear
> # the slow scorecard runs in the background so its 75s is not dead screen time
> (uv run python -m eval.run_eval --dataset eval/datasets/v0 \
>    --skip-chat --skip-baseline > /tmp/eval_v0.txt 2>&1) &
>
> echo "=== delta-chat: two P&ID revisions -> structured delta + grounded chat ==="
> echo "=== pair: real vendor content, hand-edited (data/samples/real_pair) ==="
> sleep 4
>
> echo; echo "--- 1. compute the delta ---"; sleep 1
> make run A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
> sleep 2; cat reports/delta_report.md; sleep 14
>
> echo; echo "--- 2. annotated PDFs (real annotation objects, not flattened) ---"
> make markup A=data/samples/real_pair/a/L0.pdf B=data/samples/real_pair/b/L0.pdf
> sleep 1; open reports/markup_b.pdf; sleep 12
>
> echo; echo "--- 3. grounded chat: cites the delta, refuses what it cannot support ---"
> uv run python -m src.cli chat --a data/samples/real_pair/a/L0.pdf \
>   --b data/samples/real_pair/b/L0.pdf \
>   --question "What changed in the pipe class for any line tags, and does it matter?" \
>   --question "What is the recommended lubricant viscosity for the compressor bearings?"
> sleep 4
>
> echo; echo "--- 4. eval scorecard (seeded set) ---"
> # `wait` blocks ~17s here on a warm machine -- the chat output stays on
> # screen through it, which is why the sleep above is 4 and not 16.
> wait; cat /tmp/eval_v0.txt; sleep 18
>
> echo; echo "--- 5. HELD-OUT real EPA P&ID -- never tuned against ---"; sleep 1
> make eval-holdout; sleep 16
>
> echo; echo "--- 6. per-request trace: ingest -> delta -> retrieval -> LLM ---"
> make trace ID=$(ls -t traces/*.json | head -1 | xargs basename | sed 's/.json//')
> sleep 10
> echo; echo "=== DEMO.md has all of this in written form ==="
> ```
>
> **To record:** bring Terminal to the front, then start
> `screencapture -v -V 210 -k ~/Desktop/delta-chat-demo.mov` and immediately
> run `/tmp/demo_run.sh` in the visible Terminal window. When Preview opens the
> annotated PDF at step 2, bring Terminal back to the front after ~10s so the
> rest stays visible. The recording stops itself at 210s.
>
> **Afterwards:** tell me the file path and its duration, and flag anything
> that scrolled off-screen or looked wrong — do not re-record without asking.

**Measured, not estimated:** the commands total **79.8s** wall (dominated by
the background scorecard); with the sleeps the recording lands at **≈2m05s**,
leaving ~35s of headroom inside `-V 210` for window switching. If you want to
be closer to the brief's 4-minute ceiling, lengthen the `sleep` after shots 4
and 5 — those are the two screens worth dwelling on.

If Preview stealing focus turns out to be fiddly, drop the `open` line and
take a still of the annotated PDF separately; everything else is
terminal-only and will record cleanly unattended.

## Time budget

| Shot | Measured | Required by the brief |
|---|---|---|
| 1 intro | ~10s | — |
| 2 delta | 1s + ~25s talking | ✅ one delta |
| 3 markup | 1s + screenshot | bonus |
| 4 chat | 3.5s + ~25s talking | ✅ one grounded exchange |
| 5 scorecard | 75s live → ~4s played | ✅ scorecard output |
| 6 holdout | 2.5s + ~30s talking | — |
| 7 trace | instant + ~10s | — |

Drop shots 3 and 7 first if you need to cut. Never drop 2, 4 or 5 — those are
the three the assignment names explicitly.
