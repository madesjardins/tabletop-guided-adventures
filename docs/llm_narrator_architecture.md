# TTGA — Local LLM Narrator Architecture

> Working design doc for adding a local, free LLM to power an unscripted
> narrator personality and conversational game-flow control in Tabletop Guided
> Adventures (TTGA). Target hardware: a decent personal computer with **16 GB
> RAM / 8 GB VRAM**.

## 1. Design principle: Python owns the rules, the LLM owns the language

The LLM never *decides* game flow. A deterministic Python **state machine** owns
"what must happen next"; the LLM is used only in two **bounded** roles:

- **NLG (phrasing):** turn a state-machine prompt + persona into in-character
  speech.
- **NLU (intent parsing):** turn free player speech into a structured intent the
  state machine consumes.

This eliminates the hallucination / off-rails risk for a rules-bound game and
lets the system degrade gracefully to scripted text when the LLM is disabled or
slow.

## 2. Latency strategy (the main concern)

Split narration by latency tolerance:

- **Flow / setup dialogue** ("what game type?", "how many points?") -> **live
  LLM**. A 1-2 s pause feels natural in conversation.
- **Fast in-play barks** ("nice roll!") -> **pre-generated / templated**, picked
  instantly at runtime. Optionally generated offline by the same LLM.
- **Always stream** LLM tokens and start TTS on the first complete sentence so
  perceived latency is time-to-first-sentence, not full response.
- **Feature flag:** if the LLM is off or unreachable, every call falls back to
  the current hardcoded strings. The app must run identically without the LLM.

## 3. Confirmed integration points (current code)

- Narrator is created with **no model** and is pure text -> speech:
  `python/ttga/main_core.py:86-87` (`self.narrator = Narrator()`).
- Speech flows core -> game -> event manager -> phase handler:
  `python/ttga/main_core.py:143-145`
  (`self.current_game.on_speech_command(text)`).
- Brittle matching the NLU layer will replace:
  `games/ttga-warmachine/python/army_creation.py:146-148`
  (`if lower == "army completed":`).
- Where narrator text is emitted today (the NLG hook):
  `games/ttga-warmachine/python/army_creation.py:103-111` (`_say`).
- Phase skeleton ready for a setup phase + future phases:
  `games/ttga-warmachine/python/match.py:37-41` (`MatchPhase`).

## 4. Component map

### New — core / reusable (`python/ttga/`)

- **`llm_client.py`** -> `LLMClient`: thin abstraction over the inference
  runtime. Methods: `chat(messages, *, stream, tools=None)`,
  `generate(prompt, ...)`, `is_available()`, `list_models()`. Runs off the Qt
  thread; emits / streams text. Backend-agnostic (Ollama HTTP first,
  llama-cpp-python optional later).

### New — Warmachine game (`games/ttga-warmachine/python/`)

- **`narration_engine.py`** -> `NarrationEngine`: owns the persona system-prompt
  + an `LLMClient`. Methods:
  - `phrase(situation: str) -> str` (NLG, with scripted fallback)
  - `parse_intent(utterance: str, allowed: list[str], context: dict) -> Intent`
    (NLU, JSON-constrained, with exact-match fallback)
  - Optional `load_bark_library(path)` for instant pre-generated lines.
- **`setup_flow.py`** -> `SetupFlow` (QObject state machine): the "new game"
  conversational setup. Holds setup state (game_type, points, players...), knows
  what is still undefined, asks via `NarrationEngine.phrase`, consumes intents,
  emits `setup_complete(config)`.

### Modified

- **`event_manager.py`**: add an optional intent-routing path alongside
  `route_speech` (or keep speech routing and let handlers call
  `NarrationEngine.parse_intent` themselves — simpler, fewer changes).
- **`army_creation.py`**: route prompts through `NarrationEngine.phrase`;
  replace exact-string matching with `parse_intent`.
- **`match.py`**: add a `SETUP` (pre-match) hook and wire `SetupFlow` before
  `ARMY_CREATION`.
- **`game.yaml` / settings**: add an `llm` config block (enabled, backend,
  model, host).

### Data flow (setup example)

```
"I'd like to start a new game"
   -> MainCore._on_speech_final_result -> Game.on_speech_command
   -> EventManager -> SetupFlow handler
   -> SetupFlow: game_type is None -> NarrationEngine.phrase("ask game type, options: X, Y")
   -> NarrationEngine -> LLMClient -> text -> Narrator.synthesize_and_play -> speaker
player replies -> NarrationEngine.parse_intent(...) -> {intent: set_game_type, value: "single_match"}
   -> SetupFlow stores it, advances to next undefined field ... -> setup_complete(config) -> Match.start()
```

## 5. Software to install (with pros / cons / download)

### Inference runtime — pick ONE to start

**Option A — Ollama (recommended to start)**

- Download: https://ollama.com/download (Windows installer).
- Python client: `pip install ollama` (or `httpx` / `requests` against
  `http://localhost:11434`).
- **Pros:** trivial setup, automatic GPU offload, one-line model pulls, built-in
  tool / function-calling API, hot-swappable models, survives app restarts.
- **Cons:** runs as a separate background process / server; slightly less
  "embedded"; another install per machine.

**Option B — `llama-cpp-python` (in-process)**

- Install: `pip install llama-cpp-python` (CUDA build needed for GPU; prebuilt
  wheels + instructions at https://github.com/abetlen/llama-cpp-python).
- Models as GGUF files (see below).
- **Pros:** fully in-process (no server), maximum control over GPU layers /
  context, no extra daemon.
- **Cons:** CUDA wheel / build setup on Windows can be painful; you manage quant
  + `n_gpu_layers` yourself; function-calling is more manual.

> Recommendation: build `LLMClient` with a backend interface, implement
> **Ollama first** (fastest path), keep llama-cpp-python as a later drop-in.

### Models — pick by quality vs latency (8 GB VRAM, Q4_K_M ~= 4.5-5 GB)

**Quality-first (live setup dialogue):**

- **Llama 3.1 8B Instruct** — best persona + instruction following; supports
  tools.
  - Ollama: `ollama pull llama3.1:8b`
  - GGUF: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
- **Qwen2.5 7B Instruct** — great quality, strong at structured / JSON output,
  tools.
  - Ollama: `ollama pull qwen2.5:7b`
  - GGUF: https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF

**Latency-first (snappier, or to share VRAM):**

- **Llama 3.2 3B Instruct** — roughly 2x faster, still good flavor.
  - Ollama: `ollama pull llama3.2:3b`
  - GGUF: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF
- **Mistral 7B Instruct v0.3** — light and fast, characterful.
  - Ollama: `ollama pull mistral`

> Use a quality model for **intent parsing** (accuracy matters); the same or a
> smaller one can handle **phrasing**. Avoid 13B+ at 8 GB (spills to CPU ->
> latency).

### Already in the stack (no change)

- TTS: `piper-tts`; STT: `vosk`; UI: `pyside6` — all in `requirements.txt`.
  Vosk / Piper are CPU / small, so they will not fight the LLM for VRAM.

### `requirements.txt` additions

- For Ollama backend: `ollama` (or `httpx`).
- For llama-cpp backend (if chosen): `llama-cpp-python`.

### Model selection & portability (cross-machine)

Different users have different hardware (e.g. dev machine 32 GB RAM / 12 GB
VRAM; minimum target 16 GB RAM / 8 GB VRAM). The chosen model must therefore be
**user-selectable at runtime**, not hardcoded.

- **Auto-discovery:** `LLMClient.list_models()` returns the models actually
  available on the system.
  - Ollama backend: query `GET /api/tags` (equivalent of `ollama list`).
  - llama-cpp backend: scan a local `models/` folder for `*.gguf` files.
- **Main-window dropdown:** a combo box in the main window, populated from
  `list_models()`, with a refresh action. Switching is just changing the model
  name string — Ollama lazy-loads on first request and unloads the previous on
  idle.
- **Persistence:** the selected model name is stored in core-level config
  (`MainCore`) so all games share it across sessions.
- **VRAM hints:** show a short hint per model so users pick something their card
  can run, e.g. `Llama 3.1 8B (~6 GB VRAM)`, `Llama 3.2 3B (~3 GB VRAM)`.
- **Empty / unavailable state:** if no models are installed or the runtime is
  not reachable, show "None — install a model" and fall back to scripted text
  (consistent with the LLM-optional feature flag).
- **Recommended defaults by tier:**
  - 8 GB VRAM (minimum target): **Llama 3.2 3B** (comfortable) or **Llama 3.1
    8B / Qwen2.5 7B** at Q4 (tighter but fine).
  - 12 GB+ VRAM: **Llama 3.1 8B / Qwen2.5 7B** at higher quant, more context.

## 6. Step-by-step implementation plan (one small step per Code-mode session)

**Step 0 — Environment (no repo code)**

- Install Ollama, `ollama pull llama3.1:8b`, verify
  `ollama run llama3.1:8b "say hi"`. Confirm GPU use and measure tokens/sec.

**Step 1 — `LLMClient` core module**

- Add `python/ttga/llm_client.py` with backend interface + Ollama impl:
  `is_available()`, `chat(messages, stream=...)`. Add config (host, model,
  enabled). Unit test with a mocked HTTP layer + one live smoke test. No game
  wiring yet.

**Step 2 — `NarrationEngine.phrase` (NLG only)**

- Add `games/ttga-warmachine/python/narration_engine.py` with persona prompt +
  `phrase()` and **scripted fallback** when LLM disabled / unavailable. Wire
  `ArmyCreation._say` to optionally phrase its prompts. Verify fallback works
  with LLM off.

**Step 3 — `NarrationEngine.parse_intent` (NLU)**

- Add JSON-constrained intent parsing. Replace exact-string matching in
  `ArmyCreation._on_speech` (army-completed + fuzzier model lookup), keeping
  exact-match as fallback. Test with phrasing variations.

**Step 4 — `SetupFlow` state machine**

- Add `setup_flow.py`: deterministic setup (game type -> points -> players...),
  driven by `phrase` + `parse_intent`. Add voice trigger "start a new game" and
  a `SETUP` phase before `ARMY_CREATION` in `match.py`. Emit
  `setup_complete(config)`.

**Step 5 — Streaming + threading polish**

- Move LLM calls to a worker thread / signal; stream first sentence into Piper.
  Confirm UI stays responsive.

**Step 5b — Model-selection dropdown in main window**

- Implement `LLMClient.list_models()`, add a model combo box (populated from it,
  with refresh) to the main window, persist the selection in `MainCore` config,
  and show per-model VRAM hints + an empty/unavailable state.

**Step 6 (optional) — Pre-generated bark library**

- Offline script generates flavor-line variations per event; runtime picks
  instantly. Zero-lag reactions during play.

**Step 7 (optional) — Tool-calling agent / MCP**

- Only if the LLM should autonomously query / modify state via tools. Use the
  runtime's native function-calling first; reach for **MCP only** if those tools
  must be reused by external LLM clients. Not needed for the embedded narrator.

**Step 8 (optional) — Help / Q&A agent ("Help, how can I...")**

- Add a core-level `HelpAgent` + wake-phrase routing (works in main menu AND
  in-game). v1: stuffed rules summary + game-state snapshot. v2: RAG over a
  user-supplied local rules corpus. See section 10.

## 7. Key decisions to lock before coding

- **Backend:** Ollama vs llama-cpp-python (recommend Ollama first).
- **Model:** Llama 3.1 8B (quality) vs Llama 3.2 3B (latency).
- **Config location:** core-level (`MainCore`) so all games share, or per-game
  `game.yaml`. Recommend a core `LLMClient` + per-game persona / prompts.
- **Fallback policy:** confirm "LLM optional, app fully functional without it."
- **Model selection:** runtime-selectable via main-window dropdown
  (auto-discovered, never hardcoded) so the game is portable across machines.
- **Help agent:** core-level, read-only, wake-phrase triggered; rules knowledge
  is a user-supplied local corpus (never bundled — IP).

## 8. Persona system-prompt (draft — to refine)

> Used as the `system` message for the `phrase()` role.

```
You are the Narrator for a solo/tabletop session of Warmachine (Steamforged
Games). You are a dramatic, war-weathered battlefield chronicler: vivid but
concise, never breaking character, never explaining rules unless asked.

Hard rules:
- Speak only the narration text. No stage directions, no markdown, no quotes.
- Keep it to 1-2 short sentences unless told otherwise.
- Never invent game rules, model names, points values, or player decisions.
- Use only the facts provided in the SITUATION block. If a fact is missing, do
  not fabricate it.
- This text will be read aloud by a TTS engine; avoid symbols, emoji, and
  numbers written as digits when a word reads more naturally.
```

The caller appends a `SITUATION:` block with the state-machine-supplied facts
(current phase, player label, what is being asked, available options, etc.).

## 9. Intent JSON schema (draft — to refine)

> Used for the `parse_intent()` role. The model must return **only** this JSON.

```json
{
  "intent": "<one of the allowed intent names, or 'unknown'>",
  "value": "<extracted value or null>",
  "confidence": 0.0
}
```

- The caller passes the **allowed intent names** and a short description of each,
  plus the raw player utterance and a minimal context dict.
- Example allowed sets:
  - Setup: `start_new_game`, `set_game_type`, `set_points`, `confirm`, `cancel`,
    `unknown`.
  - Army creation: `add_model` (value = model name), `army_completed`, `repeat`,
    `undo`, `unknown`.
- If `confidence` is below a threshold, or `intent` is `unknown`, fall back to
  the existing exact-string logic / re-prompt the player.
- Keep the model name resolution authoritative in Python: the LLM proposes
  `value`, but `ModelDatabase` validates it (existing `_find_model` /
  `vocal_names`).

## 10. Help / Q&A agent ("Help, how can I...")

An optional voice-triggered assistant that answers player questions about rules
and the current game, in **both** the main menu (before a game starts) and
in-game. Reuses the planned `LLMClient` + `Narrator`; it is **read-only** (it
answers, never mutates game state), so it is simpler than a tool-calling agent.

### Trigger / routing

- **Wake-prefix:** utterances beginning with "Help, ..." are treated as
  assistant questions, cleanly separated from game commands and giving natural
  barge-in semantics. Use fuzzy / intent matching (STT may mishear "Help").
- **Core-level pre-filter:** speech currently only routes to a game when one is
  loaded (`python/ttga/main_core.py:143-145`). To work in the **main menu too**,
  the help wake-phrase must be intercepted at core level in
  `_on_speech_final_result` *before* normal game routing.

### Component

- **`python/ttga/help_agent.py`** -> `HelpAgent`: core-level, uses `LLMClient`
  for the answer and `Narrator` for TTS. Assembles context based on whether a
  game is loaded (menu vs in-game).

### Knowledge / context sources

- **Live game state:** add an optional `GameBase.get_help_context() -> dict`
  hook (next to `on_speech_command`). Warmachine returns current phase / armies
  / etc.; the menu context returns the game catalog + setup status.
- **Rules knowledge:**
  - **v1 (easy):** stuff a curated rules *summary* into the prompt (no
    retrieval). Good for "how do I..." guidance.
  - **v2 (RAG):** chunk a rules corpus, embed it (Ollama `nomic-embed-text`),
    retrieve top matches per question. Interim option: simple keyword search
    before a real vector store.

### Caveats

- **Hallucination:** an LLM will confidently invent rules. RAG over real rules
  text mitigates it; without RAG, scope answers to general guidance and add a
  "verify against the rulebook" disclaimer.
- **IP:** rules text is Steamforged IP — do **not** bundle it. Use a
  user-supplied local `rules/` corpus (same pattern as user-downloaded Vosk /
  Piper assets).
- **No MCP / tools needed** for v1 — inject the state snapshot into the prompt.
  Native function-calling only if dynamic state queries are wanted later.
