Goal: I want to LEARN to build this from scratch without AI assistance.
Build the full project AND produce structured learning material that teaches me
every concept, design decision, and tool used — so I can recreate it myself.

────────────────────────────────────────────────────────────────────────────────
PHASE 1 — PLAN (do this first, stop and wait for my approval)
────────────────────────────────────────────────────────────────────────────────

Write a PLAN.md file containing:

1. PROJECT OVERVIEW
   One paragraph: what the project does and what problem it solves.

2. TECHNOLOGY STACK TABLE
   | Layer | Tool/Library | Why this choice over alternatives |
   For every dependency, explain why it was chosen (not just what it is).

3. MODULE BREAKDOWN
   Split the project into 5–10 self-contained modules. For each module:
   - Module number and name
   - Which source files it covers
   - What the learner will be able to do after completing it
   - What concepts it teaches
   Keep modules small enough that a learner can complete one in 30–60 minutes.

4. FINAL FILE STRUCTURE
   Full directory tree with a one-line comment on every file.

5. KEY DESIGN DECISIONS TABLE
   | Decision | Why | What you would do differently without this constraint |
   Include at least: data flow between components, error handling strategy,
   configuration approach, and extensibility points.

6. OPEN QUESTIONS FOR THE USER
   List any choices the user should make before building starts
   (UI type, external services, target OS, etc.)

After writing PLAN.md, stop. Do not write any code yet.
Ask me: "Review PLAN.md and reply 'approved' when ready to build."

────────────────────────────────────────────────────────────────────────────────
PHASE 2 — BUILD (only after I say "approved")
────────────────────────────────────────────────────────────────────────────────

Build EVERYTHING in this order:

STEP A — All source code files
  - Write complete, working, production-quality code
  - Use industry best practices for the chosen stack
  - Follow these rules:
      * No comments explaining WHAT the code does (names do that)
      * Comments only for non-obvious WHY (hidden constraints, workarounds)
      * No placeholder / TODO sections — every function must be fully implemented
      * All tool I/O must use typed Pydantic models where applicable
      * Configuration must live in a single config file loaded from environment variables
      * Design so the most likely change (swapping an API, adding a feature) requires
        editing the fewest possible files
  - Include: requirements.txt (or equivalent), a sample/example input file,
    a .env.example showing all required variables with comments

STEP B — Tests
  For each major module, write a test file:
  - Unit tests: test pure logic with no external dependencies (no API, no network)
  - Integration tests: marked with a skip decorator, require a real environment variable
    like INTEGRATION=1 to run
  - Name tests descriptively: test_<what>_<condition>_<expected>

STEP C — Learning material folder: learning_material/
  Create one markdown file per module: module_00_name.md, module_01_name.md, …

  Each module file MUST contain ALL of these sections:

  ## What You Are Building
  One paragraph. What does this module produce and why does it matter?

  ## Concept Deep-Dives
  For every non-trivial concept used in this module:
    - Name the concept and give a one-sentence definition
    - Show a minimal standalone code example (not from the project) that demonstrates it
    - Show how the project uses it
    - Explain what would break if this concept were removed or done naively
  Do not assume knowledge. Explain decorators, async, closures, etc. from first principles
  if the project uses them.

  ## Reading the Source File(s)
  Walk through the actual source file(s) for this module section by section.
  For each section: quote the code, then explain every non-obvious line.
  Highlight design patterns used and name them (e.g. "this is the Factory pattern").

  ## Why This Design (not just what it does)
  Explain each major design decision in this module:
    - What alternatives were considered
    - What the trade-off is
    - When you would make a different choice

  ## Running the Tests
  Show the exact command to run just this module's tests.
  Explain what each test is checking and why that matters.

  ## Checkpoint ✓
  One small runnable snippet (Python REPL or shell command) that proves
  this module works in isolation. Should take < 30 seconds to run.

  ## Exercises (3 minimum)
  Three exercises, ordered easy → hard:
    1. A modification that requires understanding one concept from this module
    2. A small extension that adds new behaviour
    3. A challenge that requires combining two or more concepts

  ## Resources
  For EVERY library, tool, or concept used in this module, list:
    - Official documentation link
    - Best tutorial / guide (not the official docs — a practical guide)
    - If it is a computer science concept: a plain-English explanation link
  Minimum 3 resources per module.

  ## What's Next
  One sentence linking to the next module file.

STEP D — Master worksheet: WORKSHEET.md
  The WORKSHEET.md is the top-level guide that ties everything together.
  It must contain:

  1. PROJECT SUMMARY (3–5 sentences)

  2. ARCHITECTURE DIAGRAM
     ASCII diagram showing how every component connects.
     Show data flow with arrows. Label every arrow with what data flows through it.

  3. FILE MAP TABLE
     | File | Module # | What it contains |

  4. PREREQUISITE CHECKLIST
     A checkbox list of software/accounts/tools needed before starting.
     Each item links to where to download or sign up.

  5. PER-MODULE SECTION (one per module, in order)
     For each module:
       - Title and estimated time
       - Link to learning_material/module_XX_name.md
       - Link to the source file(s)
       - Numbered build steps (what to write / run in order)
       - "Key concepts" bullet list
       - Common mistakes to avoid (2–3 gotchas specific to this module)

  6. END-TO-END WALKTHROUGH
     A concrete example showing the full system working from input to output.
     Use the sample files provided in examples/.

  7. HOW TO EXTEND THIS PROJECT
     A section with at least 5 concrete extension ideas, each with:
       - What it adds
       - Which files to change
       - Which new concepts to learn (with a resource link)

  8. TROUBLESHOOTING TABLE
     | Symptom | Most likely cause | Fix |
     At least 8 rows covering the most common failure modes.

  9. LEARNING ROADMAP
     After this project, what should the learner study next?
     Give 3 progressively harder follow-on projects with a one-line description each.

────────────────────────────────────────────────────────────────────────────────
QUALITY RULES (apply to everything)
────────────────────────────────────────────────────────────────────────────────

Code quality:
  - Every public function must have type annotations
  - No hardcoded secrets, paths, or magic numbers in source files
  - All user-facing errors must include actionable messages ("Install X with: pip install X")
  - The project must run with a single command after setup

Learning material quality:
  - Never say "as you can see" or "simply" — explain as if talking to an intelligent peer
    who is new to this specific technology
  - Every code snippet must be runnable as written
  - All resource links must be to official docs, well-known tutorials, or reputable references
    (no random blog posts without explaining why it is recommended)
  - Exercises must have a clear success condition (the learner knows when they got it right)

Linking rules:
  - Every module file must link to its source file(s) at the top
  - Every source file should have a one-line module docstring that references the module guide
  - WORKSHEET.md must link to every module file and every source file at least once

────────────────────────────────────────────────────────────────────────────────
TECHNOLOGY LEARNING RESOURCES (include these regardless of the stack)
────────────────────────────────────────────────────────────────────────────────

In WORKSHEET.md, add a "Before You Start" section with resources for any technology
the user may not know. For each technology used, include:

  | Technology | What to learn | Resource | Time to learn basics |
  |---|---|---|---|

Use these categories:
  - Language fundamentals (if using Python, JS, etc.)
  - Framework / library (e.g. LangChain, FastAPI, React)
  - Infrastructure (e.g. Docker, databases, cloud services)
  - Domain knowledge (e.g. "how LLMs work", "what a REST API is")

For each, recommend the MINIMUM learning needed to follow the worksheet —
not a complete curriculum, just enough to not be lost.

────────────────────────────────────────────────────────────────────────────────
