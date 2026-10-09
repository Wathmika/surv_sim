# Step 1 — Survival Sim

A single creature moves around a 2D world, senses the nearest food, and
steers toward it. Energy drains every frame and refills when food is eaten.
No AI yet — the steering is hard-coded. That's intentional: this is the
foundation the rest of the predator-prey/neuroevolution project gets built
on top of.

This is step 1 of a larger learning roadmap (see **Where this fits**
below).

## Project structure

```
step1_survival_sim/
├── main.py             # the entire simulation (world, creature, food, game loop)
├── requirements.txt    # Python dependencies (just pygame)
└── README.md           # this file
```

Yes — at this stage it's genuinely just the one file. That's deliberate:
everything (the `Food` class, the `Creature` class, and the game loop) is
small enough to live in one place and be read top to bottom in a few
minutes. As later steps add a neural network, a genetic algorithm, and a
second species, the project will naturally split into more files
(`brain.py`, `evolution.py`, `predator.py`, etc.) — no need to over-engineer
the structure before there's a reason to.

## Requirements

- **Python 3.9+**
- **pygame** (the only dependency — listed in `requirements.txt`)

## Setup in VS Code

1. **Install the Python extension** (if you don't have it): open the
   Extensions panel (`Ctrl+Shift+X` / `Cmd+Shift+X`) and install
   "Python" by Microsoft.

2. **Open the folder**: `File → Open Folder...` and select
   `step1_survival_sim/`.

3. **Create a virtual environment** (keeps this project's packages
   separate from the rest of your system):

   Open a terminal in VS Code (`` Ctrl+` ``) and run:

   ```bash
   python -m venv venv
   ```

   Activate it:

   - **macOS/Linux:** `source venv/bin/activate`
   - **Windows (PowerShell):** `venv\Scripts\Activate.ps1`

   VS Code will usually prompt you to select this venv as the workspace
   interpreter — click "Yes." If it doesn't, open the Command Palette
   (`Ctrl+Shift+P`), run **Python: Select Interpreter**, and pick the
   one inside `venv/`.

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run it**:

   ```bash
   python main.py
   ```

   Or press `F5` in VS Code (select "Python File" if prompted) to run
   with the debugger attached — useful once you start adding neural
   network code and want to inspect values mid-run.

6. **Quit**: close the window, or press `Esc` while it's focused.

## What to try first

Open `main.py` and look at the config block near the top — these are
meant to be tweaked:

| Constant | Try this |
|---|---|
| `ENERGY_DRAIN_PER_FRAME` | Raise it a lot (e.g. `1.0`) — watch how much harder survival gets |
| `CREATURE_TURN_SPEED` | Lower it — the creature can't turn fast enough to track food and starts missing it |
| `FOOD_COUNT` | Drop it to `3` — see how scarcity changes behavior |
| `CREATURE_SPEED` | Set it near `0` — confirm the creature just... doesn't survive |

Change one at a time, run, observe, then change it back before trying the
next. The terminal prints a line every time the creature dies, showing
how long it survived and how much it ate — watch that number as you tweak
things.

## Where this fits

This is stage 1 of a 6-stage roadmap toward a predator-prey neuroevolution
simulation:

1. **Survival sim** (this file) — world, creature, food, hard-coded steering
2. Replace hard-coded steering with a small neural network
3. Define a fitness function (what counts as "success")
4. Add a population + genetic algorithm (this is where evolution happens)
5. Add a second species (predators) for co-evolution
6. Visualize fitness over generations and analyze emergent behavior

Each stage is meant to be a complete, runnable milestone before moving to
the next — don't skip ahead to neural networks before this stage feels
solid.
