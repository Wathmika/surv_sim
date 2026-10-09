"""
Step 1: Single-agent survival sim (no AI yet)
-----------------------------------------------
A creature moves around a 2D world, senses the nearest food, and moves
toward it. Energy drains over time and refills when food is eaten.
If energy hits zero, the creature dies and the sim resets.

This is intentionally "dumb" (hard-coded behavior, no neural net, no
evolution) — the goal of this stage is just to get the simulation loop,
world, and visuals working correctly before any AI is added.

Run it with:  python main.py
Quit with:    close the window, or press ESC
"""

import math
import random
import pygame

# ---------------------------------------------------------------------------
# Config — tweak these and see what happens. That's the point of this stage.
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 900, 600
FPS = 60

CREATURE_RADIUS = 10
CREATURE_SPEED = 2.5            # pixels per frame
CREATURE_TURN_SPEED = 0.08      # radians per frame (how fast it can steer)

FOOD_RADIUS = 5
FOOD_COUNT = 10
FOOD_ENERGY = 20                # energy gained per food eaten
EAT_DISTANCE = CREATURE_RADIUS + FOOD_RADIUS

MAX_ENERGY = 100
ENERGY_DRAIN_PER_FRAME = 0.3    # try raising this — life gets harder fast

BG_COLOR = (18, 18, 24)
CREATURE_COLOR = (100, 90, 25)
FOOD_COLOR = (95, 20, 120)
TEXT_COLOR = (230, 230, 230)


class Food:
    def __init__(self):
        self.respawn()

    def respawn(self):
        self.x = random.uniform(FOOD_RADIUS, WIDTH - FOOD_RADIUS)
        self.y = random.uniform(FOOD_RADIUS, HEIGHT - FOOD_RADIUS)
        # self.x = random.uniform(0, WIDTH)
        # self.y = random.uniform(0, HEIGHT)

MAX_DISTANCE = math.hypot(WIDTH, HEIGHT)  # screen diagonal = farthest two points can be

N_INPUTS = 3
N_HIDDEN = 5   # small enough to understand/evolve, big enough for a non-linear rule
N_OUTPUTS = 1

# Random starting weights are multiplied by this. At 1.0 the sums feeding tanh get
# large, so many brains start with the output pinned near -1/+1 (always turning hard
# one way) and look alike. Smaller values keep more brains in tanh's responsive range.
INIT_WEIGHT_SCALE = 0.5


class Brain:
    """A tiny feedforward neural network: 3 inputs -> 5 hidden -> 1 output.

    Each layer does:  output = activation( weights @ inputs + biases )
      - a weight says "how much does this input matter to this neuron?"
      - a bias shifts the neuron's threshold, independent of the inputs
      - the activation (tanh) squashes the result into [-1, 1]. Without it,
        stacking layers would collapse into one big linear function and the
        network could not represent anything interesting.
    """

    def __init__(self):
        # weights_ih[h][i] = connection strength from input i to hidden neuron h
        r = INIT_WEIGHT_SCALE
        self.weights_ih = [[random.uniform(-r, r) for _ in range(N_INPUTS)]
                           for _ in range(N_HIDDEN)]
        self.bias_h = [random.uniform(-r, r) for _ in range(N_HIDDEN)]
        # weights_ho[o][h] = connection strength from hidden neuron h to output o
        self.weights_ho = [[random.uniform(-r, r) for _ in range(N_HIDDEN)]
                           for _ in range(N_OUTPUTS)]
        self.bias_o = [random.uniform(-r, r) for _ in range(N_OUTPUTS)]

        # The most recent values seen by each layer. forward() overwrites these
        # every call, so the visualization panel can read them to show what the
        # brain is "thinking" right now. They start at zero until the first call.
        self.last_inputs = [0.0] * N_INPUTS
        self.last_hidden = [0.0] * N_HIDDEN
        self.last_outputs = [0.0] * N_OUTPUTS

    def genes(self):
        """All weights and biases as one flat list (the creature's 'DNA')."""
        flat = []
        for row in self.weights_ih:
            flat += row
        flat += self.bias_h
        for row in self.weights_ho:
            flat += row
        flat += self.bias_o
        return flat

    def set_genes(self, flat):
        """Load a flat list produced by genes() back into the weights and biases."""
        k = 0
        for h in range(N_HIDDEN):
            for i in range(N_INPUTS):
                self.weights_ih[h][i] = flat[k]
                k += 1
        for h in range(N_HIDDEN):
            self.bias_h[h] = flat[k]
            k += 1
        for o in range(N_OUTPUTS):
            for h in range(N_HIDDEN):
                self.weights_ho[o][h] = flat[k]
                k += 1
        for o in range(N_OUTPUTS):
            self.bias_o[o] = flat[k]
            k += 1

    def forward(self, inputs):
        """Run the network. `inputs` is a list of N_INPUTS numbers; returns a
        list of N_OUTPUTS numbers, each in [-1, 1]."""
        # Hidden layer: for each hidden neuron, take a weighted sum of all
        # inputs, add its bias, then squash with tanh.
        hidden = []
        for h in range(N_HIDDEN):
            total = self.bias_h[h]
            for i in range(N_INPUTS):
                total += self.weights_ih[h][i] * inputs[i]
            hidden.append(math.tanh(total))

        # Output layer: same recipe, but the "inputs" are the hidden activations.
        outputs = []
        for o in range(N_OUTPUTS):
            total = self.bias_o[o]
            for h in range(N_HIDDEN):
                total += self.weights_ho[o][h] * hidden[h]
            outputs.append(math.tanh(total))

        # Remember this pass's values for the visualization (no effect on the sim).
        self.last_inputs = list(inputs)
        self.last_hidden = hidden
        self.last_outputs = outputs
        return outputs


class Creature:
    def __init__(self, brain=None):
        self.brain = brain if brain is not None else Brain()
        self.x = WIDTH / 2
        self.y = HEIGHT / 2
        self.angle = random.uniform(0, 2 * math.pi)
        self.energy = MAX_ENERGY
        self.alive = True
        self.food_eaten = 0
        self.age = 0  # frames survived — this will become our "fitness" later

    def fitness(self):
        """Score for how well this creature did: survival time plus a big bonus per food."""
        return self.age + self.food_eaten * 100

    def nearest_food(self, foods):
        """Return the closest Food object (and its distance)."""
        closest = None
        closest_dist = float("inf")
        for f in foods:
            d = math.hypot(f.x - self.x, f.y - self.y)
            if d < closest_dist:
                closest_dist = d
                closest = f
        return closest, closest_dist

    def update(self, foods):
        if not self.alive:
            return

        self.age += 1

        # --- "Sense" the world: find nearest food -----------------------
        target, dist = self.nearest_food(foods)

        # --- "Decide": the neural network chooses how much to turn ------
        if target is not None:
            desired_angle = math.atan2(target.y - self.y, target.x - self.x)
            # signed angle to food relative to where we're facing, wrapped to [-pi, pi]
            angle_diff = (desired_angle - self.angle + math.pi) % (2 * math.pi) - math.pi

            inputs = [
                angle_diff,                # already in [-pi, pi], a sensible scale
                dist / MAX_DISTANCE,       # normalized to ~[0, 1]
                self.energy / MAX_ENERGY,  # normalized to [0, 1]
            ]
            # forward() returns a value in [-1, 1]: -1 = full left, +1 = full right.
            # Scaling by CREATURE_TURN_SPEED keeps the turn rate physically limited.
            output = self.brain.forward(inputs)[0]
            self.angle += output * CREATURE_TURN_SPEED

        # --- "Act": move forward in the current facing direction --------
        self.x += math.cos(self.angle) * CREATURE_SPEED
        self.y += math.sin(self.angle) * CREATURE_SPEED

        # keep inside the world (simple wall bounce)
        if self.x < CREATURE_RADIUS or self.x > WIDTH - CREATURE_RADIUS:
            self.angle = math.pi - self.angle
            self.x = max(CREATURE_RADIUS, min(WIDTH - CREATURE_RADIUS, self.x))
        if self.y < CREATURE_RADIUS or self.y > HEIGHT - CREATURE_RADIUS:
            self.angle = -self.angle
            self.y = max(CREATURE_RADIUS, min(HEIGHT - CREATURE_RADIUS, self.y))

        # --- Energy: drain over time, refill by eating -------------------
        self.energy -= ENERGY_DRAIN_PER_FRAME
        if target is not None and dist < EAT_DISTANCE:
            self.energy = min(MAX_ENERGY, self.energy + FOOD_ENERGY)
            self.food_eaten += 1
            target.respawn()

        if self.energy <= 0:
            self.alive = False

    def draw(self, surface, color=CREATURE_COLOR):
        pygame.draw.circle(surface, color, (int(self.x), int(self.y)), CREATURE_RADIUS)
        # draw a little "nose" so you can see which way it's facing
        nose_x = self.x + math.cos(self.angle) * CREATURE_RADIUS
        nose_y = self.y + math.sin(self.angle) * CREATURE_RADIUS
        pygame.draw.circle(surface, (255, 255, 255), (int(nose_x), int(nose_y)), 2)


# ---------------------------------------------------------------------------
# Evolution (stage 4)
# ---------------------------------------------------------------------------
POP_SIZE = 80
ELITE_COUNT = 4            # top brains copied unchanged into the next generation
TOURNAMENT_SIZE = 5        # parents are the fittest of this many random creatures
MUTATION_RATE = 0.1        # chance that any single weight gets nudged
MUTATION_STRENGTH = 0.3    # std-dev of the nudge
MAX_GENERATION_FRAMES = 3600   # safety cap so a generation of "immortal" creatures still ends
LEADER_COLOR = (80, 255, 120)


def tournament_select(creatures):
    """Pick TOURNAMENT_SIZE random creatures and return the fittest of them.
    Fitter creatures win more often, but weak ones still have a small chance."""
    contenders = random.sample(creatures, TOURNAMENT_SIZE)
    return max(contenders, key=lambda c: c.fitness())


def crossover(genes_a, genes_b):
    """Child gets each weight independently from parent A or parent B (coin flip)."""
    return [a if random.random() < 0.5 else b for a, b in zip(genes_a, genes_b)]


def mutate(genes):
    """With small probability per weight, add a small random nudge."""
    return [g + random.gauss(0, MUTATION_STRENGTH) if random.random() < MUTATION_RATE else g
            for g in genes]


class Population:
    """Many creatures sharing one world and one set of food, plus the
    generation-to-generation evolution step."""

    def __init__(self, size=POP_SIZE):
        self.size = size
        self.foods = [Food() for _ in range(FOOD_COUNT)]
        self.creatures = [Creature() for _ in range(size)]
        self.generation = 1
        self.frames = 0
        self.best_ever = 0
        self.history = []   # one (generation, best, avg) tuple per finished generation

    def update(self):
        """Advance every living creature one frame."""
        for c in self.creatures:
            c.update(self.foods)
        self.frames += 1
        if self.frames >= MAX_GENERATION_FRAMES:
            for c in self.creatures:
                c.alive = False

    def alive_count(self):
        return sum(c.alive for c in self.creatures)

    def all_dead(self):
        return self.alive_count() == 0

    def leader(self):
        """Fittest living creature (or fittest overall if everyone is dead)."""
        pool = [c for c in self.creatures if c.alive] or self.creatures
        return max(pool, key=lambda c: c.fitness())

    def evolve(self):
        """Score the finished generation, print stats, and build the next one."""
        ranked = sorted(self.creatures, key=lambda c: c.fitness(), reverse=True)
        best = ranked[0].fitness()
        avg = sum(c.fitness() for c in ranked) / len(ranked)
        self.best_ever = max(self.best_ever, best)
        self.history.append((self.generation, best, avg))
        print(f"Gen {self.generation:3d} | best {best:5d} | avg {avg:7.1f} | best ever {self.best_ever:5d}")

        # Elitism: copy the best brains over untouched so we can never get worse.
        new_creatures = []
        for c in ranked[:ELITE_COUNT]:
            brain = Brain()
            brain.set_genes(c.brain.genes())
            new_creatures.append(Creature(brain))

        # Fill the rest with children of tournament-selected parents.
        while len(new_creatures) < self.size:
            parent_a = tournament_select(ranked)
            parent_b = tournament_select(ranked)
            child_genes = mutate(crossover(parent_a.brain.genes(), parent_b.brain.genes()))
            brain = Brain()
            brain.set_genes(child_genes)
            new_creatures.append(Creature(brain))

        self.creatures = new_creatures
        for f in self.foods:
            f.respawn()
        self.generation += 1
        self.frames = 0


PANEL_WIDTH = 420
PANEL_BG = (28, 28, 38)
POS_COLOR = (60, 140, 255)   # positive weight = blue
NEG_COLOR = (255, 70, 70)    # negative weight = red
INPUT_LABELS = ["angle to food", "food distance", "energy"]


def activation_color(level):
    """Map a value in [-1, 1] to a gray: dark = -1, mid-gray = 0, bright = +1."""
    level = max(-1.0, min(1.0, level))
    v = int((level + 1) / 2 * 255)
    return (v, v, v)


def draw_brain(surface, brain, font):
    """Draw the network in the right-hand panel, using the brain's stored
    weights (for the lines) and last_* activations (for the neuron fills)."""
    pygame.draw.rect(surface, PANEL_BG, (WIDTH, 0, PANEL_WIDTH, HEIGHT))
    title = font.render("Brain", True, TEXT_COLOR)
    surface.blit(title, (WIDTH + 15, 10))

    # Raw inputs have different ranges, so rescale each to [-1, 1] for coloring only:
    #   angle is in [-pi, pi] -> divide by pi;  distance/energy are in [0, 1] -> 2v - 1.
    ins = brain.last_inputs
    input_levels = [ins[0] / math.pi, ins[1] * 2 - 1, ins[2] * 2 - 1]

    def column(n, x):
        """Evenly spaced (x, y) positions for n neurons in one vertical column."""
        top, bottom = 80, HEIGHT - 60
        return [(x, int(top + (bottom - top) * (k + 0.5) / n)) for k in range(n)]

    in_pos = column(N_INPUTS, WIDTH + 140)
    hid_pos = column(N_HIDDEN, WIDTH + 280)
    out_pos = column(N_OUTPUTS, WIDTH + 380)

    def draw_connections(src, dst, weights):
        # weights[d][s] connects source neuron s to destination neuron d
        for d, dp in enumerate(dst):
            for s, sp in enumerate(src):
                w = weights[d][s]
                color = POS_COLOR if w >= 0 else NEG_COLOR
                pygame.draw.line(surface, color, sp, dp, max(1, int(abs(w) * 5)))

    # Lines first so the neuron circles are drawn on top of them.
    draw_connections(in_pos, hid_pos, brain.weights_ih)
    draw_connections(hid_pos, out_pos, brain.weights_ho)

    def draw_neurons(positions, levels, values):
        for (x, y), level, value in zip(positions, levels, values):
            pygame.draw.circle(surface, activation_color(level), (x, y), 16)
            pygame.draw.circle(surface, (200, 200, 200), (x, y), 16, 1)
            num = font.render(f"{value:+.2f}", True, TEXT_COLOR)
            surface.blit(num, (x - num.get_width() // 2, y + 20))

    draw_neurons(in_pos, input_levels, ins)
    draw_neurons(hid_pos, brain.last_hidden, brain.last_hidden)
    draw_neurons(out_pos, brain.last_outputs, brain.last_outputs)

    # Labels: inputs on the left of their circles, output above its circle.
    for (x, y), text in zip(in_pos, INPUT_LABELS):
        label = font.render(text, True, TEXT_COLOR)
        surface.blit(label, (x - 24 - label.get_width(), y - label.get_height() // 2))
    label = font.render("turn", True, TEXT_COLOR)
    surface.blit(label, (out_pos[0][0] - label.get_width() // 2, out_pos[0][1] - 40))


GRAPH_HEIGHT = 200
BEST_COLOR = (255, 210, 60)   # gold
AVG_COLOR = (70, 220, 230)    # cyan
AXIS_COLOR = (120, 120, 140)


def draw_fitness_graph(surface, history, font):
    """Line graph of best and average fitness per generation, in a strip along the
    bottom of the window. Redraws purely from `history`; nothing is recomputed."""
    area = pygame.Rect(0, HEIGHT, WIDTH + PANEL_WIDTH, GRAPH_HEIGHT)
    pygame.draw.rect(surface, PANEL_BG, area)

    # Plot rectangle inside the strip (margins leave room for labels and legend).
    left, right = area.left + 70, area.right - 20
    top, bottom = area.top + 30, area.bottom - 28
    pygame.draw.line(surface, AXIS_COLOR, (left, top), (left, bottom))      # y axis
    pygame.draw.line(surface, AXIS_COLOR, (left, bottom), (right, bottom))  # x axis

    surface.blit(font.render("Fitness per generation", True, TEXT_COLOR), (left, area.top + 6))
    # Legend: colored swatch + name, right-aligned in the title row.
    lx = right - 170
    for color, name in ((BEST_COLOR, "best"), (AVG_COLOR, "avg")):
        pygame.draw.line(surface, color, (lx, area.top + 14), (lx + 22, area.top + 14), 3)
        text = font.render(name, True, TEXT_COLOR)
        surface.blit(text, (lx + 28, area.top + 6))
        lx += 28 + text.get_width() + 24

    if not history:
        msg = font.render("waiting for the first generation to finish...", True, AXIS_COLOR)
        surface.blit(msg, (left + 20, (top + bottom) // 2 - msg.get_height() // 2))
        return

    # Auto-scale: y runs from 0 to the highest fitness seen so far (best >= avg, so
    # the max of the best line covers both); x runs over the generations played so far.
    y_max = max(max(b for _, b, _ in history), 1)
    first_gen, last_gen = history[0][0], history[-1][0]

    def to_px(gen, value):
        if last_gen == first_gen:
            x = (left + right) // 2          # a single point goes in the middle
        else:
            x = left + (gen - first_gen) / (last_gen - first_gen) * (right - left)
        y = bottom - value / y_max * (bottom - top)
        return (int(x), int(y))

    # Y-axis labels (0, middle, max) with faint gridlines for the upper two.
    for frac in (0.0, 0.5, 1.0):
        y = int(bottom - frac * (bottom - top))
        if frac > 0:
            pygame.draw.line(surface, (50, 50, 66), (left + 1, y), (right, y))
        label = font.render(f"{y_max * frac:.0f}", True, TEXT_COLOR)
        surface.blit(label, (left - 8 - label.get_width(), y - label.get_height() // 2))

    # X-axis labels: first and last generation number.
    first = font.render(f"gen {first_gen}", True, TEXT_COLOR)
    surface.blit(first, (left, bottom + 6))
    if last_gen != first_gen:
        last = font.render(f"gen {last_gen}", True, TEXT_COLOR)
        surface.blit(last, (right - last.get_width(), bottom + 6))

    for index, color in ((1, BEST_COLOR), (2, AVG_COLOR)):
        points = [to_px(entry[0], entry[index]) for entry in history]
        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, 2)
        pygame.draw.circle(surface, color, points[-1], 3)   # marks the newest generation


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH + PANEL_WIDTH, HEIGHT + GRAPH_HEIGHT))
    pygame.display.set_caption("Step 4 — Evolution")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    population = Population()
    fast = False   # SPACE toggles: run many sim steps per drawn frame to watch evolution faster

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                fast = not fast

        for _ in range(10 if fast else 1):
            population.update()
            if population.all_dead():
                population.evolve()

        leader = population.leader()

        # --- draw --------------------------------------------------------
        screen.fill(BG_COLOR)
        for f in population.foods:
            pygame.draw.circle(screen, FOOD_COLOR, (int(f.x), int(f.y)), FOOD_RADIUS)
        for c in population.creatures:
            if c.alive and c is not leader:
                c.draw(screen)
        leader.draw(screen, LEADER_COLOR)   # drawn last so it sits on top

        hud = font.render(
            f"gen: {population.generation}   alive: {population.alive_count()}/{population.size}"
            f"   best ever: {population.best_ever}{'   [FAST]' if fast else ''}",
            True, TEXT_COLOR,
        )
        screen.blit(hud, (10, 10))
        hud2 = font.render(
            f"leader  energy: {leader.energy:4.0f}   age: {leader.age:5d}   eaten: {leader.food_eaten}   fitness: {leader.fitness()}",
            True, LEADER_COLOR,
        )
        screen.blit(hud2, (10, 32))

        draw_brain(screen, leader.brain, font)
        draw_fitness_graph(screen, population.history, font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
