"""Pygame GUI: play the trained DQN agent head-to-head as a human.

Wraps `CreatureBattleEnv` directly and reuses its `resolve_turn` /
`get_observation` helpers so the exact same damage / type-effectiveness /
cooldown math used during training also governs human play. The only
difference from `env.step()` is that the opponent's move is chosen by the
trained DQN model instead of the environment's built-in heuristic AI.

Usage:
    python play_gui.py

Requires `dqn_battle_agent.pth` to exist (run `python train.py` first).
"""

from __future__ import annotations

import random
import sys

import pygame
import torch

from battle_env import (
    NUM_MOVES,
    TYPE_NAMES,
    CreatureBattleEnv,
    Creature,
    get_observation,
    resolve_turn,
)
from model import DQN

MODEL_PATH = "dqn_battle_agent.pth"

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 30

# --- Dark-mode palette -------------------------------------------------
COLOR_BG = (24, 24, 36)
COLOR_PANEL = (36, 36, 52)
COLOR_TEXT = (230, 230, 240)
COLOR_TEXT_DIM = (150, 150, 165)
COLOR_HP_GREEN = (76, 201, 106)
COLOR_HP_YELLOW = (230, 200, 60)
COLOR_HP_RED = (220, 70, 70)
COLOR_HP_BG = (60, 60, 78)
COLOR_BUTTON = (52, 52, 74)
COLOR_BUTTON_HOVER = (68, 68, 96)
COLOR_BUTTON_DISABLED = (40, 40, 50)
COLOR_LOG_BG = (18, 18, 28)
COLOR_OVERLAY = (10, 10, 16, 210)

TYPE_COLORS = {
    0: (190, 190, 220),  # Moon
    1: (240, 170, 40),   # Sun
    2: (110, 160, 90),   # Earth
    3: (210, 90, 50),    # Meteor
    4: (90, 40, 130),    # Black Hole
}

MAX_LOG_LINES = 6


class BattleGUI:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Alien Battle: Human vs DQN")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont(None, 32)
        self.font_medium = pygame.font.SysFont(None, 24)
        self.font_small = pygame.font.SysFont(None, 18)

        self.env = CreatureBattleEnv()
        self.rng = random.Random()

        self.device = torch.device("cpu")
        self.model = DQN().to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

        self.log_lines: list[str] = []
        self.game_over = False
        self.player_won = False

        self._new_battle()

    def _new_battle(self) -> None:
        self.env.reset()
        self.log_lines = [f"{self.player.name} vs {self.ai.name} — a wild alien battle begins!"]
        self.game_over = False
        self.player_won = False

    @property
    def player(self) -> Creature:
        return self.env.agent

    @property
    def ai(self) -> Creature:
        return self.env.opponent

    def _log(self, message: str) -> None:
        self.log_lines.append(message)
        if len(self.log_lines) > MAX_LOG_LINES:
            self.log_lines = self.log_lines[-MAX_LOG_LINES:]

    def _ai_choose_action(self) -> int:
        """Predict the AI creature's move from its own perspective."""
        state = get_observation(self.ai, self.player)
        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.model(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def _decrement_cooldowns(self) -> None:
        for creature in (self.player, self.ai):
            creature.cooldowns = [max(0, cd - 1) for cd in creature.cooldowns]

    def handle_player_move(self, action: int) -> None:
        if self.game_over or self.player.cooldowns[action] > 0:
            return

        _, log = resolve_turn(self.player, self.ai, action, self.rng)
        self._log(log)

        if self.ai.is_fainted():
            self._log("You win!")
            self.game_over = True
            self.player_won = True
            return

        ai_action = self._ai_choose_action()
        _, ai_log = resolve_turn(self.ai, self.player, ai_action, self.rng)
        self._log(ai_log)

        if self.player.is_fainted():
            self._log("You lose!")
            self.game_over = True
            self.player_won = False
            return

        self._decrement_cooldowns()

    # --- Drawing ---------------------------------------------------------

    def _hp_color(self, hp_pct: float) -> tuple[int, int, int]:
        if hp_pct > 0.6:
            return COLOR_HP_GREEN
        if hp_pct > 0.3:
            return COLOR_HP_YELLOW
        return COLOR_HP_RED

    def _draw_creature_panel(self, creature: Creature, label: str, x: int, y: int) -> None:
        width, height = 260, 104
        pygame.draw.rect(self.screen, COLOR_PANEL, (x, y, width, height), border_radius=8)

        name_surf = self.font_medium.render(label, True, COLOR_TEXT)
        self.screen.blit(name_surf, (x + 12, y + 8))

        alien_surf = self.font_small.render(creature.name, True, COLOR_TEXT_DIM)
        self.screen.blit(alien_surf, (x + 12, y + 30))

        type_name = TYPE_NAMES[creature.creature_type]
        type_color = TYPE_COLORS[creature.creature_type]
        badge_surf = self.font_small.render(type_name, True, (0, 0, 0))
        badge_rect = pygame.Rect(x + width - 90, y + 10, 78, 22)
        pygame.draw.rect(self.screen, type_color, badge_rect, border_radius=6)
        self.screen.blit(badge_surf, badge_surf.get_rect(center=badge_rect.center))

        hp_pct = creature.hp_pct
        bar_x, bar_y, bar_w, bar_h = x + 12, y + 58, width - 24, 18
        pygame.draw.rect(self.screen, COLOR_HP_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        pygame.draw.rect(
            self.screen,
            self._hp_color(hp_pct),
            (bar_x, bar_y, int(bar_w * hp_pct), bar_h),
            border_radius=4,
        )
        hp_text = self.font_small.render(f"{max(0, creature.hp)} / {100} HP", True, COLOR_TEXT)
        self.screen.blit(hp_text, (bar_x, bar_y + bar_h + 4))

    def _draw_log(self) -> None:
        log_rect = pygame.Rect(20, 200, WINDOW_WIDTH - 40, 140)
        pygame.draw.rect(self.screen, COLOR_LOG_BG, log_rect, border_radius=8)
        for i, line in enumerate(self.log_lines):
            surf = self.font_small.render(line, True, COLOR_TEXT_DIM)
            self.screen.blit(surf, (log_rect.x + 12, log_rect.y + 10 + i * 20))

    def _move_button_rects(self) -> list[pygame.Rect]:
        button_w, button_h, gap = 180, 70, 15
        total_w = button_w * NUM_MOVES + gap * (NUM_MOVES - 1)
        start_x = (WINDOW_WIDTH - total_w) // 2
        y = WINDOW_HEIGHT - button_h - 20
        return [
            pygame.Rect(start_x + i * (button_w + gap), y, button_w, button_h)
            for i in range(NUM_MOVES)
        ]

    def _draw_move_buttons(self, mouse_pos: tuple[int, int]) -> None:
        for i, rect in enumerate(self._move_button_rects()):
            move = self.player.moves[i]
            on_cooldown = self.player.cooldowns[i] > 0

            if on_cooldown:
                color = COLOR_BUTTON_DISABLED
            elif rect.collidepoint(mouse_pos):
                color = COLOR_BUTTON_HOVER
            else:
                color = COLOR_BUTTON

            pygame.draw.rect(self.screen, color, rect, border_radius=8)

            name_color = COLOR_TEXT_DIM if on_cooldown else COLOR_TEXT
            name_surf = self.font_small.render(move.name, True, name_color)
            self.screen.blit(name_surf, (rect.x + 10, rect.y + 8))

            type_surf = self.font_small.render(TYPE_NAMES[move.move_type], True, TYPE_COLORS[move.move_type])
            self.screen.blit(type_surf, (rect.x + 10, rect.y + 30))

            status = f"CD: {self.player.cooldowns[i]}" if on_cooldown else "Ready"
            status_surf = self.font_small.render(status, True, name_color)
            self.screen.blit(status_surf, (rect.x + 10, rect.y + 50))

    def _draw_game_over(self) -> pygame.Rect:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill(COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        message = "You Win!" if self.player_won else "You Lose!"
        msg_surf = self.font_large.render(message, True, COLOR_TEXT)
        self.screen.blit(msg_surf, msg_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40)))

        button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 90, WINDOW_HEIGHT // 2 + 10, 180, 48)
        pygame.draw.rect(self.screen, COLOR_BUTTON_HOVER, button_rect, border_radius=8)
        again_surf = self.font_medium.render("Play Again", True, COLOR_TEXT)
        self.screen.blit(again_surf, again_surf.get_rect(center=button_rect.center))
        return button_rect

    def draw(self, mouse_pos: tuple[int, int]) -> pygame.Rect | None:
        self.screen.fill(COLOR_BG)
        self._draw_creature_panel(self.player, "You", 20, WINDOW_HEIGHT - 220)
        self._draw_creature_panel(self.ai, "AI Opponent", WINDOW_WIDTH - 280, 20)
        self._draw_log()
        self._draw_move_buttons(mouse_pos)

        play_again_rect = None
        if self.game_over:
            play_again_rect = self._draw_game_over()

        pygame.display.flip()
        return play_again_rect

    def run(self) -> None:
        while True:
            mouse_pos = pygame.mouse.get_pos()
            play_again_rect = self.draw(mouse_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.game_over:
                        if play_again_rect and play_again_rect.collidepoint(event.pos):
                            self._new_battle()
                        continue

                    for i, rect in enumerate(self._move_button_rects()):
                        if rect.collidepoint(event.pos) and self.player.cooldowns[i] == 0:
                            self.handle_player_move(i)
                            break

            self.clock.tick(FPS)


if __name__ == "__main__":
    BattleGUI().run()
