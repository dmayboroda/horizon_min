import os, sys, argparse
import pygame
import pygame.transform
from game.registry import adjpos, adjrect, adjwidth, adjheight

# Game parameters
SCREEN_WIDTH, SCREEN_HEIGHT = adjpos (800, 500)
TITLE = "Symons Media: Duck Hunt"
FRAMES_PER_SEC = 50
BG_COLOR = 255, 255, 255

# Initialize pygame before importing modules
pygame.mixer.pre_init(44100, -16, 2, 1024)
pygame.init()
pygame.display.set_caption(TITLE)
pygame.mouse.set_visible(False)

import game.driver

class Game(object):
    def __init__(self, sound_enabled=True):
        self.running = True
        self.surface = None
        self.clock = pygame.time.Clock()
        self.size = SCREEN_WIDTH, SCREEN_HEIGHT
        background = os.path.join('media', 'background.jpg')
        bg = pygame.image.load(background)
        self.background = pygame.transform.smoothscale (bg, self.size)
        self.driver = None
        self.sound_enabled = sound_enabled

    def init(self):
        self.surface = pygame.display.set_mode(self.size)
        self.driver = game.driver.Driver(self.surface, sound_enabled=self.sound_enabled)

    def handleEvent(self, event):
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key is 27):
            self.running = False
        else:
            self.driver.handleEvent(event)

    def loop(self):
        self.clock.tick(FRAMES_PER_SEC)
        self.driver.update()

    def render(self):
        self.surface.blit(self.background, (0,0))
        self.driver.render()
        pygame.display.flip()

    def cleanup(self):
        pygame.quit()
        sys.exit(0)

    def execute(self):
        self.init()

        while (self.running):
            for event in pygame.event.get():
                self.handleEvent(event)
            self.loop()
            self.render()

        self.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duck Hunt Game")
    parser.add_argument('--no-sound', action='store_true', help='Start the game with sound disabled')
    args = parser.parse_args()

    theGame = Game(sound_enabled=not args.no_sound)
    theGame.execute()
