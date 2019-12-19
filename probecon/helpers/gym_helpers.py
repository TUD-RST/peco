import pyglet

class DrawText(object):
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()