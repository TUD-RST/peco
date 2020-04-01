import pyglet

class DrawText(object):
    """
    Class that implements a text label, which is used to render the simulation time.
    """
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()