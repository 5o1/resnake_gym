"""SDL controller reader; no keyboard mapping and no game-state access.

Reference: https://pyga.me/docs/ref/sdl2_controller.html
SDL stick Y is down-positive; the normalized report uses XInput up-positive.
"""

import resnake_gym.gamepad as gamepad


class SDLGamepad:
    def __init__(self, index=0):
        import pygame
        from pygame._sdl2 import controller

        controller.init()
        if not controller.is_controller(index):
            raise RuntimeError(f"No SDL-mapped controller at index {index}")
        self.device = controller.Controller(index)
        self.pygame = pygame
        button_names = (
            "DPAD_UP",
            "DPAD_DOWN",
            "DPAD_LEFT",
            "DPAD_RIGHT",
            "START",
            "BACK",
            "LEFTSTICK",
            "RIGHTSTICK",
            "LEFTSHOULDER",
            "RIGHTSHOULDER",
            "A",
            "B",
            "X",
            "Y",
        )
        self.buttons = [getattr(pygame, "CONTROLLER_BUTTON_" + x) for x in button_names]
        self.axes = [
            getattr(pygame, "CONTROLLER_AXIS_" + x)
            for x in (
                "LEFTX",
                "LEFTY",
                "RIGHTX",
                "RIGHTY",
                "TRIGGERLEFT",
                "TRIGGERRIGHT",
            )
        ]

    def read(self):
        self.pygame.event.pump()
        if not self.device.attached():
            raise RuntimeError("Controller disconnected; stop execution explicitly")
        report = gamepad.neutral()
        report[:14] = [self.device.get_button(button) for button in self.buttons]
        for i, axis in enumerate(self.axes):
            value = self.device.get_axis(axis)
            if i < 4:
                value /= 32768 if value < 0 else 32767
                if i in (1, 3):
                    value = -value
            else:
                value = min(max(value / 32767, 0), 1)
            report[14 + i] = value
        return gamepad.validate(report)

    def close(self):
        self.device.quit()
