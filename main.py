"""SAMcloud application entry point."""

from nicegui import ui

from samcloud.app.application import create_application
from samcloud.core.paths import ASSETS_DIRECTORY


def main() -> None:
    """Register the UI and start the local SAMcloud application."""
    create_application()
    ui.run(title="SAMcloud", favicon=ASSETS_DIRECTORY / "favicon.png", reload=False, port=8080)


if __name__ in {"__main__", "__mp_main__"}:
    main()
