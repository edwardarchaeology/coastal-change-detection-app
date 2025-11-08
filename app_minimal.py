# Minimal Shiny app to test basic rendering
from shiny import App, ui

app_ui = ui.page_fluid(
    ui.h1("Coastal Change Monitor - Test"),
    ui.p("If you can see this, basic Shiny is working!"),
    ui.card(
        ui.card_header("Status Check"),
        ui.p("✓ Shiny UI rendering correctly"),
        ui.p("✓ Page is visible"),
        ui.p("✓ Styles are applied"),
    )
)

def server(input, output, session):
    pass

app = App(app_ui, server)
