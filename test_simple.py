# Simple test app to verify Shiny + ipyleaflet works
from shiny import App, ui
from shinywidgets import output_widget, render_widget
from ipyleaflet import Map

app_ui = ui.page_fluid(
    ui.h2("Test Map"),
    ui.p("If you see a map below, ipyleaflet is working!"),
    output_widget("test_map", height="500px")
)

def server(input, output, session):
    @render_widget
    def test_map():
        m = Map(center=(29.24, -90.06), zoom=10)
        return m

app = App(app_ui, server)
