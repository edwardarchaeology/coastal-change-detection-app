"""
Diagnostic: Test if ipyleaflet widgets work in Shiny
This will help us identify if the issue is with ipyleaflet rendering
"""
from shiny import App, ui
from shinywidgets import output_widget, render_widget
from ipyleaflet import Map
import sys

print("="*60, file=sys.stderr)
print("DIAGNOSTIC APP STARTING", file=sys.stderr)
print("="*60, file=sys.stderr)

app_ui = ui.page_fluid(
    ui.panel_title("Ipyleaflet Diagnostic"),
    ui.h2("Testing ipyleaflet in Shiny"),
    ui.hr(),
    ui.p("Step 1: If you see this text, basic Shiny rendering works ✓"),
    ui.p("Step 2: If you see a map below, ipyleaflet widgets are working ✓"),
    ui.hr(),
    ui.h3("Map Test:"),
    output_widget("diagnostic_map", height="500px"),
    ui.hr(),
    ui.p("If the map appears above, everything is working!"),
    ui.p("If not, check the browser console for errors (F12)."),
)

def server(input, output, session):
    print("Server function called", file=sys.stderr)
    
    @render_widget
    def diagnostic_map():
        print("Creating diagnostic map...", file=sys.stderr)
        try:
            m = Map(
                center=(29.24, -90.06),
                zoom=10,
                scroll_wheel_zoom=True
            )
            print(f"Map created successfully: center={m.center}, zoom={m.zoom}", file=sys.stderr)
            return m
        except Exception as e:
            print(f"ERROR creating map: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise

app = App(app_ui, server)

if __name__ == "__main__":
    print("App object created", file=sys.stderr)
