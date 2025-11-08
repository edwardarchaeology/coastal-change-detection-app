# Alternative: Simpler Shiny app without complex widgets for testing
from shiny import App, ui, render, reactive
from datetime import date, timedelta

app_ui = ui.page_navbar(
    ui.nav_panel("🌊 Coastal Monitor",
        ui.div(
            ui.h2("Coastal Change Detection App"),
            ui.p("✓ If you see this, the Shiny app is rendering!", 
                 style="background-color: #d4edda; padding: 15px; border-radius: 5px; border: 2px solid #28a745;"),
            style="padding: 20px;"
        ),
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("📍 Location"),
                ui.input_text("search_location", "", placeholder="Grand Isle, Louisiana"),
                ui.input_action_button("btn_test", "Test Button", class_="btn-primary w-100"),
                ui.hr(),
                ui.p("Date range:"),
                ui.input_date("date_start", "Start", value=date.today() - timedelta(days=30)),
                ui.input_date("date_end", "End", value=date.today()),
                width=300
            ),
            ui.card(
                ui.card_header("Status"),
                ui.output_text_verbatim("status_output"),
            ),
            ui.card(
                ui.card_header("Instructions"),
                ui.p("This is a simplified version to test if Shiny is working."),
                ui.p("If you see this page, the issue is specifically with ipyleaflet widgets."),
                ui.hr(),
                ui.h4("Next Steps:"),
                ui.tags.ol(
                    ui.tags.li("Check browser console (F12) for JavaScript errors"),
                    ui.tags.li("Verify ipywidgets and ipyleaflet are installed"),
                    ui.tags.li("Try updating: uv pip install --upgrade shinywidgets ipyleaflet ipywidgets"),
                ),
            ),
        ),
    ),
    title="Coastal Monitor - Test Version"
)

def server(input, output, session):
    click_count = reactive.Value(0)
    
    @reactive.Effect
    @reactive.event(input.btn_test)
    def handle_click():
        click_count.set(click_count.get() + 1)
        ui.notification_show(f"Button clicked {click_count.get()} times!", type="message")
    
    @output
    @render.text
    def status_output():
        return f"""
Shiny App Status: ✓ WORKING
        
Location: {input.search_location() or '(not set)'}
Date Range: {input.date_start()} to {input.date_end()}
Button Clicks: {click_count.get()}

If you see this updating when you interact with controls,
then Shiny itself is working fine!

The blank page issue is specifically with ipyleaflet widgets.
        """

app = App(app_ui, server)
