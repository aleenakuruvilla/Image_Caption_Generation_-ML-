import streamlit as st
from streamlit_option_menu import option_menu
import home,Definitions,code,caption_generator

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function": function
        })

    def run(self):
        selected = option_menu(
            menu_title=None,
            options=['Home','Definitions', 'Code','Caption Generator'],
            icons=['🏠', '📊', '🧪', '💳'],
            default_index=0,
            orientation="horizontal"
        )

        for app in self.apps:
            if selected == app['title']:
                app['function']()

# Instantiate the MultiApp class and run the application
app = MultiApp()

# Add all your apps to the MultiApp instance
app.add_app('Home', home.app)
app.add_app('Definitions',Definitions.app)
app.add_app('Code', code.app)
app.add_app('Caption Generator', caption_generator.app)
# app.add_app('Prediction', prediction.app)

# Run the application
app.run()
