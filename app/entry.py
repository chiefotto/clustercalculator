import streamlit as st
# from pages.home import home_page
# from pages.game_details import game_details_page

# home = st.Page(home_page, title="Home")
# details = st.Page(game_details_page, title="Game Details")

home = st.Page("pages/home.py", title="Home", default=True)
details = st.Page("pages/game_details.py", title="Game Details")
player = st.Page("pages/player.py", title="Player View")
settings = st.Page("pages/settings.py", title="Settings")
game_logs = st.Page("pages/game_logs.py", title="Game Logs")
pg = st.navigation([home, details,player,settings, game_logs])
pg.run()
