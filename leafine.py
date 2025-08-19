import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stMainBlockContainer, .stMain, .block-container {
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }
    body, html {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }
    .full-screen-iframe {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        border: none;
        z-index: 0;
    }
    /* Button style overrides */
    .custom-button {
        background: linear-gradient(135deg, #8b7355 0%, #ede4d3 50%, #7a8471 100%);
        padding: 1rem 2.5rem;
        border-radius: 20px;
        color: #2d2d2d !important;
        text-align: center;
        font-weight: 700;
        font-size: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-decoration: none !important; 
        display: inline-block;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        position: absolute;
        top: 40px;
        right: 40px;
        z-index: 10;
    }
    .custom-button:hover, .custom-button:focus, .custom-button:visited, .custom-button:active {
        color: #2d2d2d !important;
        transform: scale(1.05);
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
        text-decoration: none !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    APP_URL = "https://brindha-m-leafine.streamlit.app/"
    st.markdown(
        f"""
        <iframe src="{APP_URL}?embedded=true" 
                class="full-screen-iframe"
                loading="lazy">
        </iframe>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f'<a href="{APP_URL}" target="_blank" class="custom-button">🚀 Open Leafine..</a>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
