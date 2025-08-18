import streamlit as st

# Remove page config since it's handled by main app
# st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
    .stMainBlockContainer {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
    }
    .stMain {
        padding: 0rem;
    }
    iframe {
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        width: 100%;
        height: 100%;
        border: none;
        margin: 0;
        padding: 0;
        overflow: hidden;
        z-index: 999999;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown(
        """
        <iframe src="https://brindha-m-leafine.streamlit.app/?embedded=true" 
                width="100%" 
                height="100%" 
                frameborder="0"
                loading="lazy">
        </iframe>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
