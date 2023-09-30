custom_css = f"""
        .gradio-header {{
            color: white;
        }}
        .gradio-description {{
            color: white;
        }}
        gradio-app {{
            background-image: url('https://raw.githubusercontent.com/kevinbtalbert/cloudera_kb/main/app_assets/cldr_bg.jpg') !important;
            background-size: cover  !important;
            background-position: center center  !important;
            background-repeat: no-repeat  !important;
            background-attachment: fixed  !important;
        }}
        #custom-logo {{
            text-align: center;
        }}
        .dark {{
            background-image: url('https://raw.githubusercontent.com/kevinbtalbert/cloudera_kb/main/app_assets/cldr_bg.jpg') !important;
            background-size: cover  !important;
            background-position: center center  !important;
            background-repeat: no-repeat  !important;
            background-attachment: fixed  !important;
        }}
        .gr-interface {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        .gradio-header {{
            background-color: rgba(0, 0, 0, 0.5);
        }}
        .gradio-input-box, .gradio-output-box {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        h1 {{
            color: white; 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: large; !important;
        }}
"""