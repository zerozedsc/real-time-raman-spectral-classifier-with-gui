def get_main_stylesheet(font_family: str) -> str:
    """
    Generates the main application stylesheet with a dynamically provided font family.

    Args:
        font_family (str): The CSS font-family string (e.g., "'Inter', 'sans-serif'").

    Returns:
        str: The complete application stylesheet.
    """
    return f"""
        QWidget {{
            font-family: {font_family};
            font-size: 15px;
            background-color: #f3f6f9;
            color: #34495e;
        }}
        
        /* --- App Tab Bar Styles --- */
        #appTabBar {{
            background-color: #e4e9ef;
            border-bottom: 1px solid #d1dbe5;
        }}
        #appTabBar QPushButton {{
            font-size: 15px;
            font-weight: 600;
            color: #566573;
            padding: 12px 25px;
            border: none;
            background-color: transparent;
            border-bottom: 3px solid transparent; /* Inactive state */
        }}
        #appTabBar QPushButton:hover {{
            color: #2c3e50;
            background-color: #dce3e9;
        }}
        #appTabBar QPushButton:checked {{
            color: #005a9e;
            border-bottom: 3px solid #0078d4; /* Active state */
            background-color: #f3f6f9;
        }}
        #workspaceStack > QWidget {{
            background-color: #f3f6f9;
        }}
        
        /* --- Home Page Specific Styles --- */
        #homeSidebar {{
            background-color: #ffffff;
            border-right: 1px solid #e0e5ea;
            min-width: 280px;
            max-width: 320px;
        }}
        #sidebarTitle {{
            font-size: 24px;
            font-weight: 600;
            color: #00406b;
        }}
        #sidebarSubtitle {{
            font-size: 15px;
            color: #566573;
        }}
        #sidebarButton {{
            font-size: 16px;
            font-weight: 500;
            padding: 16px;
            text-align: left;
            background-color: transparent;
            border: none;
            border-radius: 8px;
            color: #2c3e50;
        }}
        #sidebarButton:hover {{
            background-color: #eaf2f8;
            color: #005a9e;
        }}
        #homeMainArea {{
            background-color: #f3f6f9;
        }}
        #mainAreaTitle {{
            font-size: 28px;
            font-weight: 300;
            color: #34495e;
            padding-bottom: 15px;
            border-bottom: 1px solid #e0e5ea;
        }}
        #recentProjectsList {{
            border: none;
            background-color: transparent;
        }}
        #recentProjectsList::item {{
            border: none;
            padding: 2px;
        }}
        #recentProjectItem {{
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #e0e5ea;
            padding: 15px 20px;
        }}
        #recentProjectItem:hover {{
            border: 1px solid #0078d4;
        }}
        #projectNameLabel {{
            font-size: 17px;
            font-weight: 600;
            color: #2c3e50;
        }}
        #projectPathLabel {{
            font-size: 13px;
            color: #7f8c8d;
        }}
        
        /* --- General Widget Styles --- */
        #pageTitle {{
            color: #00406b;
            font-size: 22px;
            font-weight: 600;
        }}
        QGroupBox {{
            font-size: 16px;
            font-weight: 600;
            color: #005a9e;
            border: 1px solid #d1dbe5;
            border-radius: 8px;
            margin-top: 10px;
            background-color: #ffffff;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 10px 5px 10px;
        }}
        QLineEdit {{
            border: 1px solid #c0c8d0;
            border-radius: 5px;
            padding: 9px;
            background-color: #ffffff;
            font-size: 14px;
        }}
        QPushButton {{
            background-color: #e1e8ed;
            color: #2c3e50;
            border: 1px solid #c0c8d0;
            border-radius: 6px;
            padding: 10px 15px;
            font-weight: 500;
        }}
        QPushButton:hover {{
            background-color: #d4dde3;
            border-color: #0078d4;
        }}
        #ctaButton {{
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0078d4, stop:1 #005a9e);
            color: white;
            font-size: 16px;
            font-weight: 600;
            padding: 12px 20px;
            border: 1px solid #00406b;
        }}
        #ctaButton:hover {{
            background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0084e8, stop:1 #006ab8);
        }}
        #dragDropLabel {{
            border: 2px dashed #b0c4de;
            border-radius: 8px;
            background-color: #f8f9fa;
            color: #7f8c8d;
            font-weight: 500;
        }}

        /* --- Toast Notification Styles --- */
        #toastLabel {{
            color: white;
            font-size: 15px;
            font-weight: 500;
            padding: 12px 20px;
            border-radius: 8px;
        }}
        #toastLabel[level="info"] {{
            background-color: rgba(44, 62, 80, 0.85); /* Dark Slate */
        }}
        #toastLabel[level="success"] {{
            background-color: rgba(39, 174, 96, 0.85); /* Green */
        }}
        #toastLabel[level="error"] {{
            background-color: rgba(192, 57, 43, 0.85); /* Red */
        }}
    """
