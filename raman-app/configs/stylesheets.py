def get_main_stylesheet(font_family: str) -> str:
    """
    Generates the main application stylesheet with a dynamically provided font family.
    """
    return f"""
        QWidget {{
            font-family: {font_family};
            font-size: 14px;
            background-color: #f8f9fa;
            color: #2c3e50;
        }}
        
        /* --- Home Page Layout --- */
        #homePage {{
            background-color: #f8f9fa;
        }}
        
        #homeSidebar {{
            background-color: #ffffff;
            border-right: 1px solid #e9ecef;
        }}
        
        #homeContentArea {{
            background-color: #f8f9fa;
        }}
        
        /* --- Home Page Typography --- */
        #homeTitle {{
            font-size: 28px;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 4px;
        }}
        
        #homeSubtitle {{
            font-size: 14px;
            font-weight: 400;
            color: #6c757d;
            margin-bottom: 0px;
        }}
        
        #recentProjectsHeader {{
            font-size: 18px;
            font-weight: 600;
            color: #1a1a1a;
        }}
        
        /* --- Action Cards --- */
        #actionCard {{
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 0px;
        }}
        
        #actionCard[hover="true"] {{
            border-color: #0078d4;
            box-shadow: 0 4px 12px rgba(0, 120, 212, 0.15);
            transform: translateY(-2px);
        }}
        
        #actionCard #cardTitle {{
            font-size: 16px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 4px;
        }}
        
        #actionCard #cardDescription {{
            font-size: 13px;
            color: #6c757d;
            line-height: 1.4;
        }}
        
        /* --- Recent Projects List --- */
        #recentProjectsList {{
            background-color: transparent;
            border: none;
            outline: none;
        }}
        
        #recentProjectsList::item {{
            border: none;
            background-color: transparent;
            padding: 0px;
            margin-bottom: 8px;
        }}
        
        #recentProjectsList::item:selected {{
            background-color: transparent;
        }}
        
        #recentProjectItem {{
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin: 0px;
        }}
        
        #recentProjectItem:hover {{
            border-color: #0078d4;
            background-color: #f8fcff;
        }}
        
        #recentProjectItem #projectName {{
            font-size: 15px;
            font-weight: 600;
            color: #1a1a1a;
            background-color: transparent;
        }}
        
        #recentProjectItem #projectTime {{
            font-size: 12px;
            color: #6c757d;
            background-color: transparent;
        }}
        
        /* --- App Tab Bar Styles --- */
        #appTabBar {{
            background-color: #ffffff;
            border-bottom: 1px solid #e9ecef;
        }}
        
        #appTabBar QPushButton {{
            font-size: 14px;
            font-weight: 500;
            color: #6c757d;
            padding: 12px 24px;
            border: none;
            background-color: transparent;
            border-bottom: 2px solid transparent;
        }}
        
        #appTabBar QPushButton:hover {{
            color: #1a1a1a;
            background-color: #f8f9fa;
        }}
        
        /* --- COLOR CHANGE --- */
        /* This changes the style for the selected tab button. */
        #appTabBar QPushButton:checked {{
            color: #2c3e50;
            border-bottom: 2px solid #adb5bd;
            background-color: #e9ecef; /* This is the gray background for the tab */
        }}
        
        /* --- General Widget Styles --- */
        QGroupBox {{
            font-size: 16px;
            font-weight: 600;
            color: #1a1a1a;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-top: 12px;
            background-color: #ffffff;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            background-color: #ffffff;
        }}
        
        QLineEdit, QTextEdit {{
            border: 1px solid #ced4da;
            border-radius: 6px;
            padding: 8px 12px;
            background-color: #ffffff;
            font-size: 14px;
        }}
        
        QLineEdit:focus, QTextEdit:focus {{
            border-color: #0078d4;
            outline: none;
        }}
        
        QPushButton {{
            background-color: #ffffff;
            color: #1a1a1a;
            border: 1px solid #ced4da;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
        }}
        
        QPushButton:hover {{
            background-color: #f8f9fa;
            border-color: #adb5bd;
        }}
        
        QPushButton:pressed {{
            background-color: #e9ecef;
        }}
        
        #ctaButton {{
            background-color: #0078d4;
            color: #ffffff;
            border: 1px solid #0078d4;
            font-weight: 600;
        }}
        
        #ctaButton:hover {{
            background-color: #106ebe;
            border-color: #106ebe;
        }}
        
        /* --- Scrollbar Styles --- */
        QScrollBar:vertical {{
            background-color: #f8f9fa;
            width: 8px;
            border-radius: 4px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: #ced4da;
            border-radius: 4px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: #adb5bd;
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        /* --- Matplotlib Toolbar --- */
        /* This targets the QToolBar inside our custom matplotlib widget */
        #matplotlibWidget QToolBar {{
            background-color: #e9ecef; /* A light gray background for the toolbar */
            border-bottom: 1px solid #ced4da;
        }}

        #matplotlibWidget QToolBar QToolButton {{
            background-color: transparent;
            border-radius: 4px;
            padding: 4px;
            margin: 1px;
        }}

        #matplotlibWidget QToolBar QToolButton:hover {{
            background-color: #ced4da; /* Slightly darker on hover */
        }}

        #matplotlibWidget QToolBar QToolButton:checked {{
            background-color: #adb5bd; /* Even darker when a tool is active (like zoom) */
        }}
        
        
    """
