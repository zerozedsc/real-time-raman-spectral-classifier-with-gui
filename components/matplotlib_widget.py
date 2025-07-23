import sys
import pandas as pd
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class MatplotlibWidget(QWidget):
    """
    A custom widget to embed a Matplotlib plot into a PySide6 application.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("matplotlibWidget")
        
        # --- Create a Figure and a Canvas ---
        self.figure = Figure(figsize=(5, 4), dpi=100, facecolor='whitesmoke')
        self.canvas = FigureCanvas(self.figure)
        
        # --- Create a Toolbar ---
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def update_plot(self, new_figure: Figure):
        """
        Clears the current figure and replaces it with a new one.
        """
        self.figure.clear()
        # This is a way to "copy" the contents of the new figure
        # to the existing figure managed by the canvas.
        axes_list = new_figure.get_axes()
        
        if not axes_list:
            # No axes to copy
            self.canvas.draw()
            return
            
        for i, ax in enumerate(axes_list):
            # Create a new subplot in the same position
            # For simple cases, we can use add_subplot(111) for single plots
            if len(axes_list) == 1:
                new_ax = self.figure.add_subplot(111)
            else:
                # For multiple subplots, try to preserve layout
                new_ax = self.figure.add_subplot(len(axes_list), 1, i+1)
            
            # Copy all lines from the original axes
            for line in ax.get_lines():
                new_ax.plot(line.get_xdata(), line.get_ydata(), 
                           label=line.get_label(), 
                           color=line.get_color(),
                           linestyle=line.get_linestyle(),
                           linewidth=line.get_linewidth())
            
            # Copy axes properties
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())
            new_ax.set_xlim(ax.get_xlim())
            new_ax.set_ylim(ax.get_ylim())
            
            # Copy legend if it exists
            if ax.get_legend():
                new_ax.legend()
            
            # Add grid
            new_ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear_plot(self):
        """Clears the plot area."""
        self.figure.clear()
        self.canvas.draw()


def plot_spectra(df: pd.DataFrame, title: str = "") -> Figure:
    """
    Generates a matplotlib Figure object containing a plot of the spectra.
    Plots a maximum of 10 spectra for clarity and applies themed styling.
    """
    fig = Figure(figsize=(8, 6), dpi=100, facecolor='#eaf2f8') # Themed background
    ax = fig.add_subplot(111, facecolor='#eaf2f8') # Themed background

    # --- Robustness Check ---
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data to display.", ha='center', va='center', fontsize=14, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig

    # --- Plotting Logic ---
    num_spectra = df.shape[1]
    plot_title = "Loaded Raman Spectra"
    
    # Limit the number of plotted spectra for clarity
    if num_spectra > 10:
        df_to_plot = df.iloc[:, :10]
        plot_title += f" (showing first 10 of {num_spectra})"
    else:
        df_to_plot = df

    # Plot each spectrum
    for column in df_to_plot.columns:
        ax.plot(df_to_plot.index, df_to_plot[column], label=column)
    
    ax.set_title(plot_title, fontsize=14, weight='bold')
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#d1dbe5')
    
    # Customize legend
    legend = ax.legend(facecolor='#ffffff', framealpha=0.7)
    for text in legend.get_texts():
        text.set_color('#34495e')

    # Customize tick colors
    ax.tick_params(axis='x', colors='#34495e')
    ax.tick_params(axis='y', colors='#34495e')

    # Customize spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor('#34495e')

    fig.tight_layout()
    return fig
