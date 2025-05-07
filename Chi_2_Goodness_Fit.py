# Chi-Square Test Visualizer (Improved Version)
# Educational tool for understanding statistical tests in Alzheimer's research

import pygame
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Required for pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from scipy import stats
import os
import sys
import pandas as pd
import traceback

# Initialize pygame
pygame.init()

# Colors for pygame (RGB format)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
DARK_BLUE = (0, 0, 180)
GRAY = (200, 200, 200)
LIGHT_BLUE = (173, 216, 230)
LIGHT_RED = (255, 182, 193)
LIGHT_GREEN = (144, 238, 144)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 100, 0)
DARK_GRAY = (80, 80, 80)
LIGHT_GRAY = (230, 230, 230)
MEDIUM_GRAY = (180, 180, 180)
PANEL_BLUE = (240, 248, 255)  # AliceBlue
PANEL_GREEN = (240, 255, 240)  # HoneyDew
PANEL_RED = (255, 240, 240)    # MistyRose
PANEL_YELLOW = (255, 255, 224) # LightYellow
NEON_GREEN = (57, 255, 20)     # Bright green for buttons

# Colors for matplotlib (normalized RGB format)
MPL_LIGHT_BLUE = (0, 0, 1)
MPL_LIGHT_RED = (1, 0, 0)

# Layout constants
HEADER_HEIGHT = 60
TITLE_MARGIN = 10
RESULT_MARGIN = 20
PANEL_MARGIN = 15
H_MARGIN = 20
V_MARGIN = 10
SLIDER_SPACING = 40
ROW_SPACING = 40
CELL_WIDTH = 100
CELL_HEIGHT = 50

# Grid layout constants for parameter adjustment section
LABEL_COLUMN_WIDTH = 140
VALUE_COLUMN_WIDTH = 70
SLIDER_MIN_WIDTH = 200
LABEL_SLIDER_GAP = 20
SLIDER_VALUE_GAP = 16

# Screen dimensions with minimum sizes
MIN_WIDTH, MIN_HEIGHT = 1000, 700
initial_width, initial_height = 1200, 800
WIDTH, HEIGHT = initial_width, initial_height
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Chi-Square Test Visualizer: Statistical Tests for Alzheimer's Research")

# Fonts
font_small = pygame.font.SysFont('Arial', 16)
font_medium = pygame.font.SysFont('Arial', 20)
font_large = pygame.font.SysFont('Arial', 24)
font_title = pygame.font.SysFont('Arial', 40, bold=True)
font_result = pygame.font.SysFont('Arial', 22, bold=True)
font_panel_title = pygame.font.SysFont('Arial', 22, bold=True)

# Game states
MENU = 0
GOODNESS_OF_FIT = 1
INDEPENDENCE = 2
TUTORIAL = 3

# Initial state
state = MENU

# Default data for Goodness of Fit test
# Example: Genotype frequencies in Alzheimer's patients
# Categories could be AA, Aa, aa genotypes
gof_categories = ["AA", "Aa", "aa"]
gof_observed = [74, 50, 20]  # Observed counts based on Image 1
gof_expected = [25.0, 42.7, 33.5]  # Expected percentages based on Image 1

# Default data for Independence test
# Example: Treatment response in Alzheimer's patients with different APOE variants
independence_rownames = ["APOE ε4+", "APOE ε4-"]
independence_colnames = ["Improved", "No Change", "Declined"]
independence_data = np.array([
    [15, 20, 25],  # APOE ε4+ responses
    [25, 20, 15]   # APOE ε4- responses
])

# Sliders for adjusting data
sliders_gof_observed = []
sliders_gof_expected = []

# Initialize sliders for Goodness of Fit test
for i in range(len(gof_categories)):
    # Observed count sliders
    sliders_gof_observed.append({
        "pos": (0, 0), 
        "width": SLIDER_MIN_WIDTH, 
        "value": gof_observed[i], 
        "min": 0, 
        "max": 150, 
        "dragging": False
    })
    
    # Expected proportion sliders (0-100%)
    sliders_gof_expected.append({
        "pos": (0, 0), 
        "width": SLIDER_MIN_WIDTH, 
        "value": gof_expected[i], 
        "min": 0, 
        "max": 100, 
        "dragging": False
    })

# Sliders for Independence test (converted to editable cells)
independence_cells = []
for i in range(independence_data.shape[0]):
    row_cells = []
    for j in range(independence_data.shape[1]):
        row_cells.append({
            "rect": pygame.Rect(0, 0, CELL_WIDTH, CELL_HEIGHT),
            "value": independence_data[i, j], 
            "active": False
        })
    independence_cells.append(row_cells)

# Chi-square test results
chi2_stat = 57.82  # From Image 1
p_value = 0.0000
df = 2
has_calculated = True  # Set to True to show results initially as in the images

# Tutorial pages
tutorial_page = 0
num_tutorial_pages = 4

# Toggleable instruction section
show_instructions = True

# Initialize buttons with empty Rects
calc_button = pygame.Rect(0, 0, 0, 0)
reset_button = pygame.Rect(0, 0, 0, 0)

# For managing input in text cells
active_cell = None

def calculate_goodness_of_fit():
    """Calculate Chi-square Goodness of Fit test"""
    global chi2_stat, p_value, df, has_calculated
    
    try:
        # Get the current observed values from sliders
        observed = [int(slider["value"]) for slider in sliders_gof_observed]
        
        # Calculate the total observed to normalize expected proportions
        total_observed = sum(observed)
        
        if total_observed == 0:
            chi2_stat = 0
            p_value = 1.0
            df = len(observed) - 1
            has_calculated = True
            return
        
        # Convert expected sliders values to proportions and calculate expected counts
        expected_props = [slider["value"] / 100 for slider in sliders_gof_expected]
        
        # Normalize proportions to ensure they sum to 1
        sum_props = sum(expected_props)
        if sum_props > 0:
            expected_props = [prop / sum_props for prop in expected_props]
        else:
            expected_props = [1/len(expected_props)] * len(expected_props)
            
        expected = [prop * total_observed for prop in expected_props]
        
        # Calculate chi-square statistic
        chi2_stat, p_value = stats.chisquare(observed, expected)
        df = len(observed) - 1
        has_calculated = True
        
    except Exception as e:
        print(f"Error calculating chi-square: {e}")
        print(traceback.format_exc())
        chi2_stat = 0
        p_value = 1.0
        df = len(gof_observed) - 1
        has_calculated = True

def calculate_independence():
    """Calculate Chi-square Test of Independence"""
    global chi2_stat, p_value, df, has_calculated
    
    try:
        # Get values from cells
        observed = np.array([[cell["value"] for cell in row] for row in independence_cells])
        
        # Check if any row or column sums to zero
        row_sums = observed.sum(axis=1)
        col_sums = observed.sum(axis=0)
        
        if any(row_sums == 0) or any(col_sums == 0):
            chi2_stat = 0
            p_value = 1.0
            df = (observed.shape[0]-1) * (observed.shape[1]-1)
            has_calculated = True
            return
        
        # Calculate chi-square statistic
        chi2_stat, p_value, df, _ = stats.chi2_contingency(observed)
        has_calculated = True
        
    except Exception as e:
        print(f"Error calculating chi-square: {e}")
        print(traceback.format_exc())
        chi2_stat = 0
        p_value = 1.0
        df = (independence_data.shape[0]-1) * (independence_data.shape[1]-1)
        has_calculated = True

def create_goodness_of_fit_plot(width=600, height=400):
    """Create bar chart comparing observed vs expected frequencies"""
    try:
        # Get data from sliders
        observed = [slider["value"] for slider in sliders_gof_observed]
        
        # Total observed for calculating expected
        total = sum(observed)
        
        # Get expected proportions and calculate expected counts
        expected_props = [slider["value"] / 100 for slider in sliders_gof_expected]
        
        # Normalize proportions
        sum_props = sum(expected_props)
        if sum_props > 0:
            expected_props = [prop / sum_props for prop in expected_props]
        else:
            expected_props = [1/len(expected_props)] * len(expected_props)
            
        expected = [prop * total for prop in expected_props]
        
        # Create figure
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        fig.subplots_adjust(top=0.85, right=0.85, bottom=0.15)
        
        ax = fig.add_subplot(111)
        
        # Bar positions
        x = np.arange(len(gof_categories))
        width_bar = 0.35
        
        # Create bars - match the colors from the screenshot
        ax.bar(x - width_bar/2, observed, width_bar, label='Observed', color='blue', alpha=0.7)
        ax.bar(x + width_bar/2, expected, width_bar, label='Expected', color='red', alpha=0.7)
        
        # Labels and formatting
        ax.set_ylabel('Frequency')
        ax.set_xticks(x)
        ax.set_xticklabels(gof_categories)
        ax.legend()
        
        # Add a title with chi-square results if calculated
        if has_calculated:
            ax.set_title(f'χ² = {chi2_stat:.2f}, p = {p_value:.4f}, df = {df}')
        else:
            ax.set_title('Observed vs Expected Frequencies')
        
        # Convert matplotlib figure to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        plt.close(fig)
        
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        return surf
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        print(traceback.format_exc())
        
        # Create empty surface with error message
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        error_message = f"Error creating plot.\nTry different parameters."
        ax.text(0.5, 0.5, error_message, ha='center', va='center', color='red', fontsize=14)
        ax.axis('off')
        
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        plt.close(fig)
        
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        return surf

def create_independence_plot(width=600, height=400):
    """Create visualization for chi-square test of independence"""
    try:
        # Get data from cells
        observed = np.array([[cell["value"] for cell in row] for row in independence_cells])
        
        # Create figure
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        fig.subplots_adjust(top=0.85, right=0.85, bottom=0.15, left=0.15)
        
        ax = fig.add_subplot(111)
        
        # Convert to DataFrame for easier plotting
        df_obs = pd.DataFrame(observed, index=independence_rownames, columns=independence_colnames)
        
        # Calculate row percentages for better visualization
        df_perc = df_obs.div(df_obs.sum(axis=1), axis=0) * 100
        
        # Create stacked bar chart with custom colors to match the screenshot
        colors = ['darkblue', 'teal', 'yellow']  # These match Image 2
        df_perc.plot(kind='bar', stacked=True, ax=ax, color=colors)
        
        # Add value labels on bars
        for i, row in enumerate(observed):
            cum_sum = 0
            for j, val in enumerate(row):
                if val > 0:  # Only label non-zero values
                    val_perc = df_perc.iloc[i, j]
                    if val_perc > 5:  # Only show percentages for visible segments
                        mid_point = cum_sum + val_perc / 2
                        ax.text(i, mid_point, f'{val_perc:.1f}', ha='center', va='center', 
                               fontsize=9, fontweight='bold')
                    cum_sum += val_perc
        
        # Labels and formatting
        ax.set_ylabel('Percent')
        ax.set_ylim(0, 100)
        ax.set_title('Distribution Comparison')
        
        # Adjust legend position
        ax.legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Convert matplotlib figure to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        plt.close(fig)
        
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        return surf
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        print(traceback.format_exc())
        
        # Create empty surface with error message
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        error_message = f"Error creating plot.\nTry different parameters."
        ax.text(0.5, 0.5, error_message, ha='center', va='center', color='red', fontsize=14)
        ax.axis('off')
        
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        plt.close(fig)
        
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        return surf

def calculate_layout():
    """Calculate all UI component positions based on current window size"""
    global calc_button, reset_button
    
    # Calculate component heights
    header_height = HEADER_HEIGHT
    results_panel_height = 40
    chart_ratio = 0.4  # Proportion of usable height for chart
    instructions_height = 40
    panel_height = 250 if state == GOODNESS_OF_FIT else 300  # Taller for Independence test
    button_height = 40
    
    # Calculate total available height after header
    available_height = HEIGHT - header_height - PANEL_MARGIN
    
    # Results panel - positioned below title with margin
    results_panel_top = header_height + RESULT_MARGIN
    
    # Calculate remaining height after results panel
    remaining_height = available_height - results_panel_height - RESULT_MARGIN
    
    # Chart dimensions - responsive to window size
    chart_height = min(remaining_height * chart_ratio, 450)
    chart_width = min(WIDTH * 0.8, 700)
    
    # Chart position - centered horizontally, below results panel
    chart_left = (WIDTH - chart_width) // 2
    chart_top = results_panel_top + results_panel_height + PANEL_MARGIN
    
    # Instructions panel - placed below the chart
    instructions_top = chart_top + chart_height + PANEL_MARGIN
    
    # Panel dimensions - responsive to window size with minimum widths
    min_panel_width = LABEL_COLUMN_WIDTH + SLIDER_MIN_WIDTH + VALUE_COLUMN_WIDTH + LABEL_SLIDER_GAP + SLIDER_VALUE_GAP + H_MARGIN*2
    panel_width = max(min(WIDTH * 0.8, 800), min_panel_width)
    
    # Panel position
    panel_left = (WIDTH - panel_width) // 2
    panel_top = instructions_top + instructions_height + PANEL_MARGIN
    
    # Button positions
    button_width = 200
    button_height = 40
    button_spacing = 20
    button_top = panel_top + panel_height + PANEL_MARGIN * 2
    
    # Calculate horizontal button positions
    left_button_left = (WIDTH - 2 * button_width - button_spacing) // 2
    right_button_left = left_button_left + button_width + button_spacing
    
    # Update buttons
    calc_button = pygame.Rect(left_button_left, button_top, button_width, button_height)
    reset_button = pygame.Rect(right_button_left, button_top, button_width, button_height)
    
    # Special layout calculations based on current state
    if state == GOODNESS_OF_FIT:
        # Calculate observed slider positions with better spacing
        left_column = panel_left + 100
        right_column = panel_left + panel_width//2 + 100
        
        # Category label column 
        category_column = panel_left + 40
        
        # Update observed sliders
        for i, slider in enumerate(sliders_gof_observed):
            y_pos = panel_top + 100 + i * 40
            slider["pos"] = (left_column + 80, y_pos)  # Moved right to align with labels
            slider["width"] = SLIDER_MIN_WIDTH
        
        # Update expected sliders
        for i, slider in enumerate(sliders_gof_expected):
            y_pos = panel_top + 100 + i * 40
            slider["pos"] = (right_column + 80, y_pos)  # Moved right to align with labels
            slider["width"] = SLIDER_MIN_WIDTH
    
    elif state == INDEPENDENCE:
        # Calculate cell positions for Independence table with fixed spacing
        table_width = CELL_WIDTH * (len(independence_colnames) + 1)
        table_left = panel_left + (panel_width - table_width) // 2
        table_top = panel_top + 80  # Position table below panel title
        
        # Update cell positions
        for i in range(len(independence_cells)):
            for j in range(len(independence_cells[i])):
                cell_rect = pygame.Rect(
                    table_left + (j + 1) * CELL_WIDTH,  # +1 to skip row header column
                    table_top + (i + 1) * CELL_HEIGHT,  # +1 to skip column header row
                    CELL_WIDTH, 
                    CELL_HEIGHT
                )
                independence_cells[i][j]["rect"] = cell_rect
    
    # Return layout parameters
    return {
        "header_height": header_height,
        "chart": {
            "left": chart_left,
            "top": chart_top,
            "width": chart_width,
            "height": chart_height
        },
        "results": {
            "left": chart_left,
            "top": results_panel_top,
            "width": chart_width,
            "height": results_panel_height
        },
        "instructions": {
            "left": panel_left,
            "top": instructions_top,
            "width": panel_width,
            "height": instructions_height
        },
        "panel": {
            "left": panel_left,
            "top": panel_top,
            "width": panel_width,
            "height": panel_height
        },
        "toggle_rect": pygame.Rect(
            panel_left, 
            instructions_top, 
            30, 30
        ),
        "buttons": {
            "top": button_top,
            "height": button_height
        },
        "category_column": panel_left + 40,
        "table": {
            "left": panel_left + (panel_width - CELL_WIDTH * (len(independence_colnames) + 1)) // 2,
            "top": panel_top + 80,
            "cell_width": CELL_WIDTH,
            "cell_height": CELL_HEIGHT
        }
    }

def draw_panel(rect, color, border_radius=10, border_color=None, border_width=0):
    """Draw a panel with optional border"""
    pygame.draw.rect(screen, color, rect, border_radius=border_radius)
    if border_color:
        pygame.draw.rect(screen, border_color, rect, width=border_width, border_radius=border_radius)

def draw_text(text, font, color, x, y, align="left"):
    """Draw text on screen with alignment options"""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    
    if align == "center":
        text_rect.center = (x, y)
    elif align == "right":
        text_rect.right = x
        text_rect.centery = y
    else:  # left align
        text_rect.left = x
        text_rect.centery = y
        
    screen.blit(text_surface, text_rect)
    return text_rect

def draw_slider(slider, label, mouse_pos, value_format="{:.1f}", value_suffix=""):
    """Draw a slider with label and current value"""
    # Create hover area for slider track
    slider_rect = pygame.Rect(slider["pos"][0] - 5, slider["pos"][1] - 10, slider["width"] + 10, 20)
    
    # Check if mouse is hovering over this slider
    is_hovering = slider_rect.collidepoint(mouse_pos)
    
    # Draw slider track with different color when hovered
    track_color = BLUE if is_hovering or slider["dragging"] else GRAY
    pygame.draw.rect(screen, track_color, (slider["pos"][0], slider["pos"][1] - 5, slider["width"], 10))
    
    # Calculate handle position
    handle_pos = slider["pos"][0] + (slider["value"] - slider["min"]) / (slider["max"] - slider["min"]) * slider["width"]
    
    # Draw handle - larger when hovered or dragging
    handle_radius = 12 if is_hovering or slider["dragging"] else 10
    handle_color = DARK_BLUE if is_hovering or slider["dragging"] else BLUE
    pygame.draw.circle(screen, handle_color, (int(handle_pos), slider["pos"][1]), handle_radius)
    
    # Draw label with fixed position and right alignment
    if label:
        label_x = slider["pos"][0] - LABEL_SLIDER_GAP
        draw_text(label, font_medium, BLACK, label_x, slider["pos"][1], "right")
    
    # Draw value with fixed position and left alignment
    value_x = slider["pos"][0] + slider["width"] + SLIDER_VALUE_GAP
    draw_text(value_format.format(slider["value"]) + value_suffix, font_medium, BLACK, value_x, slider["pos"][1], "left")
    
    return is_hovering

def draw_editable_cell(rect, value, active=False, align="center"):
    """Draw an editable cell for the contingency table"""
    # Draw cell background
    cell_color = LIGHT_BLUE if active else WHITE
    pygame.draw.rect(screen, cell_color, rect)
    pygame.draw.rect(screen, BLACK, rect, width=1)
    
    # Draw value
    value_text = str(int(value)) if isinstance(value, (int, float)) else str(value)
    draw_text(value_text, font_medium, BLACK, rect.centerx, rect.centery, align)

def draw_instructions_toggle(toggle_rect):
    """Draw a toggle button for instructions"""
    pygame.draw.rect(screen, LIGHT_BLUE if show_instructions else LIGHT_GRAY, toggle_rect)
    pygame.draw.rect(screen, BLACK, toggle_rect, width=2)
    draw_text("i", font_medium, BLACK, toggle_rect.centerx, toggle_rect.centery, "center")
    return toggle_rect

def draw_menu():
    """Draw the main menu screen"""
    screen.fill(WHITE)
    
    # Draw title
    title_rect = draw_text("Chi-Square Test Visualizer", font_title, BLUE, WIDTH // 2, HEIGHT // 4, "center")
    
    # Draw subtitle
    subtitle_rect = draw_text("Statistical Tests for Alzheimer's Research", font_large, 
                             BLACK, WIDTH // 2, title_rect.bottom + 30, "center")
    
    # Draw description
    description = [
        "Understand how chi-square tests work in clinical research",
        "Perfect for Alzheimer's disease studies and genetic analysis",
        "Explore two common types of chi-square tests:"
    ]
    
    for i, line in enumerate(description):
        draw_text(line, font_medium, BLACK, WIDTH // 2, subtitle_rect.bottom + 50 + i * 40, "center")
    
    # Draw buttons
    gof_button = pygame.Rect(WIDTH // 2 - 200, HEIGHT // 2 + 100, 400, 60)
    ind_button = pygame.Rect(WIDTH // 2 - 200, HEIGHT // 2 + 180, 400, 60)
    tutorial_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 260, 300, 60)
    
    pygame.draw.rect(screen, LIGHT_BLUE, gof_button, border_radius=10)
    pygame.draw.rect(screen, LIGHT_RED, ind_button, border_radius=10)
    pygame.draw.rect(screen, LIGHT_GREEN, tutorial_button, border_radius=10)
    
    draw_text("Chi-Square Goodness of Fit Test", font_large, BLACK, gof_button.centerx, gof_button.centery, "center")
    draw_text("Chi-Square Test of Independence", font_large, BLACK, ind_button.centerx, ind_button.centery, "center")
    draw_text("Tutorial", font_large, BLACK, tutorial_button.centerx, tutorial_button.centery, "center")
    
    # Footer
    draw_text("Created for Alzheimer's Research Statistical Education", font_small, GRAY, 
             WIDTH // 2, HEIGHT - 50, "center")
    
    return gof_button, ind_button, tutorial_button

def draw_goodness_of_fit(layout, mouse_pos):
    """Draw the Goodness of Fit test screen"""
    screen.fill(WHITE)
    
    # Draw header with blue background to match Image 1
    header_bg = pygame.Rect(0, 0, WIDTH, layout["header_height"])
    pygame.draw.rect(screen, PANEL_BLUE, header_bg)
    draw_text("Chi-Square Goodness of Fit Test", font_title, BLUE, WIDTH // 2, layout["header_height"] // 2, "center")
    
    # Draw back button
    back_button = pygame.Rect(30, 10, 100, 40)
    pygame.draw.rect(screen, LIGHT_RED, back_button, border_radius=5)
    draw_text("Menu", font_small, BLACK, back_button.centerx, back_button.centery, "center")
    
    # Draw results panel with proper styling to match Image 1
    if has_calculated:
        result_panel_color = PANEL_GREEN if p_value < 0.05 else PANEL_RED
        result_text_color = DARK_GREEN if p_value < 0.05 else RED
        result_text = "Significant Difference!" if p_value < 0.05 else "No Significant Difference"
        
        result_panel = pygame.Rect(
            layout["results"]["left"], 
            layout["results"]["top"], 
            layout["results"]["width"], 
            layout["results"]["height"]
        )
        draw_panel(result_panel, PANEL_GREEN, border_radius=5)  # Use green as in Image 1
        
        # Divide the panel into thirds for the three values
        panel_width = layout["results"]["width"]
        panel_third = panel_width // 3
        
        # Format results to match Image 1
        chi2_pos_x = result_panel.left + panel_third // 2
        draw_text(f"χ² = {chi2_stat:.3f}", font_result, BLACK, chi2_pos_x, result_panel.centery, "center")
        
        p_pos_x = result_panel.left + panel_width // 2
        draw_text(f"p-value = {p_value:.4f}", font_result, BLACK, p_pos_x, result_panel.centery, "center")
        
        # Show df value and significance text in the right third
        df_pos_x = result_panel.left + 2 * panel_third + panel_third // 2
        draw_text(f"df = {df}", font_result, BLACK, df_pos_x - 40, result_panel.centery, "left")
        draw_text(result_text, font_result, result_text_color, df_pos_x + 40, result_panel.centery, "right")
    
    # Create and draw chart with border
    chart_rect = pygame.Rect(
        layout["chart"]["left"], 
        layout["chart"]["top"], 
        layout["chart"]["width"], 
        layout["chart"]["height"]
    )
    draw_panel(chart_rect, WHITE, border_color=GRAY, border_width=2)
    
    try:
        hist_surface = create_goodness_of_fit_plot(
            width=layout["chart"]["width"] - 20, 
            height=layout["chart"]["height"] - 20
        )
        screen.blit(hist_surface, (layout["chart"]["left"] + 10, layout["chart"]["top"] + 10))
    except Exception as e:
        print(f"Error creating chart: {e}")
        pygame.draw.rect(screen, LIGHT_BLUE, (
            layout["chart"]["left"] + 10, 
            layout["chart"]["top"] + 10, 
            layout["chart"]["width"] - 20, 
            layout["chart"]["height"] - 20
        ))
        draw_text("Error creating chart. Try adjusting parameters.", 
                 font_medium, RED, WIDTH // 2, layout["chart"]["top"] + layout["chart"]["height"] // 2, "center")
    
    # Instructions toggle button
    toggle_rect = draw_instructions_toggle(layout["toggle_rect"])
    
    # Draw instructions if enabled
    if show_instructions:
        instruction_panel = pygame.Rect(
            layout["instructions"]["left"], 
            layout["instructions"]["top"], 
            layout["instructions"]["width"], 
            layout["instructions"]["height"]
        )
        draw_panel(instruction_panel, PANEL_YELLOW, border_color=DARK_GRAY, border_width=1)
        instruction_text = "Adjust observed counts and expected percentages to test if data fits the expected distribution."
        draw_text(instruction_text, font_medium, BLACK, instruction_panel.centerx, instruction_panel.centery, "center")
    
    # Draw data entry panel with light gray background as in Image 1
    panel = pygame.Rect(
        layout["panel"]["left"], 
        layout["panel"]["top"], 
        layout["panel"]["width"], 
        layout["panel"]["height"]
    )
    draw_panel(panel, LIGHT_GRAY, border_color=DARK_GRAY, border_width=1)
    
    # Panel title
    panel_title_y = panel.top + 20
    draw_text("Adjust Data", font_panel_title, BLACK, panel.centerx, panel_title_y, "center")
    
    # Column headers
    col1_x = panel.left + panel.width // 4
    col2_x = panel.left + 3 * panel.width // 4
    header_y = panel.top + 60
    
    draw_text("Observed Counts", font_medium, BLUE, col1_x, header_y, "center")
    draw_text("Expected Percentages", font_medium, RED, col2_x, header_y, "center")
    
    # Draw category labels and sliders
    category_left = layout["category_column"]
    for i, category in enumerate(gof_categories):
        cat_y = panel.top + 100 + i * 40
        draw_text(category, font_medium, BLACK, category_left, cat_y, "left")
        
        # Draw observed sliders
        draw_slider(sliders_gof_observed[i], "", mouse_pos, "{:.0f}")
        
        # Draw expected sliders
        draw_slider(sliders_gof_expected[i], "", mouse_pos, "{:.1f}", "%")
    
    # Draw buttons to match Image 1
    pygame.draw.rect(screen, GRAY if has_calculated else LIGHT_GRAY, calc_button, border_radius=5)
    draw_text("Calculate Chi-Square", font_medium, BLACK, calc_button.centerx, calc_button.centery, "center")
    
    pygame.draw.rect(screen, LIGHT_BLUE, reset_button, border_radius=5)
    draw_text("Reset Data", font_medium, BLACK, reset_button.centerx, reset_button.centery, "center")
    
    return back_button, toggle_rect

def draw_independence(layout, mouse_pos):
    """Draw the Test of Independence screen to match Image 2"""
    screen.fill(WHITE)
    
    # Draw header with red background to match Image 2
    header_bg = pygame.Rect(0, 0, WIDTH, layout["header_height"])
    pygame.draw.rect(screen, PANEL_RED, header_bg)
    draw_text("Chi-Square Test of Independence", font_title, RED, WIDTH // 2, layout["header_height"] // 2, "center")
    
    # Draw back button
    back_button = pygame.Rect(30, 10, 100, 40)
    pygame.draw.rect(screen, LIGHT_RED, back_button, border_radius=5)
    draw_text("Menu", font_small, BLACK, back_button.centerx, back_button.centery, "center")
    
    # Draw results panel
    if has_calculated:
        result_panel_color = PANEL_GREEN if p_value < 0.05 else PANEL_RED
        result_text_color = DARK_GREEN if p_value < 0.05 else RED
        result_text = "Significant Association!" if p_value < 0.05 else "No Significant Association"
        
        result_panel = pygame.Rect(
            layout["results"]["left"], 
            layout["results"]["top"], 
            layout["results"]["width"], 
            layout["results"]["height"]
        )
        draw_panel(result_panel, result_panel_color, border_radius=5)
        
        panel_width = layout["results"]["width"]
        chi2_pos_x = result_panel.left + 10
        draw_text(f"χ² = {chi2_stat:.3f}", font_result, BLACK, chi2_pos_x, result_panel.centery)
        
        p_pos_x = result_panel.left + panel_width // 3
        draw_text(f"p-value = {p_value:.4f}", font_result, BLACK, p_pos_x, result_panel.centery)
        
        df_pos_x = result_panel.left + 2 * panel_width // 3
        draw_text(f"df = {df}", font_result, BLACK, df_pos_x, result_panel.centery)
        
        # Draw significance result
        draw_text(result_text, font_result, result_text_color, result_panel.right - 10, result_panel.centery, "right")
    
    # Create and draw chart with border
    chart_rect = pygame.Rect(
        layout["chart"]["left"], 
        layout["chart"]["top"], 
        layout["chart"]["width"], 
        layout["chart"]["height"]
    )
    draw_panel(chart_rect, WHITE, border_color=GRAY, border_width=2)
    
    try:
        chart_surface = create_independence_plot(
            width=layout["chart"]["width"] - 20, 
            height=layout["chart"]["height"] - 20
        )
        screen.blit(chart_surface, (layout["chart"]["left"] + 10, layout["chart"]["top"] + 10))
    except Exception as e:
        print(f"Error creating chart: {e}")
        pygame.draw.rect(screen, LIGHT_RED, (
            layout["chart"]["left"] + 10, 
            layout["chart"]["top"] + 10, 
            layout["chart"]["width"] - 20, 
            layout["chart"]["height"] - 20
        ))
        draw_text("Error creating chart. Try adjusting parameters.", 
                 font_medium, RED, WIDTH // 2, layout["chart"]["top"] + layout["chart"]["height"] // 2, "center")
    
    # Instructions toggle button
    toggle_rect = draw_instructions_toggle(layout["toggle_rect"])
    
    # Draw instructions if enabled - pale yellow background as in screenshots
    if show_instructions:
        instruction_panel = pygame.Rect(
            layout["instructions"]["left"], 
            layout["instructions"]["top"], 
            layout["instructions"]["width"], 
            layout["instructions"]["height"]
        )
        draw_panel(instruction_panel, PANEL_YELLOW, border_color=DARK_GRAY, border_width=1)
        instruction_text = "Adjust the contingency table values to test if two variables are associated."
        draw_text(instruction_text, font_medium, BLACK, instruction_panel.centerx, instruction_panel.centery, "center")
    
    # Draw contingency table panel
    panel = pygame.Rect(
        layout["panel"]["left"], 
        layout["panel"]["top"], 
        layout["panel"]["width"], 
        layout["panel"]["height"]
    )
    draw_panel(panel, LIGHT_GRAY, border_color=DARK_GRAY, border_width=1)
    
    # Panel title
    panel_title_y = panel.top + 20
    draw_text("Contingency Table", font_panel_title, BLACK, panel.centerx, panel_title_y, "center")
    
    # Calculate table dimensions
    table_left = layout["table"]["left"]
    table_top = layout["table"]["top"]
    cell_width = layout["table"]["cell_width"]
    cell_height = layout["table"]["cell_height"]
    
    # Draw column headers - light blue background as in Image 2
    for j, col_name in enumerate(independence_colnames):
        cell_rect = pygame.Rect(
            table_left + (j + 1) * cell_width, 
            table_top, 
            cell_width, 
            cell_height
        )
        pygame.draw.rect(screen, LIGHT_BLUE, cell_rect)
        pygame.draw.rect(screen, BLACK, cell_rect, width=1)
        draw_text(col_name, font_small, BLACK, cell_rect.centerx, cell_rect.centery, "center")
    
    # Draw row headers - light blue background as in Image 2
    for i, row_name in enumerate(independence_rownames):
        cell_rect = pygame.Rect(
            table_left, 
            table_top + (i + 1) * cell_height, 
            cell_width, 
            cell_height
        )
        pygame.draw.rect(screen, LIGHT_BLUE, cell_rect)
        pygame.draw.rect(screen, BLACK, cell_rect, width=1)
        draw_text(row_name, font_small, BLACK, cell_rect.centerx, cell_rect.centery, "center")
    
    # Draw the empty top-left cell
    corner_rect = pygame.Rect(
        table_left, 
        table_top, 
        cell_width, 
        cell_height
    )
    pygame.draw.rect(screen, LIGHT_GRAY, corner_rect)
    pygame.draw.rect(screen, BLACK, corner_rect, width=1)
    
    # Draw data cells
    for i in range(len(independence_cells)):
        for j in range(len(independence_cells[i])):
            cell = independence_cells[i][j]
            draw_editable_cell(cell["rect"], cell["value"], cell["active"])
    
    # Draw buttons - bright green for Calculate as in Image 2
    pygame.draw.rect(screen, NEON_GREEN, calc_button, border_radius=5)
    draw_text("Calculate Chi-Square", font_medium, BLACK, calc_button.centerx, calc_button.centery, "center")
    
    pygame.draw.rect(screen, LIGHT_BLUE, reset_button, border_radius=5)
    draw_text("Reset Data", font_medium, BLACK, reset_button.centerx, reset_button.centery, "center")
    
    return back_button, toggle_rect

def draw_tutorial():
    """Draw tutorial screens based on current page"""
    screen.fill(WHITE)
    
    # Draw header
    draw_text("Chi-Square Test Tutorial", font_title, BLUE, WIDTH // 2, 50, "center")
    
    if tutorial_page == 0:
        # Introduction to Chi-square tests
        draw_text("What are Chi-Square Tests?", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "Chi-square tests are statistical methods used to determine if there is a",
            "significant association between categorical variables.",
            "",
            "In Alzheimer's research, they are useful for analyzing:",
            "• Genetic marker frequencies",
            "• Treatment response rates",
            "• Demographic distributions",
            "• Risk factor associations",
            "",
            "This simulation helps you understand how chi-square tests work",
            "and how to interpret their results in research contexts."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 1:
        # Goodness of Fit test
        draw_text("Chi-Square Goodness of Fit Test", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "The Goodness of Fit test compares observed frequencies to expected frequencies",
            "to determine if a sample follows an expected distribution.",
            "",
            "Applications in Alzheimer's research:",
            "• Testing if genetic markers follow Hardy-Weinberg equilibrium",
            "• Comparing observed patient demographics to population statistics",
            "• Analyzing if symptom frequencies match expected disease patterns",
            "",
            "In the simulation, you can adjust observed counts and expected percentages",
            "to see how discrepancies affect statistical significance."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 2:
        # Test of Independence
        draw_text("Chi-Square Test of Independence", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "The Test of Independence evaluates whether two categorical variables",
            "are related or independent of each other.",
            "",
            "Applications in Alzheimer's research:",
            "• Testing if treatment response varies by genotype (like APOE ε4 status)",
            "• Analyzing if symptom presence correlates with demographic factors",
            "• Determining associations between comorbidities and disease severity",
            "",
            "In the simulation, you can modify a contingency table to see how",
            "different patterns of association affect statistical significance."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 3:
        # Interpreting results
        draw_text("Interpreting Chi-Square Results", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "The key outputs of a chi-square test are:",
            "",
            "• Chi-square statistic (χ²): Higher values indicate greater differences",
            "  between observed and expected frequencies",
            "",
            "• p-value: The probability of observing the data by chance",
            "  p < 0.05 is typically considered statistically significant",
            "",
            "• Degrees of freedom (df): Depends on the number of categories",
            "  For Goodness of Fit: df = (categories - 1)",
            "  For Independence: df = (rows - 1) × (columns - 1)",
            "",
            "Remember: Statistical significance does not always imply clinical relevance!"
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    # Navigation buttons
    back_button = pygame.Rect(50, HEIGHT - 100, 200, 50)
    next_button = pygame.Rect(WIDTH - 250, HEIGHT - 100, 200, 50)
    menu_button = pygame.Rect(30, 10, 100, 40)
    
    pygame.draw.rect(screen, LIGHT_RED, menu_button, border_radius=5)
    pygame.draw.rect(screen, LIGHT_BLUE, back_button, border_radius=5)
    pygame.draw.rect(screen, LIGHT_BLUE, next_button, border_radius=5)
    
    draw_text("Menu", font_small, BLACK, menu_button.centerx, menu_button.centery, "center")
    
    if tutorial_page > 0:
        draw_text("Previous", font_medium, BLACK, back_button.centerx, back_button.centery, "center")
    
    if tutorial_page < num_tutorial_pages - 1:
        draw_text("Next", font_medium, BLACK, next_button.centerx, next_button.centery, "center")
    else:
        draw_text("Start Simulation", font_medium, BLACK, next_button.centerx, next_button.centery, "center")
    
    # Page indicator
    draw_text(f"Page {tutorial_page + 1}/{num_tutorial_pages}", font_small, BLACK, WIDTH // 2, HEIGHT - 75, "center")
    
    return menu_button, back_button, next_button

def reset_goodness_of_fit():
    """Reset Goodness of Fit test data"""
    global has_calculated
    
    # Reset to default values
    for i, slider in enumerate(sliders_gof_observed):
        slider["value"] = gof_observed[i]
    
    for i, slider in enumerate(sliders_gof_expected):
        slider["value"] = gof_expected[i]
    
    has_calculated = False

def reset_independence():
    """Reset Independence test data"""
    global has_calculated
    
    # Reset to default values
    for i in range(len(independence_cells)):
        for j in range(len(independence_cells[i])):
            independence_cells[i][j]["value"] = independence_data[i, j]
            independence_cells[i][j]["active"] = False
    
    # Reset active cell
    global active_cell
    active_cell = None
    
    has_calculated = False

# Initialize layout
layout = calculate_layout()

# Main game loop
running = True
clock = pygame.time.Clock()

# Calculate minimum height based on component heights
def calculate_min_height():
    """Calculate minimum required height for all UI elements"""
    header_height = HEADER_HEIGHT
    results_height = 40
    chart_min_height = 300
    instructions_height = 40
    panel_min_height = 250
    button_min_height = 40
    
    min_margins = PANEL_MARGIN * 5
    
    min_required_height = (header_height + RESULT_MARGIN + 
                          results_height + PANEL_MARGIN + 
                          chart_min_height + PANEL_MARGIN + 
                          instructions_height + PANEL_MARGIN + 
                          panel_min_height + PANEL_MARGIN*3 + 
                          button_min_height + PANEL_MARGIN)
    
    return max(min_required_height, 700)

while running:
    mouse_pos = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.VIDEORESIZE:
            # Calculate minimum dimensions
            calculated_min_width = 800
            calculated_min_height = calculate_min_height()
            
            # Handle window resizing with dynamic minimum dimensions
            WIDTH = max(event.w, calculated_min_width, MIN_WIDTH)
            HEIGHT = max(event.h, calculated_min_height)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            
            # Recalculate layout with new dimensions
            layout = calculate_layout()
            
            # Prevent any current slider dragging operations from continuing
            for slider in sliders_gof_observed + sliders_gof_expected:
                slider["dragging"] = False
            
            # Reset active cell
            active_cell = None
            for row in independence_cells:
                for cell in row:
                    cell["active"] = False
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # Return to menu when escape is pressed
                if state != MENU:
                    state = MENU
                    has_calculated = False
            
            # Handle number input for active cell in Independence test
            if active_cell and state == INDEPENDENCE:
                if event.key == pygame.K_BACKSPACE:
                    # Handle backspace to delete last digit
                    value_str = str(int(active_cell["value"]))
                    if len(value_str) > 1:
                        active_cell["value"] = int(value_str[:-1])
                    else:
                        active_cell["value"] = 0
                    has_calculated = False
                    
                elif event.key >= pygame.K_0 and event.key <= pygame.K_9:
                    # Handle number keys
                    digit = event.key - pygame.K_0
                    current_val = int(active_cell["value"])
                    
                    # Append digit or replace if current value is 0
                    if current_val == 0:
                        active_cell["value"] = digit
                    else:
                        # Limit to 3 digits max
                        if current_val < 100:
                            active_cell["value"] = current_val * 10 + digit
                    
                    has_calculated = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                
                if state == MENU:
                    # Check for button clicks in menu
                    gof_button, ind_button, tutorial_button = draw_menu()
                    
                    if gof_button.collidepoint(mouse_pos):
                        state = GOODNESS_OF_FIT
                        reset_goodness_of_fit()
                    elif ind_button.collidepoint(mouse_pos):
                        state = INDEPENDENCE
                        reset_independence()
                    elif tutorial_button.collidepoint(mouse_pos):
                        state = TUTORIAL
                        tutorial_page = 0
                
                elif state == TUTORIAL:
                    # Check for button clicks in tutorial
                    menu_button, back_button, next_button = draw_tutorial()
                    
                    if menu_button.collidepoint(mouse_pos):
                        state = MENU
                    elif back_button.collidepoint(mouse_pos) and tutorial_page > 0:
                        tutorial_page -= 1
                    elif next_button.collidepoint(mouse_pos):
                        if tutorial_page < num_tutorial_pages - 1:
                            tutorial_page += 1
                        else:
                            state = MENU
                
                elif state == GOODNESS_OF_FIT:
                    # Get UI elements from the draw function
                    back_button, toggle_rect = draw_goodness_of_fit(layout, mouse_pos)
                    
                    # Check for instruction toggle
                    if toggle_rect.collidepoint(mouse_pos):
                        show_instructions = not show_instructions
                        
                    # Check for button clicks
                    if back_button.collidepoint(mouse_pos):
                        state = MENU
                    elif calc_button.collidepoint(mouse_pos) and not has_calculated:
                        calculate_goodness_of_fit()
                    elif reset_button.collidepoint(mouse_pos):
                        reset_goodness_of_fit()
                    
                    # Check for slider interactions
                    for slider in sliders_gof_observed + sliders_gof_expected:
                        slider_rect = pygame.Rect(
                            slider["pos"][0] - 5, 
                            slider["pos"][1] - 10, 
                            slider["width"] + 10, 
                            20
                        )
                        if slider_rect.collidepoint(mouse_pos):
                            slider["dragging"] = True
                            # Update the value immediately based on click position
                            new_val = (mouse_pos[0] - slider["pos"][0]) / slider["width"] * (slider["max"] - slider["min"]) + slider["min"]
                            slider["value"] = max(slider["min"], min(slider["max"], new_val))
                            has_calculated = False
                
                elif state == INDEPENDENCE:
                    # Get UI elements from the draw function
                    back_button, toggle_rect = draw_independence(layout, mouse_pos)
                    
                    # Check for instruction toggle
                    if toggle_rect.collidepoint(mouse_pos):
                        show_instructions = not show_instructions
                        
                    # Check for button clicks
                    if back_button.collidepoint(mouse_pos):
                        state = MENU
                    elif calc_button.collidepoint(mouse_pos) and not has_calculated:
                        calculate_independence()
                    elif reset_button.collidepoint(mouse_pos):
                        reset_independence()
                    
                    # Reset active cell status
                    active_cell = None
                    for row in independence_cells:
                        for cell in row:
                            cell["active"] = False
                    
                    # Check for cell clicks
                    for i in range(len(independence_cells)):
                        for j in range(len(independence_cells[i])):
                            cell = independence_cells[i][j]
                            if cell["rect"].collidepoint(mouse_pos):
                                cell["active"] = True
                                active_cell = cell
                                has_calculated = False
        
        elif event.type == pygame.MOUSEBUTTONUP:
            # Stop dragging all sliders
            for slider in sliders_gof_observed + sliders_gof_expected:
                slider["dragging"] = False
        
        elif event.type == pygame.MOUSEMOTION:
            # Update sliders
            all_changed = False
            
            # Goodness of Fit sliders
            for slider in sliders_gof_observed + sliders_gof_expected:
                if slider["dragging"]:
                    # Calculate new value based on mouse position
                    new_val = (mouse_pos[0] - slider["pos"][0]) / slider["width"] * (slider["max"] - slider["min"]) + slider["min"]
                    # Clamp value to slider range
                    slider["value"] = max(slider["min"], min(slider["max"], new_val))
                    all_changed = True
            
            if all_changed:
                has_calculated = False
    
    # Draw current state
    if state == MENU:
        draw_menu()
    elif state == GOODNESS_OF_FIT:
        draw_goodness_of_fit(layout, mouse_pos)
    elif state == INDEPENDENCE:
        draw_independence(layout, mouse_pos)
    elif state == TUTORIAL:
        draw_tutorial()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()