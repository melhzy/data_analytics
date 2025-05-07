# Chi-Square Distribution Visualizer
# Educational tool for understanding Chi-square tests in Alzheimer's research
import pygame
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Required for pygame
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from scipy import stats
import os
import sys
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
TRANSPARENT_WHITE = (255, 255, 255, 180)  # Semi-transparent white

# Colors for matplotlib (normalized RGB format)
MPL_COLORS = [
    (0/255, 0/255, 255/255),      # Blue (k=1)
    (0/255, 255/255, 0/255),      # Green (k=2)
    (255/255, 0/255, 0/255),      # Red (k=3)
    (0/255, 255/255, 255/255),    # Cyan (k=4)
    (255/255, 0/255, 255/255),    # Magenta (k=5)
]

# Layout constants
HEADER_HEIGHT = 60
TITLE_MARGIN = 10
RESULT_MARGIN = 20
PANEL_MARGIN = 15
H_MARGIN = 20
V_MARGIN = 10
SLIDER_SPACING = 40
ROW_SPACING = 40

# Grid layout constants
LABEL_COLUMN_WIDTH = 160
VALUE_COLUMN_WIDTH = 50
SLIDER_MIN_WIDTH = 150
LABEL_SLIDER_GAP = 20
SLIDER_VALUE_GAP = 16

# Screen dimensions
MIN_WIDTH, MIN_HEIGHT = 1000, 700
initial_width, initial_height = 1200, 800
WIDTH, HEIGHT = initial_width, initial_height
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Chi-Square Distribution Visualizer for Alzheimer's Research")

# Fonts
font_small = pygame.font.SysFont('Arial', 16)
font_medium = pygame.font.SysFont('Arial', 20)
font_large = pygame.font.SysFont('Arial', 24)
font_title = pygame.font.SysFont('Arial', 40, bold=True)
font_result = pygame.font.SysFont('Arial', 22, bold=True)
font_panel_title = pygame.font.SysFont('Arial', 22, bold=True)

# Game states
MENU = 0
GAME = 1
TUTORIAL = 2

# Initial state
state = MENU

# Default parameters
degrees_of_freedom = 3
alpha_level = 0.05
show_critical = True
max_x_value = 20
num_distributions = 5

# Initialize sliders
slider_df = {"pos": (0, 0), "width": 200, "value": degrees_of_freedom, "min": 1, "max": 30, "dragging": False}
slider_alpha = {"pos": (0, 0), "width": 200, "value": alpha_level, "min": 0.01, "max": 0.2, "dragging": False}
slider_max_x = {"pos": (0, 0), "width": 200, "value": max_x_value, "min": 5, "max": 40, "dragging": False}
slider_num_dist = {"pos": (0, 0), "width": 200, "value": num_distributions, "min": 1, "max": 5, "dragging": False}

# Initialize buttons with empty Rects
update_button = pygame.Rect(0, 0, 0, 0)
toggle_critical_button = pygame.Rect(0, 0, 0, 0)

# Critical values and results
critical_value = stats.chi2.ppf(1 - alpha_level, degrees_of_freedom)
has_updated = False

# Tutorial pages
tutorial_page = 0
num_tutorial_pages = 5

# Toggleable instruction section
show_instructions = True

def calculate_layout():
    """Calculate all UI component positions based on current window size"""
    global slider_df, slider_alpha, slider_max_x, slider_num_dist
    global update_button, toggle_critical_button
    
    # Calculate component heights
    header_height = HEADER_HEIGHT
    results_panel_height = 40
    chart_ratio = 0.5
    instructions_height = 40
    panel_height = 180
    button_height = 40
    
    # Calculate available height after header
    available_height = HEIGHT - header_height - PANEL_MARGIN
    
    # Results panel position
    results_panel_top = header_height + RESULT_MARGIN
    
    # Calculate remaining height after results panel
    remaining_height = available_height - results_panel_height - RESULT_MARGIN
    
    # Chart dimensions - responsive to window size
    chart_height = min(remaining_height * chart_ratio, 450)
    chart_width = min(WIDTH * 0.7, 800)
    
    # Chart position - centered horizontally, below results panel
    chart_left = (WIDTH - chart_width) // 2
    chart_top = results_panel_top + results_panel_height + PANEL_MARGIN
    
    # Instructions panel - placed below the chart
    instructions_top = chart_top + chart_height + PANEL_MARGIN
    
    # Calculate remaining height for panels and buttons
    remaining_height_after_chart = HEIGHT - instructions_top - instructions_height - PANEL_MARGIN
    
    # Determine if we need to reduce panel heights to fit everything
    total_needed_height = panel_height + PANEL_MARGIN*3 + button_height
    
    # Adjust panel heights if needed to fit everything
    if total_needed_height > remaining_height_after_chart:
        scale_factor = remaining_height_after_chart / total_needed_height
        panel_height = max(120, int(panel_height * scale_factor))
    
    # Panel dimensions - responsive to window size with minimum widths
    min_panel_width = LABEL_COLUMN_WIDTH + SLIDER_MIN_WIDTH + VALUE_COLUMN_WIDTH + LABEL_SLIDER_GAP + SLIDER_VALUE_GAP + H_MARGIN*2
    panel_width = max(min(WIDTH * 0.8, 600), min_panel_width)
    
    # Panel positions with better distribution
    panel_left = (WIDTH - panel_width) // 2
    panel_top = instructions_top + instructions_height + PANEL_MARGIN
    
    # Define grid for panel with proper layout
    title_height = 40
    
    # Calculate available space for sliders after accounting for title
    available_slider_space = panel_height - title_height - 20
    
    # Distribute slider rows evenly in the available space
    slider_rows = 4  # Number of sliders
    row_height = available_slider_space / slider_rows
    
    panel_grid = {
        'label_col': panel_left + H_MARGIN,
        'label_width': LABEL_COLUMN_WIDTH,
        'slider_col': panel_left + H_MARGIN + LABEL_COLUMN_WIDTH + LABEL_SLIDER_GAP,
        'slider_width': panel_width - LABEL_COLUMN_WIDTH - VALUE_COLUMN_WIDTH - H_MARGIN*2 - LABEL_SLIDER_GAP - SLIDER_VALUE_GAP,
        'value_col': panel_left + panel_width - VALUE_COLUMN_WIDTH - H_MARGIN,
        'value_width': VALUE_COLUMN_WIDTH,
        'row1': panel_top + title_height + row_height * 0.5,
        'row2': panel_top + title_height + row_height * 1.5,
        'row3': panel_top + title_height + row_height * 2.5,
        'row4': panel_top + title_height + row_height * 3.5
    }
    
    # Update slider positions using the grid
    slider_df = {
        "pos": (panel_grid['slider_col'], panel_grid['row1']),
        "width": panel_grid['slider_width'],
        "value": slider_df["value"],
        "min": 1,
        "max": 30,
        "dragging": slider_df["dragging"]
    }
    
    slider_alpha = {
        "pos": (panel_grid['slider_col'], panel_grid['row2']),
        "width": panel_grid['slider_width'],
        "value": slider_alpha["value"],
        "min": 0.01,
        "max": 0.2,
        "dragging": slider_alpha["dragging"]
    }
    
    slider_max_x = {
        "pos": (panel_grid['slider_col'], panel_grid['row3']),
        "width": panel_grid['slider_width'],
        "value": slider_max_x["value"],
        "min": 5,
        "max": 40,
        "dragging": slider_max_x["dragging"]
    }
    
    slider_num_dist = {
        "pos": (panel_grid['slider_col'], panel_grid['row4']),
        "width": panel_grid['slider_width'],
        "value": slider_num_dist["value"],
        "min": 1,
        "max": 5,
        "dragging": slider_num_dist["dragging"]
    }
    
    # Calculate button positions
    button_width = 200
    button_top = panel_top + panel_height + PANEL_MARGIN
    
    # If buttons would go off screen, adjust their position
    if button_top + button_height + PANEL_MARGIN > HEIGHT:
        button_top = HEIGHT - button_height - PANEL_MARGIN
    
    # Calculate horizontal button positions
    left_button_left = (WIDTH - 2 * button_width - 20) // 2
    right_button_left = left_button_left + button_width + 20
    
    # Update buttons
    update_button = pygame.Rect(left_button_left, button_top, button_width, button_height)
    toggle_critical_button = pygame.Rect(right_button_left, button_top, button_width, button_height)
    
    # Return layout parameters for drawing functions
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
            "width": WIDTH - 2 * panel_left,
            "height": instructions_height
        },
        "panel": {
            "left": panel_left,
            "top": panel_top,
            "width": panel_width,
            "height": panel_height,
            "grid": panel_grid
        },
        "toggle_rect": pygame.Rect(
            panel_left, 
            instructions_top, 
            30, 30
        ),
        "buttons": {
            "top": button_top,
            "height": button_height
        }
    }

def debug_info(message):
    """Print debug information"""
    print(f"DEBUG: {message}")

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

def draw_slider(slider, label, grid_pos, mouse_pos, value_format="{:.1f}", value_suffix=""):
    """Draw a slider with label and current value using grid layout with hover effects"""
    # Create hover area for slider track
    slider_rect = pygame.Rect(slider["pos"][0] - 5, slider["pos"][1] - 10, slider["width"] + 10, 20)
    
    # Check if mouse is hovering over this slider
    is_hovering = slider_rect.collidepoint(mouse_pos)
    
    # Draw slider track with different color when hovered
    track_color = BLUE if is_hovering or slider["dragging"] else GRAY
    track_alpha = 200 if is_hovering or slider["dragging"] else 150
    
    pygame.draw.rect(screen, track_color, (slider["pos"][0], slider["pos"][1] - 5, slider["width"], 10))
    
    # Calculate handle position
    handle_pos = slider["pos"][0] + (slider["value"] - slider["min"]) / (slider["max"] - slider["min"]) * slider["width"]
    
    # Draw handle - larger when hovered or dragging
    handle_radius = 12 if is_hovering or slider["dragging"] else 10
    handle_color = DARK_BLUE if is_hovering or slider["dragging"] else BLUE
    pygame.draw.circle(screen, handle_color, (int(handle_pos), slider["pos"][1]), handle_radius)
    
    # Draw label with fixed position and right alignment
    label_x = slider["pos"][0] - LABEL_SLIDER_GAP
    draw_text(label, font_medium, BLACK, label_x, slider["pos"][1], "right")
    
    # Draw value with fixed position and left alignment
    value_x = slider["pos"][0] + slider["width"] + SLIDER_VALUE_GAP
    draw_text(value_format.format(slider["value"]) + value_suffix, font_medium, BLACK, value_x, slider["pos"][1], "left")
    
    return is_hovering

def add_hover_tooltip(rect, text, mouse_pos):
    """Show tooltip when hovering over an element"""
    # Check if rect is a tuple of (x, y) coordinates or a Rect object
    if isinstance(rect, tuple):
        # Create a small Rect around the position
        x, y = rect
        hover_area = pygame.Rect(x - 10, y - 10, 20, 20)
        is_hovering = hover_area.collidepoint(mouse_pos)
    else:
        # Use the rect directly if it's already a Rect object
        is_hovering = rect.collidepoint(mouse_pos)
        
    if is_hovering:
        tooltip_surface = font_small.render(text, True, BLACK, LIGHT_GRAY)
        tooltip_rect = tooltip_surface.get_rect(topleft=(mouse_pos[0] + 10, mouse_pos[1] + 10))
        
        # Ensure tooltip stays within screen bounds
        if tooltip_rect.right > WIDTH:
            tooltip_rect.right = WIDTH - 5
        if tooltip_rect.bottom > HEIGHT:
            tooltip_rect.bottom = HEIGHT - 5
            
        # Draw tooltip background
        pygame.draw.rect(screen, LIGHT_GRAY, tooltip_rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, tooltip_rect, width=1, border_radius=5)
        
        # Draw tooltip text
        screen.blit(tooltip_surface, tooltip_rect)
        return True
    return False

def create_chisquare_surface(df, alpha, max_x, num_distributions, show_critical, width=600, height=400):
    """Create a matplotlib figure with Chi-square distributions"""
    try:
        # Create figure with proper scaling
        dpi = 100
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        
        # Add more space for labels
        fig.subplots_adjust(top=0.9, right=0.9, bottom=0.15, left=0.1)
        
        ax = fig.add_subplot(111)
        
        # Create x values for plotting
        x = np.linspace(0.01, max_x, 1000)
        
        # Calculate critical value for the current df
        critical_value = stats.chi2.ppf(1 - alpha, df)
        
        # Plot the main Chi-square distribution
        y = stats.chi2.pdf(x, df)
        ax.plot(x, y, color=MPL_COLORS[0], linewidth=3, label=f'k={df}')
        
        # Plot critical value line
        if show_critical:
            ax.axvline(x=critical_value, color='black', linestyle='--', 
                      label=f'Critical value: {critical_value:.3f}')
            
            # Shade the rejection region
            rejection_x = x[x >= critical_value]
            rejection_y = stats.chi2.pdf(rejection_x, df)
            ax.fill_between(rejection_x, rejection_y, alpha=0.3, color='red',
                           label=f'α={alpha:.2f}')
        
        # Plot additional distributions
        for i in range(1, int(min(num_distributions, 5))):
            add_df = max(1, df - 2 + i * 2)  # Different degrees of freedom
            if add_df != df:  # Don't plot the same distribution twice
                y_add = stats.chi2.pdf(x, add_df)
                ax.plot(x, y_add, color=MPL_COLORS[i % len(MPL_COLORS)], 
                        linewidth=1.5, label=f'k={add_df}')
        
        ax.set_xlabel('χ² Value', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(f'Chi-square Distribution (df={df})', fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        
        # Set axis limits
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max(stats.chi2.pdf(x, df)) * 1.1)
        
        # Grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Convert matplotlib figure to pygame surface
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        plt.close(fig)
        
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        return surf
        
    except Exception as e:
        print(f"Error creating Chi-square plot: {e}")
        print(traceback.format_exc())
        
        # Create empty surface with error message
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111)
        
        error_message = f"Error creating Chi-square plot.\nTry different parameters."
        ax.text(0.5, 0.5, error_message, ha='center', va='center', color='red', fontsize=14)
        ax.axis('off')
        
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        plt.close(fig)
        
        surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
        return surf

def draw_tutorial():
    """Draw tutorial screens based on current page"""
    screen.fill(WHITE)
    
    # Header
    draw_text("Chi-Square Distribution Tutorial", font_title, BLUE, WIDTH // 2, 50, "center")
    
    if tutorial_page == 0:
        # Introduction
        draw_text("What is the Chi-square Distribution?", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "The Chi-square distribution is a probability distribution that is used extensively",
            "in statistical analysis, especially for tests that involve categorical data.",
            "",
            "Key applications in Alzheimer's research include:",
            "• Testing the association between genetic variants and disease",
            "• Analyzing differences in symptom frequencies between patient groups",
            "• Validating the goodness-of-fit of cognitive decline models",
            "• Comparing observed vs. expected frequencies in clinical trials",
            "",
            "This interactive simulation helps you understand how degrees of freedom,",
            "critical values, and significance levels affect statistical hypothesis testing."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 1:
        # Degrees of freedom
        draw_text("Degrees of Freedom", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "Degrees of freedom (df) is a key parameter of the Chi-square distribution.",
            "",
            "In Alzheimer's research, degrees of freedom typically correspond to:",
            "• (Number of categories - 1) in goodness-of-fit tests",
            "• (rows-1) × (columns-1) in contingency table analysis",
            "",
            "As degrees of freedom increase:",
            "• The distribution becomes more symmetric",
            "• The peak shifts to the right",
            "• The distribution approaches a normal distribution",
            "",
            "Try adjusting the degrees of freedom slider to see how it changes the distribution."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 2:
        # Significance level
        draw_text("Significance Level (Alpha)", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "The significance level (α) determines the threshold for rejecting the null hypothesis.",
            "",
            "In Chi-square tests for Alzheimer's studies:",
            "• Standard α values: 0.05, 0.01, or 0.001",
            "• α = 0.05 means a 5% risk of incorrectly rejecting the null hypothesis",
            "• Lower α values (e.g., 0.01) indicate stronger evidence is required",
            "",
            "The critical value partitions the Chi-square distribution:",
            "• Values above the critical threshold fall in the rejection region",
            "• This region represents the probability α of incorrectly rejecting a true null hypothesis",
            "",
            "Try adjusting the alpha slider to see how it affects the critical value and rejection region."
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 3:
        # Chi-square test types
        draw_text("Chi-square Tests in Alzheimer's Research", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "Common applications of Chi-square tests in Alzheimer's disease research:",
            "",
            "1. Chi-square Test of Independence",
            "   • Tests association between two categorical variables",
            "   • Example: Is APOE4 gene status related to early onset of symptoms?",
            "",
            "2. Chi-square Goodness-of-Fit Test",
            "   • Tests whether observed data matches expected frequencies",
            "   • Example: Do observed patterns of inheritance match Mendelian ratios?",
            "",
            "3. Chi-square Test of Homogeneity",
            "   • Tests if different populations have the same distribution of a categorical variable",
            "   • Example: Do treatment and control groups have similar rates of side effects?"
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    elif tutorial_page == 4:
        # Interpretation
        draw_text("Interpreting Chi-square Results", font_large, BLACK, WIDTH // 2, 120, "center")
        
        tutorial_text = [
            "Steps to interpret Chi-square test results:",
            "",
            "1. Calculate the test statistic (χ²) from your observed and expected data",
            "",
            "2. Determine degrees of freedom based on your study design",
            "",
            "3. Compare your χ² value to the critical value:",
            "   • If χ² > critical value: Reject null hypothesis (p < α)",
            "   • If χ² ≤ critical value: Fail to reject null hypothesis (p ≥ α)",
            "",
            "4. Reporting: Include χ² value, degrees of freedom, and p-value",
            "   • Example: \"χ²(2) = 9.21, p = 0.01\" for df=2 with significant finding",
            "",
            "Remember: Statistical significance doesn't always imply clinical relevance!"
        ]
        
        for i, line in enumerate(tutorial_text):
            draw_text(line, font_medium, BLACK, WIDTH // 2, 180 + i * 40, "center")
    
    # Navigation buttons
    back_button = pygame.Rect(50, HEIGHT - 100, 200, 50)
    next_button = pygame.Rect(WIDTH - 250, HEIGHT - 100, 200, 50)
    
    pygame.draw.rect(screen, LIGHT_BLUE, back_button, border_radius=5)
    pygame.draw.rect(screen, LIGHT_BLUE, next_button, border_radius=5)
    
    if tutorial_page > 0:
        draw_text("Previous", font_medium, BLACK, back_button.centerx, back_button.centery, "center")
    
    if tutorial_page < num_tutorial_pages - 1:
        draw_text("Next", font_medium, BLACK, next_button.centerx, next_button.centery, "center")
    else:
        draw_text("Start Simulation", font_medium, BLACK, next_button.centerx, next_button.centery, "center")
    
    # Page indicator
    draw_text(f"Page {tutorial_page + 1}/{num_tutorial_pages}", font_small, BLACK, WIDTH // 2, HEIGHT - 75, "center")

def draw_instructions_toggle(toggle_rect):
    """Draw a toggle button for instructions"""
    pygame.draw.rect(screen, LIGHT_BLUE if show_instructions else LIGHT_GRAY, toggle_rect)
    
    # Draw icon for info
    pygame.draw.rect(screen, BLACK, toggle_rect, width=2)
    draw_text("i", font_medium, BLACK, toggle_rect.centerx, toggle_rect.centery, "center")
    
    return toggle_rect

def draw_menu():
    """Draw the main menu screen"""
    screen.fill(WHITE)
    
    # Draw title
    title_rect = draw_text("Chi-Square Distribution Visualizer", font_title, BLUE, WIDTH // 2, HEIGHT // 4, "center")
    
    # Draw subtitle
    subtitle_rect = draw_text("For Alzheimer's Research Statistical Education", font_large, 
                             BLACK, WIDTH // 2, title_rect.bottom + 30, "center")
    
    # Draw description
    description = [
        "Understand Chi-square tests for categorical data analysis",
        "Perfect for analyzing genetic associations and population differences",
        "Explore the effects of degrees of freedom and significance levels"
    ]
    
    for i, line in enumerate(description):
        draw_text(line, font_medium, BLACK, WIDTH // 2, subtitle_rect.bottom + 50 + i * 40, "center")
    
    # Draw buttons
    start_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 100, 300, 60)
    tutorial_button = pygame.Rect(WIDTH // 2 - 150, HEIGHT // 2 + 200, 300, 60)
    
    pygame.draw.rect(screen, LIGHT_BLUE, start_button, border_radius=10)
    pygame.draw.rect(screen, LIGHT_GREEN, tutorial_button, border_radius=10)
    
    draw_text("Start Visualization", font_large, BLACK, start_button.centerx, start_button.centery, "center")
    draw_text("Tutorial", font_large, BLACK, tutorial_button.centerx, tutorial_button.centery, "center")
    
    # Footer
    draw_text("For Statistical Analysis in Alzheimer's Disease Research", font_small, GRAY, 
             WIDTH // 2, HEIGHT - 50, "center")
    
    return start_button, tutorial_button

def draw_game(layout, mouse_pos):
    """Draw the main visualization screen with improved responsive layout"""
    screen.fill(WHITE)
    
    # Draw header with clear separation
    header_bg = pygame.Rect(0, 0, WIDTH, layout["header_height"])
    pygame.draw.rect(screen, PANEL_BLUE, header_bg)
    draw_text("Chi-Square Distribution Visualizer", font_title, BLUE, WIDTH // 2, layout["header_height"] // 2, "center")
    
    # Draw back button
    back_button = pygame.Rect(30, 10, 100, 40)
    pygame.draw.rect(screen, LIGHT_RED, back_button, border_radius=5)
    draw_text("Menu", font_small, BLACK, back_button.centerx, back_button.centery, "center")
    
    # Draw results panel
    if has_updated:
        result_panel = pygame.Rect(
            layout["results"]["left"], 
            layout["results"]["top"], 
            layout["results"]["width"], 
            layout["results"]["height"]
        )
        draw_panel(result_panel, PANEL_YELLOW, border_radius=5, border_color=DARK_GRAY, border_width=1)
        
        # Draw results with proper spacing
        panel_width = layout["results"]["width"]
        df_pos_x = result_panel.left + 10
        draw_text(f"Degrees of Freedom: {int(degrees_of_freedom)}", font_result, BLACK, df_pos_x, result_panel.centery)
        
        critical_pos_x = result_panel.centerx
        draw_text(f"Critical Value: {critical_value:.3f}", font_result, BLACK, critical_pos_x, result_panel.centery)
        
        alpha_pos_x = result_panel.right - 10
        draw_text(f"α = {alpha_level:.3f}", font_result, BLACK, alpha_pos_x, result_panel.centery, "right")
    
    # Create and draw Chi-square plot
    chart_rect = pygame.Rect(
        layout["chart"]["left"], 
        layout["chart"]["top"], 
        layout["chart"]["width"], 
        layout["chart"]["height"]
    )
    draw_panel(chart_rect, WHITE, border_color=GRAY, border_width=2)
    
    try:
        chi_square_surface = create_chisquare_surface(
            degrees_of_freedom, alpha_level, max_x_value, num_distributions, show_critical,
            width=layout["chart"]["width"] - 20, 
            height=layout["chart"]["height"] - 20
        )
        screen.blit(chi_square_surface, (layout["chart"]["left"] + 10, layout["chart"]["top"] + 10))
    except Exception as e:
        # Fallback if plot creation fails
        pygame.draw.rect(screen, LIGHT_BLUE, (
            layout["chart"]["left"] + 10, 
            layout["chart"]["top"] + 10, 
            layout["chart"]["width"] - 20, 
            layout["chart"]["height"] - 20
        ))
        draw_text("Error creating Chi-square plot. Try adjusting parameters.", 
                 font_medium, RED, WIDTH // 2, layout["chart"]["top"] + layout["chart"]["height"] // 2, "center")
        print(f"Plot error: {e}")
    
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
        instruction_text = "Adjust the sliders to visualize different Chi-square distributions and critical values."
        
        # Center the text in the instruction panel
        draw_text(instruction_text, font_medium, BLACK, instruction_panel.centerx, instruction_panel.centery, "center")
        
    # Draw parameter panel
    param_panel = pygame.Rect(
        layout["panel"]["left"], 
        layout["panel"]["top"], 
        layout["panel"]["width"], 
        layout["panel"]["height"]
    )
    draw_panel(param_panel, PANEL_BLUE, border_color=DARK_GRAY, border_width=1)

    # Panel title
    title_y = param_panel.top + 20
    draw_text("Chi-Square Distribution Parameters", font_panel_title, BLACK, param_panel.centerx, title_y, "center")
    
    # Draw sliders with improved hover detection
    df_hover = draw_slider(slider_df, "Degrees of Freedom:", layout["panel"]["grid"], mouse_pos, "{:.0f}")
    alpha_hover = draw_slider(slider_alpha, "Significance Level (α):", layout["panel"]["grid"], mouse_pos, "{:.3f}")
    max_x_hover = draw_slider(slider_max_x, "X-Axis Range:", layout["panel"]["grid"], mouse_pos, "{:.0f}")
    num_dist_hover = draw_slider(slider_num_dist, "Compare Distributions:", layout["panel"]["grid"], mouse_pos, "{:.0f}")
    
    # Draw buttons
    pygame.draw.rect(screen, GREEN, update_button, border_radius=5)
    draw_text("Update Plot", font_medium, BLACK, update_button.centerx, update_button.centery, "center")
    
    toggle_text = "Hide Critical Values" if show_critical else "Show Critical Values"
    pygame.draw.rect(screen, LIGHT_BLUE, toggle_critical_button, border_radius=5)
    draw_text(toggle_text, font_medium, BLACK, toggle_critical_button.centerx, toggle_critical_button.centery, "center")
    
    # Add tooltips for interactive elements when hovering
    if (not df_hover and not alpha_hover and not max_x_hover and not num_dist_hover):
        # Only show tooltips if we're not already hovering over a slider
        handle_size = 20
        df_rect = pygame.Rect(slider_df["pos"][0] - handle_size//2, slider_df["pos"][1] - handle_size//2, handle_size, handle_size)
        alpha_rect = pygame.Rect(slider_alpha["pos"][0] - handle_size//2, slider_alpha["pos"][1] - handle_size//2, handle_size, handle_size)
        max_x_rect = pygame.Rect(slider_max_x["pos"][0] - handle_size//2, slider_max_x["pos"][1] - handle_size//2, handle_size, handle_size)
        num_dist_rect = pygame.Rect(slider_num_dist["pos"][0] - handle_size//2, slider_num_dist["pos"][1] - handle_size//2, handle_size, handle_size)
        
        if add_hover_tooltip(df_rect, "Number of categories - 1 in categorical data", mouse_pos):
            pass
        elif add_hover_tooltip(alpha_rect, "Probability of Type I error (false positive)", mouse_pos):
            pass
        elif add_hover_tooltip(max_x_rect, "Adjust visible range of χ² values", mouse_pos):
            pass
        elif add_hover_tooltip(num_dist_rect, "Show multiple distributions for comparison", mouse_pos):
            pass
        elif add_hover_tooltip(update_button, "Update the plot with current parameters", mouse_pos):
            pass
        elif add_hover_tooltip(toggle_critical_button, "Show/hide critical value and rejection region", mouse_pos):
            pass
    
    return back_button, toggle_rect

def update_parameters():
    """Update the visualization parameters based on slider values"""
    global degrees_of_freedom, alpha_level, max_x_value, num_distributions, critical_value, has_updated
    
    # Ensure degrees of freedom is at least 1
    degrees_of_freedom = max(1, int(slider_df["value"]))
    
    # Ensure alpha is between 0.01 and 0.2
    alpha_level = max(0.01, min(0.2, slider_alpha["value"]))
    
    # Update max x value
    max_x_value = slider_max_x["value"]
    
    # Update number of distributions
    num_distributions = int(slider_num_dist["value"])
    
    # Recalculate critical value
    critical_value = stats.chi2.ppf(1 - alpha_level, degrees_of_freedom)
    
    has_updated = True
    debug_info(f"Updated parameters: df={degrees_of_freedom}, alpha={alpha_level:.3f}, critical={critical_value:.3f}")

# Initialize layout at startup
layout = calculate_layout()

# Main game loop
running = True
clock = pygame.time.Clock()

# Calculate minimum required height
def calculate_min_height():
    """Calculate minimum required height for all UI elements"""
    header_height = HEADER_HEIGHT
    results_height = 40
    chart_min_height = 300
    instructions_height = 40
    panel_min_height = 120
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
            calculated_min_width = 2 * (LABEL_COLUMN_WIDTH + SLIDER_MIN_WIDTH + VALUE_COLUMN_WIDTH) + PANEL_MARGIN*3 + H_MARGIN*4
            calculated_min_height = calculate_min_height()
            
            # Handle window resizing with dynamic minimum dimensions
            WIDTH = max(event.w, calculated_min_width, MIN_WIDTH)
            HEIGHT = max(event.h, calculated_min_height)
            screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            
            # Recalculate layout with new dimensions
            layout = calculate_layout()
            
            # Prevent any current slider dragging operations from continuing
            for slider in [slider_df, slider_alpha, slider_max_x, slider_num_dist]:
                slider["dragging"] = False
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                # Return to menu when escape is pressed
                if state != MENU:
                    state = MENU
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                
                if state == MENU:
                    # Check for button clicks in menu
                    start_button, tutorial_button = draw_menu()
                    
                    if start_button.collidepoint(mouse_pos):
                        state = GAME
                        update_parameters()
                    elif tutorial_button.collidepoint(mouse_pos):
                        state = TUTORIAL
                        tutorial_page = 0
                
                elif state == TUTORIAL:
                    # Check for button clicks in tutorial
                    back_button = pygame.Rect(50, HEIGHT - 100, 200, 50)
                    next_button = pygame.Rect(WIDTH - 250, HEIGHT - 100, 200, 50)
                    menu_button = pygame.Rect(30, 10, 100, 40)
                    
                    if menu_button.collidepoint(mouse_pos):
                        state = MENU
                    elif back_button.collidepoint(mouse_pos) and tutorial_page > 0:
                        tutorial_page -= 1
                    elif next_button.collidepoint(mouse_pos):
                        if tutorial_page < num_tutorial_pages - 1:
                            tutorial_page += 1
                        else:
                            state = GAME
                            update_parameters()
                
                elif state == GAME:
                    # Get UI elements from the draw function
                    back_button, toggle_rect = draw_game(layout, mouse_pos)
                    
                    # Check for instruction toggle
                    if toggle_rect.collidepoint(mouse_pos):
                        show_instructions = not show_instructions
                        
                    # Check for button clicks in game
                    if back_button.collidepoint(mouse_pos):
                        state = MENU
                    elif update_button.collidepoint(mouse_pos):
                        update_parameters()
                    elif toggle_critical_button.collidepoint(mouse_pos):
                        show_critical = not show_critical
                        update_parameters()
                    
                    # Check for slider interactions
                    for slider in [slider_df, slider_alpha, slider_max_x, slider_num_dist]:
                        slider_rect = pygame.Rect(
                            slider["pos"][0] - 5, 
                            slider["pos"][1] - 10, 
                            slider["width"] + 10, 
                            20
                        )
                        if slider_rect.collidepoint(mouse_pos):
                            slider["dragging"] = True
                            # Update the value immediately based on click position
                            slider["value"] = (mouse_pos[0] - slider["pos"][0]) / slider["width"] * (slider["max"] - slider["min"]) + slider["min"]
                            slider["value"] = max(slider["min"], min(slider["max"], slider["value"]))
        
        elif event.type == pygame.MOUSEBUTTONUP:
            # Stop dragging
            for slider in [slider_df, slider_alpha, slider_max_x, slider_num_dist]:
                if slider["dragging"]:
                    slider["dragging"] = False
                    # No automatic update on release to give better control
        
        elif event.type == pygame.MOUSEMOTION:
            # Update sliders
            for slider in [slider_df, slider_alpha, slider_max_x, slider_num_dist]:
                if slider["dragging"]:
                    # Calculate new value based on mouse position
                    new_val = (mouse_pos[0] - slider["pos"][0]) / slider["width"] * (slider["max"] - slider["min"]) + slider["min"]
                    # Clamp value to slider range
                    slider["value"] = max(slider["min"], min(slider["max"], new_val))
    
    # Draw current state
    if state == MENU:
        draw_menu()
    elif state == TUTORIAL:
        draw_tutorial()
        # Draw back button in tutorial mode
        back_button = pygame.Rect(30, 10, 100, 40)
        pygame.draw.rect(screen, LIGHT_RED, back_button, border_radius=5)
        draw_text("Menu", font_small, BLACK, back_button.centerx, back_button.centery, "center")
    elif state == GAME:
        draw_game(layout, mouse_pos)
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()